"""
Monster Detection Engine for MapleStoryGo.

Adapted from the monster detection logic in:
  MapleStoryAutoLevelUp-main/src/engine/MapleStoryAutoLevelUp.py

Core features:
  - Loads monster template images from monster/<name>/ directories
  - Captures live game frames from the "MapleStoryGo" window
  - Performs template matching with mask support
  - Applies Non-Maximum Suppression (NMS) to filter overlapping detections
  - Supports multiple detection modes: color, grayscale, contour_only, template_free
  - Provides real-time debug visualization
"""
import time
import glob
import os
import threading
import traceback

import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.common import (
    load_yaml, load_image, get_mask, find_pattern_sqdiff,
    nms, draw_rectangle, screenshot, to_opencv_hsv,
    activate_game_window, is_mac, is_windows,
    get_minimap_loc_size, mask_route_colors
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.engine.HealthMonitor import HealthMonitor


class MonsterDetector:
    """
    Detects monsters in the MapleStoryGo game window using template matching.

    Usage:
        detector = MonsterDetector(cfg)
        detector.load_monster_images()
        detector.start()
        # Frames are processed automatically in the main loop
    """

    def __init__(self, cfg: dict, test_image: str = ""):
        """
        Initialize the monster detector.

        Args:
            cfg: Configuration dictionary.
            test_image: Path to a test image for offline debugging.
        """
        self.cfg = cfg
        self.test_image = test_image

        # Monster data
        self.monsters_info: dict = {}   # {name: [(img, mask), ...]}
        self.monster_cfg: dict = {}     # {name: {diff_thres, scales, ...}}
        self.monsters: list = []        # Detected monsters in current frame

        # Frame data
        self.frame = None               # Raw captured frame
        self.img_frame = None           # Working-size game frame (BGR)
        self.img_frame_gray = None      # Grayscale version
        self.img_frame_debug = None     # Debug visualization frame
        self.img_display = None         # Pre-scaled frame for display thread

        # Player location — detected via title template matching
        self.loc_player = (0, 0)
        self.loc_title = None  # Last known title position (for fast local search)
        self._title_last_seen_frame = -1
        self._title_state_lock = threading.Lock()
        self._title_candidates_snapshot = []
        self._manual_title_locked = False
        self._selected_title_loc = None
        self._pending_title_select_loc = None
        self._title_lock_miss_count = 0
        self._title_fixed = False       # F12 toggles: freeze the title box (and player pos)

        # Minimap
        self.loc_minimap = (0, 0)
        self.img_minimap = None
        self.loc_player_minimap = None
        self._dot_last_pos = None            # 上一次绿点位置（判断是否卡住）
        self._dot_last_move_time = time.time()

        # Route following
        self.img_map = None             # Global stitched map (map.png)
        self.img_routes = []            # Route images (route*.png)
        self.img_route = None           # Current route
        self.idx_routes = 0             # Current route index
        self.color_code = {}            # RGB→command mapping
        self.color_code_up_down = {}    # Up/down RGB→command mapping
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self.loc_player_global = (0.0, 0.0)
        self.loc_minimap_global = (0, 0)
        self._last_pmm = None   # Last player minimap position (for displacement tracking)
        self.is_on_ladder = False
        self._route_miss_frames = 0  # consecutive frames with no nearby route color

        # Stuck recovery (walk left when the character hasn't moved for a while)
        self._stuck_last_pos = None
        self._stuck_last_move_time = time.time()
        self._unstick_until = 0.0
        self._unstick_dir = "left"
        self._last_walk_dir = "left"   # 最后一次实际移动的方向（卡住反方向用）
        self._kb = None  # Keyboard controller (set externally)
        self.health_monitor = None  # Health monitor (initialized in start())

        # Hunt-mode search state (used when no monsters are detected)
        self._hunt_no_monster_frames = 0
        self._hunt_search_dir = "left"
        self._hunt_search_flip_frame = 0
        self._hunt_search_sweeps = 0
        self._hunt_search_pause_until = 0
        self._hunt_search_anchor_x = None
        self._hunt_search_last_progress_x = None
        self._hunt_search_stall_frames = 0
        self._hunt_search_last_reason = "idle"

        # In-place attack "walk about" state machine (prevents attack lockout)
        self._in_place_state = "attack"       # "attack" | "pre_walk" | "walk_away" | "walk_back"
        self._in_place_state_frame = 0        # frame counter for current state
        self._in_place_walk_dir = "left"      # fallback direction when no facing
        self._in_place_backward_dir = "left"  # backward dir (opposite of facing)
        self._in_place_start_x = None         # minimap X at walk-about start (for return-to-start)
        self._in_place_no_monster_frames = 0  # consecutive frames without a monster (for idle AoE)

        # Load title template for player detection
        self.img_title = None
        self.img_title_gray = None
        if os.path.exists("character/title.png"):
            self.img_title = load_image("character/title.png")
            self.img_title_gray = load_image("character/title.png", cv2.IMREAD_GRAYSCALE)
            logger.info(f"  Loaded title template: {self.img_title.shape[1]}x{self.img_title.shape[0]}")
        else:
            logger.warning("  No character/title.png found, player will stay at frame center")

        # Load character appearance template (unique anchor used to disambiguate
        # the shared 称号 when it is lost or collides with other players).
        self.img_appearance = None
        self.img_appearance_gray = None
        self.img_appearance_mask = None
        appearance_path = self.cfg.get("appearance", {}).get("template", "character/fashi.png")
        if os.path.exists(appearance_path):
            raw = load_image(appearance_path, cv2.IMREAD_UNCHANGED)
            if len(raw.shape) == 3 and raw.shape[2] == 4:
                self.img_appearance = raw[:, :, :3].copy()
                self.img_appearance_mask = np.where(raw[:, :, 3] > 128, 255, 0).astype(np.uint8)
            else:
                self.img_appearance = raw
                self.img_appearance_mask = self._build_appearance_mask(raw)
            self.img_appearance_gray = cv2.cvtColor(self.img_appearance, cv2.COLOR_BGR2GRAY)
            logger.info(
                f"  Loaded appearance template: {self.img_appearance.shape[1]}x{self.img_appearance.shape[0]}"
            )
        else:
            logger.warning(f"  No appearance template at {appearance_path}, title-only localization")

        # Debug state for the appearance anchor (drawn on the overlay).
        self.appearance_result = None    # {"loc","size","score","feet"}
        self.appearance_score = None     # 外观匹配分（0~1），始终显示用
        self.appearance_override = False
        self._appearance_last_pos = None   # appearance's own tracked feet position
        self._appearance_last_seen_frame = -1  # last frame appearance matched confidently
        self.title_sim = None   # last title match similarity (0~1), for display

        # Manual minimap selection (drag a box in the display window to calibrate).
        self._minimap_select_mode = False
        self._mm_drag_start = None      # display-space (x, y)
        self._mm_drag_end = None        # display-space (x, y)
        self._mm_hover = None           # display-space (x, y) — for the magnifier
        self._manual_minimap_rect = None   # full-frame (x, y, w, h)

        # Timers
        self.t_last_frame = time.time()
        self.frame_count = 0
        self.fps = 0
        self.display_fps = 0
        self.frame_seq = 0
        self.best_score = 1.0       # Best (lowest) match score this frame

        # Flags
        self.is_terminated = False
        self.is_frame_done = False
        self.is_show_debug = not cfg.get("disable_viz", False)

        # Capture thread
        self.capture = None

    # -------------------------------------------------------------------
    # Monster image loading
    # -------------------------------------------------------------------

    def load_monster_images(self) -> bool:
        """
        Load monster template images from monster/ directories.

        For each monster directory, loads all PNG images, creates
        green-background masks, and generates horizontally-flipped
        copies for detecting facing-left monsters.

        The expected directory layout:
            monster/
              OrangeMushroom/
                stand_0.png
                move_0.png
                ...

        Returns:
            True if at least one monster was loaded successfully.
        """
        # Load specified monsters from config, or scan all directories.
        # monsters can be:
        #   - dict: {name: {diff_thres, scales, ...}}  (per-monster config)
        #   - list: [name, ...]                        (all use global defaults)
        specified = self.cfg["monster_detect"].get("monsters", [])
        self.monster_cfg = {}
        if isinstance(specified, dict):
            monster_names = list(specified.keys())
            for name, m_cfg in specified.items():
                self.monster_cfg[name] = m_cfg if isinstance(m_cfg, dict) else {}
        else:
            monster_names = list(specified)

        if monster_names:
            monster_dirs = [os.path.join("monster", m) for m in monster_names
                            if os.path.isdir(os.path.join("monster", m))]
            if not monster_dirs:
                logger.error(f"Specified monster dirs not found: {monster_names}")
                return False
        else:
            monster_dirs = [d for d in glob.glob("monster/*") if os.path.isdir(d)]
            if not monster_dirs:
                logger.error("No monster directories found in monster/")
                return False

        for monster_dir in monster_dirs:
            monster_name = os.path.basename(monster_dir)
            imgs = []

            # Load monster images: try <name>*.png first (reference pattern),
            # fall back to all *.png in the directory
            png_files = sorted(glob.glob(os.path.join(monster_dir, f"{monster_name}*.png")))
            if not png_files:
                png_files = sorted(glob.glob(os.path.join(monster_dir, "*.png")))
            if not png_files:
                logger.warning(f"No usable PNG images in {monster_dir}, skipping.")
                continue

            # Load all images + horizontally flipped copies (for both directions)
            for file in png_files:
                try:
                    img, mask = self._load_monster_template(file)
                    imgs.append((img, mask))
                    # Flipped copy for right-facing monsters
                    img_flip = cv2.flip(img, 1)
                    mask_flip = cv2.flip(mask, 1) if mask is not None else None
                    imgs.append((img_flip, mask_flip))
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")

            if not imgs:
                continue

            if imgs:
                self.monsters_info[monster_name] = imgs
                logger.info(
                    f"  Loaded {len(png_files)} images for '{monster_name}' "
                    f"({len(imgs)} with flips)"
                )
            else:
                logger.warning(f"No valid images in {monster_dir}")

        if not self.monsters_info:
            logger.error("Failed to load any monster images.")
            return False

        logger.info(f"Total monsters loaded: {list(self.monsters_info.keys())}")
        return True

    def _load_monster_template(self, file: str) -> tuple:
        """
        Load a monster template image and create its mask.

        Supports three formats:
          - RGBA with alpha channel: alpha > 128 = monster, alpha == 0 = ignore
          - BGR with green (0,255,0) background: green pixels ignored
          - BGR without special background: no mask, match full image

        Returns:
            (img_bgr, mask) — mask is None if no masking needed
        """
        raw = load_image(file, cv2.IMREAD_UNCHANGED)

        if len(raw.shape) == 3 and raw.shape[2] == 4:
            # RGBA image — use alpha channel as mask
            alpha = raw[:, :, 3]
            mask = np.where(alpha > 128, 255, 0).astype(np.uint8)
            img = raw[:, :, :3].copy()
            img[mask == 0] = [0, 0, 0]
        else:
            img = raw[:, :, :3] if len(raw.shape) == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            # Check if green-background (reference project style)
            green_px = np.sum(np.all(img == [0, 255, 0], axis=2))
            total_px = img.shape[0] * img.shape[1]
            if green_px > total_px * 0.1:  # >10% green pixels
                mask = get_mask(img, (0, 255, 0))
            else:
                mask = None  # No mask — match full image

        return img, mask

    # -------------------------------------------------------------------
    # Frame capture
    # -------------------------------------------------------------------

    def start_capture(self):
        """Initialize and start the game window capturer."""
        self.capture = GameWindowCapturor(self.cfg, self.test_image)

    def get_img_frame(self) -> np.ndarray:
        """
        Get the latest processed game frame.

        Returns:
            BGR numpy array of the game frame (working size), or None.
        """
        if self.capture is None:
            self.start_capture()

        frame = self.capture.get_frame()
        if frame is None:
            return None

        self.frame = frame
        return frame

    # -------------------------------------------------------------------
    # Monster detection (core logic)
    # -------------------------------------------------------------------

    def get_monsters_in_range(
        self,
        top_left: tuple,
        bottom_right: tuple
    ) -> list:
        """
        Detect monsters within a specified region of the game frame.

        This is the core detection function. It iterates through all
        loaded monster templates and performs template matching within
        the region of interest (ROI).

        Supports multiple detection modes:
          - "color": Full-color template matching with mask
          - "grayscale": Grayscale template matching with mask
          - "contour_only": Black contour matching (blurred)
          - "template_free": Connected-component-based (no templates)

        Args:
            top_left: (x0, y0) top-left corner of the search region.
            bottom_right: (x1, y1) bottom-right corner.

        Returns:
            List of detected monster dicts, each with:
              - "name": monster name
              - "position": (x, y) top-left of bounding box
              - "size": (h, w) of bounding box
              - "score": match quality (lower = better for SQDIFF)
        """
        x0, y0 = top_left
        x1, y1 = bottom_right

        # Clamp to frame bounds
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.img_frame.shape[1], x1)
        y1 = min(self.img_frame.shape[0], y1)

        if x1 <= x0 or y1 <= y0:
            return []

        img_roi = self.img_frame[y0:y1, x0:x1]

        # Shift player location into ROI coordinate system
        px, py = self.loc_player
        px_in_roi = px - x0
        py_in_roi = py - y0

        # Define player character bounding box in ROI coords (to exclude)
        char_half_w = self.cfg["character"]["width"] // 2
        char_half_h = self.cfg["character"]["height"] // 2
        char_x_min = max(0, px_in_roi - char_half_w)
        char_x_max = min(img_roi.shape[1], px_in_roi + char_half_w)
        char_y_min = max(0, py_in_roi - char_half_h)
        char_y_max = min(img_roi.shape[0], py_in_roi + char_half_h)

        mode = self.cfg["monster_detect"]["mode"]
        monsters = []
        self.best_score = 1.0  # Reset each frame

        if mode == "template_free":
            monsters = self._detect_template_free(
                img_roi, x0, y0, char_x_min, char_y_min,
                char_x_max, char_y_max
            )
        elif mode in ("color", "grayscale", "contour_only"):
            roi_h, roi_w = img_roi.shape[:2]

            for monster_name, monster_imgs in self.monsters_info.items():
                for img_monster, mask_monster in monster_imgs:
                    th, tw = img_monster.shape[:2]
                    if th > roi_h or tw > roi_w or tw < 10 or th < 10:
                        continue

                    found = self._match_template(
                        img_roi,
                        img_monster, mask_monster,
                        x0, y0, monster_name, mode,
                        char_x_min, char_y_min, char_x_max, char_y_max
                    )
                    monsters.extend(found)
        else:
            logger.error(f"Unsupported detection mode: {mode}")
            return []

        # Apply Non-Maximum Suppression
        monsters = nms(monsters, iou_threshold=0.4)

        # Also detect via enemy HP bar if enabled
        if self.cfg["monster_detect"].get("with_enemy_hp_bar", False):
            hp_bar_monsters = self._detect_by_hp_bar(img_roi, x0, y0)
            monsters.extend(hp_bar_monsters)

        return monsters

    @staticmethod
    def _color_spread(patch: np.ndarray, mask: np.ndarray) -> float:
        """
        Mean max-min channel spread over masked pixels.

        0 = neutral gray/white (clouds, sky); high = colorful (monster).
        Used to reject cloud false positives that pass the SQDIFF threshold.
        """
        if mask is None or patch is None or patch.size == 0:
            return 0.0
        sel = mask > 0
        if not np.any(sel):
            return 0.0
        px = patch[sel].astype(int)
        b, g, r = px[:, 0], px[:, 1], px[:, 2]
        spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        return float(spread.mean())

    def _match_template(
        self,
        img_roi: np.ndarray,
        img_monster: np.ndarray,
        mask_monster: np.ndarray,
        x0: int, y0: int,
        monster_name: str,
        mode: str,
        char_x_min: int, char_y_min: int,
        char_x_max: int, char_y_max: int
    ) -> list:
        """
        Match a single monster template against an ROI.
        Uses multi-scale matching and returns best matches.
        """
        monsters = []
        m_cfg = self.monster_cfg.get(monster_name, {})
        try:
            diff_thres = float(m_cfg.get("diff_thres", self.cfg["monster_detect"]["diff_thres"]))
        except (TypeError, ValueError):
            diff_thres = float(self.cfg["monster_detect"].get("diff_thres", 0.3))
        h, w = img_monster.shape[:2]

        # Multi-scale matching (per-monster scales override the global default)
        scales = m_cfg.get("scales", self.cfg["monster_detect"].get("scales", [1.0]))

        for scale in scales:
            if scale == 1.0:
                tmpl, msk, th, tw = img_monster, mask_monster, h, w
            else:
                tw = max(8, int(w * scale))
                th = max(8, int(h * scale))
                tmpl = cv2.resize(img_monster, (tw, th), interpolation=cv2.INTER_NEAREST)
                if mask_monster is not None:
                    msk = cv2.resize(mask_monster, (tw, th), interpolation=cv2.INTER_NEAREST)
                else:
                    msk = None

            try:
                if mode == "contour_only":
                    mask_pat = np.all(tmpl == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi[char_y_min:char_y_max, char_x_min:char_x_max] = 0
                    blur = self.cfg["monster_detect"].get("contour_blur", 5)
                    res = cv2.matchTemplate(
                        cv2.GaussianBlur(mask_roi, (blur, blur), 0),
                        cv2.GaussianBlur(mask_pat, (blur, blur), 0),
                        cv2.TM_SQDIFF_NORMED
                    )
                elif mode == "grayscale":
                    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    if msk is not None:
                        res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_SQDIFF_NORMED, mask=msk)
                    else:
                        res = cv2.matchTemplate(roi_gray, tmpl_gray, cv2.TM_SQDIFF_NORMED)
                elif mode == "color":
                    if msk is not None:
                        res = cv2.matchTemplate(img_roi, tmpl, cv2.TM_SQDIFF_NORMED, mask=msk)
                    else:
                        res = cv2.matchTemplate(img_roi, tmpl, cv2.TM_SQDIFF_NORMED)
                else:
                    return []
            except cv2.error:
                continue

            # Track best score
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < self.best_score:
                self.best_score = min_val

            # Collect matches below threshold
            match_locations = np.where(res <= diff_thres)
            min_spread = float(m_cfg.get(
                "min_color_spread",
                self.cfg["monster_detect"].get("min_color_spread", 30)
            ))
            count = 0
            for pt in zip(*match_locations[::-1]):
                if count >= 50:
                    break
                # Color guard: reject neutral gray/white patches (clouds, sky)
                # by requiring enough color spread over the monster's pixels.
                if msk is not None and min_spread > 0:
                    patch = img_roi[pt[1]:pt[1] + th, pt[0]:pt[0] + tw]
                    if self._color_spread(patch, msk) < min_spread:
                        continue
                monsters.append({
                    "name": monster_name,
                    "position": (pt[0] + x0, pt[1] + y0),
                    "size": (th, tw),
                    "score": res[pt[1], pt[0]],
                })
                count += 1

        return monsters

    def _detect_template_free(
        self,
        img_roi: np.ndarray,
        x0: int, y0: int,
        char_x_min: int, char_y_min: int,
        char_x_max: int, char_y_max: int
    ) -> list:
        """
        Template-free monster detection using connected components of
        black pixels (monster outlines in the game).

        This is useful when monster templates are not available or
        when you want to detect any dark shape as a potential monster.
        """
        # Generate mask where pixel is exactly black
        black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

        # Exclude player character region
        black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

        # Morphological close to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        # Draw character bounding box for debug
        draw_rectangle(
            self.img_frame_debug,
            (char_x_min + x0, char_y_min + y0),
            (self.cfg["character"]["height"], self.cfg["character"]["width"]),
            (255, 0, 0), "Character Box"
        )

        # Find connected components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            closed_mask, connectivity=8
        )

        monsters = []
        min_area = 1000
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > min_area:
                monsters.append({
                    "name": "Unknown",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        return monsters

    def _detect_by_hp_bar(
        self,
        img_roi: np.ndarray,
        x0: int, y0: int
    ) -> list:
        """
        Detect monsters by finding enemy HP bar colors in the ROI.
        """
        hp_color = np.array(self.cfg["monster_detect"]["hp_bar_color"])
        tol = int(self.cfg["monster_detect"].get("hp_bar_tolerance", 30))
        lower = np.clip(hp_color - tol, 0, 255)
        upper = np.clip(hp_color + tol, 0, 255)
        mask = cv2.inRange(img_roi, lower, upper)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        monsters = []
        # Estimate monster height from loaded templates
        min_monster_h = min(
            img.shape[0]
            for imgs in self.monsters_info.values()
            for img, _ in imgs
        ) if self.monsters_info else 100

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 3:  # Noise filter
                continue

            # Guess a monster bounding box below the HP bar
            monster_y = y + 10
            monster_x = max(0, x)
            monster_w = 70
            monster_h = min_monster_h

            monsters.append({
                "name": "Health Bar",
                "position": (x0 + monster_x, y0 + monster_y),
                "size": (monster_h, monster_w),
                "score": 1.0,
            })

        return monsters

    # -------------------------------------------------------------------
    # Debug visualization
    # -------------------------------------------------------------------

    def _draw_detection_debug(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        monsters: list
    ):
        """Draw fresh monster detection boxes (called only on detection frames)."""
        if self.img_frame_debug is None:
            return

        for monster in monsters:
            if monster["name"] == "Health Bar":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            draw_rectangle(
                self.img_frame_debug,
                monster["position"],
                monster["size"],
                color,
                f"{monster['name']}:{monster['score']:.2f}",
                thickness=2, text_height=0.5
            )

    def _draw_search_region(self, x0: int, y0: int, x1: int, y1: int):
        """Draw the search region box on the debug frame."""
        if self.img_frame_debug is None:
            return
        # Bright cyan border, thicker for visibility
        draw_rectangle(
            self.img_frame_debug,
            (x0, y0), (y1 - y0, x1 - x0),
            (255, 255, 0), "Search", thickness=2, text_height=0.5
        )

    def _fallback_player_location(self):
        """Keep last valid player position, or fall back to frame center."""
        if self.loc_player != (0, 0):
            return
        h, w = self.img_frame.shape[:2]
        self.loc_player = (w // 2, h // 2)

    def _build_appearance_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Build a mask for the appearance template.

        fashi.png is a raw screenshot crop (no alpha), so the surrounding map
        background is baked in. We keep a central ellipse that covers the
        character body and ignore the corners (sky/clouds, grass) that change
        as the camera scrolls.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        rx = max(2, int(w * 0.46))
        ry = max(2, int(h * 0.46))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return mask

    def _detect_player_by_appearance(self):
        """
        Locate the player by matching the character appearance template.

        The camera follows the player, so the character stays near screen
        center. We search a window around the last known player position
        (falling back to screen center) to stay robust against the background
        baked into fashi.png.

        Returns:
            (loc_player_feet, score) or (None, None) if no confident match.
        """
        if self.img_appearance is None or self.img_frame is None:
            return None, None

        cfg = self.cfg.get("appearance", {})
        # Match in COLOR (BGR), not grayscale — the character's color (e.g. blue
        # robe) is a strong distinguishing feature that grayscale discards.
        tmpl = self.img_appearance
        mask = self.img_appearance_mask
        th, tw = tmpl.shape[:2]
        fh, fw = self.img_frame.shape[:2]
        if th > fh or tw > fw:
            return None, None

        search_radius = int(cfg.get("search_radius", 140))
        center_radius = int(cfg.get("center_fallback_radius", 260))
        match_threshold = float(cfg.get("match_threshold", 0.62))

        anchor = self.loc_player
        if self._appearance_last_pos is not None:
            # Appearance is primary: search around its OWN last position,
            # NOT around the title — so a wrong title can't drag it off.
            anchor = self._appearance_last_pos
            radius = search_radius
        else:
            # No appearance history yet: search around screen center
            # (camera follows the player), independent of the title.
            anchor = (fw // 2, fh // 2)
            radius = center_radius

        x0 = max(0, anchor[0] - radius)
        y0 = max(0, anchor[1] - radius)
        x1 = min(fw, anchor[0] + radius + tw)
        y1 = min(fh, anchor[1] + radius + th)
        if x1 - x0 < tw or y1 - y0 < th:
            return None, None
        roi = self.img_frame[y0:y1, x0:x1]

        best = None
        for tmpl_i, mask_i in ((tmpl, mask), (cv2.flip(tmpl, 1), cv2.flip(mask, 1))):
            res = cv2.matchTemplate(roi, tmpl_i, cv2.TM_CCORR_NORMED, mask=mask_i)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if best is None or max_val > best["score"]:
                best = {"score": float(max_val), "loc": (max_loc[0] + x0, max_loc[1] + y0)}

        if best is None:
            self.appearance_result = None
            self.appearance_score = None
            return None, None

        # 始终记录外观匹配分（右下角显示用）。
        self.appearance_score = float(best["score"])

        # 纯分数阈值：>= match_threshold 用外观定位，< 用称号。
        score_threshold = float(cfg.get("match_threshold", 0.85))
        if best["score"] < score_threshold:
            self.appearance_result = None
            return None, None

        offset = cfg.get("feet_offset", [0, -6])
        feet = (
            best["loc"][0] + tw // 2 + int(offset[0]),
            best["loc"][1] + th + int(offset[1]),
        )
        self.appearance_result = {
            "loc": best["loc"],
            "size": (th, tw),
            "score": best["score"],
            "feet": feet,
        }
        self._appearance_last_pos = feet
        return feet, best["score"]

    def _apply_appearance_anchor(self):
        """
        Appearance is the PRIMARY anchor; title is the fallback.

        When the appearance matches confidently, use its position; otherwise
        keep the title-derived position (set by _detect_player_by_title).
        """
        cfg = self.cfg.get("appearance", {})
        if not cfg.get("enable", False) or self.img_appearance is None:
            return
        # Respect F12: when the user has frozen the box, do not override it.
        if self._title_fixed:
            return

        self.appearance_override = False
        feet, score = self._detect_player_by_appearance()
        if feet is None:
            # 外观匹配不到（分数 < 阈值）：立即回退称号（_detect_player_by_title 已设置 loc_player）。
            if self.frame_count % 30 == 0:
                logger.info(
                    f"[Appearance] FAILED ap_score={getattr(self, 'appearance_score', None)} → 回退称号"
                )
            return

        self.appearance_override = True
        self.loc_player = feet
        if self.frame_count % 30 == 0:
            logger.info(
                f"[Appearance] Using appearance at {feet} (score={score:.2f})"
            )

    def _collect_title_candidates(self, img_roi, img_pat, max_candidates: int, score_threshold: float):
        """Collect multiple non-overlapping title candidates from a full-frame template match."""
        h_img, w_img = img_roi.shape[:2]
        h_pat, w_pat = img_pat.shape[:2]
        if h_pat > h_img or w_pat > w_img:
            return []

        result = cv2.matchTemplate(img_roi, img_pat, cv2.TM_SQDIFF_NORMED)
        result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)
        work = result.copy()
        suppress_x = max(12, w_pat // 2)
        suppress_y = max(8, h_pat // 2)

        candidates = []
        for _ in range(max_candidates):
            min_val, _, min_loc, _ = cv2.minMaxLoc(work)
            if min_val > score_threshold:
                break
            x, y = int(min_loc[0]), int(min_loc[1])
            candidates.append({
                "loc": (x, y),
                "score": float(min_val),
                "cost": float(min_val),
                "box": (x, y, w_pat, h_pat),
            })
            x0 = max(0, x - suppress_x)
            x1 = min(work.shape[1], x + suppress_x + 1)
            y0 = max(0, y - suppress_y)
            y1 = min(work.shape[0], y + suppress_y + 1)
            work[y0:y1, x0:x1] = 1.0

        return candidates

    def _update_title_candidates_snapshot(self, candidates, template_size):
        """Store the latest title candidates for debug rendering and click selection."""
        tw, th = template_size
        normalized = []
        with self._title_state_lock:
            selected_loc = self._selected_title_loc
            locked = self._manual_title_locked
        for idx, cand in enumerate(sorted(candidates, key=lambda c: c.get("score", 1.0))):
            x, y = cand["loc"]
            is_selected = False
            if locked and selected_loc is not None:
                is_selected = abs(x - selected_loc[0]) <= max(12, tw // 2) and abs(y - selected_loc[1]) <= max(8, th // 2)
            normalized.append({
                "idx": idx + 1,
                "loc": (int(x), int(y)),
                "box": (int(x), int(y), int(tw), int(th)),
                "score": float(cand.get("score", 1.0)),
                "src": cand.get("src", "cand"),
                "selected": is_selected,
            })
        with self._title_state_lock:
            self._title_candidates_snapshot = normalized

    def request_title_selection_from_display(self, display_x: int, display_y: int) -> bool:
        """Queue manual title selection by clicking a candidate box in the display window."""
        scale = float(self.cfg.get("monster_detect", {}).get("display_scale", 0.55))
        scale = max(scale, 0.01)
        raw_x = int(round(display_x / scale))
        raw_y = int(round(display_y / scale))

        with self._title_state_lock:
            hits = []
            for cand in self._title_candidates_snapshot:
                x, y, w, h = cand["box"]
                if x <= raw_x <= x + w and y <= raw_y <= y + h:
                    hits.append(cand)
            if not hits:
                return False
            chosen = min(hits, key=lambda c: c["score"])
            self._pending_title_select_loc = chosen["loc"]

        logger.info(
            f"[Title] Queued manual selection idx={chosen['idx']} loc={chosen['loc']} score={chosen['score']:.4f}"
        )
        return True

    def clear_title_selection(self):
        """Clear the manual title lock and return to automatic title selection."""
        with self._title_state_lock:
            self._manual_title_locked = False
            self._selected_title_loc = None
            self._pending_title_select_loc = None
            self._title_lock_miss_count = 0
        self.loc_title = None
        self._title_last_seen_frame = -1
        logger.info("[Title] Cleared manual title selection")

    def _draw_title_candidates(self):
        """Draw title candidates and manual selection state on the debug frame."""
        if self.img_frame_debug is None:
            return

        with self._title_state_lock:
            candidates = list(self._title_candidates_snapshot)
            locked = self._manual_title_locked
            miss_count = self._title_lock_miss_count

        for cand in candidates:
            x, y, w, h = cand["box"]
            color = (0, 255, 0) if cand.get("selected") else (0, 255, 255)
            # 称号用的是 SQDIFF（0=完美，越小越好），转成 0~1 相似度（越大越好），跟外观一致。
            sim = max(0.0, min(1.0, 1.0 - cand['score']))
            label = f"T{cand['idx']}:{sim:.2f}"
            cv2.rectangle(self.img_frame_debug, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                self.img_frame_debug,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                color,
                2,
                cv2.LINE_AA,
            )

        status = f"TitleSel: {'LOCKED' if locked else 'AUTO'} | LClick select | RClick/C clear | miss={miss_count}"
        cv2.putText(
            self.img_frame_debug,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255) if locked else (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _detect_player_by_title(self):
        """Detect player location by matching the title template (at character feet)."""
        if self._title_fixed:
            # Frozen: keep the current title box / player position, skip detection.
            return

        if self.img_title is None:
            self._fallback_player_location()
            return

        title_cfg = self.cfg.get("title", {})
        if not title_cfg.get("enable", True):
            self._fallback_player_location()
            return

        title_offset = title_cfg.get("offset", [0, -60])
        if not isinstance(title_offset, (list, tuple)) or len(title_offset) < 2:
            title_offset = [0, -60]

        mode = str(title_cfg.get("mode", "white_mask")).strip().lower()
        if mode not in ("white_mask", "grayscale"):
            mode = "white_mask"

        lower_white = int(title_cfg.get("white_lower", 150))
        upper_white = int(title_cfg.get("white_upper", 255))
        lower_white = max(0, min(255, lower_white))
        upper_white = max(lower_white, min(255, upper_white))

        blur_k_raw = title_cfg.get("blur_kernel", 3)
        try:
            blur_k = int(blur_k_raw)
        except (TypeError, ValueError):
            blur_k = 3
        blur_k = max(1, blur_k)
        if blur_k % 2 == 0:
            blur_k += 1

        match_threshold = float(title_cfg.get("match_threshold", 0.3))
        cache_ttl = int(title_cfg.get("cache_ttl", 30))
        center_search_radius = int(title_cfg.get("center_search_radius", 420))
        center_accept_radius = int(title_cfg.get("center_accept_radius", min(center_search_radius, 260)))
        center_bias_weight = float(title_cfg.get("center_bias_weight", 0.08))
        max_jump_px = int(title_cfg.get("max_jump_px", 220))
        prev_search_radius = int(title_cfg.get("prev_search_radius", 220))
        stick_to_prev_margin = float(title_cfg.get("stick_to_prev_margin", 0.02))
        strong_match_threshold = float(title_cfg.get("strong_match_threshold", max(0.03, match_threshold * 0.35)))
        super_strong_threshold = float(title_cfg.get("super_strong_threshold", 0.01))
        top_k = max(1, int(title_cfg.get("top_k", 12)))
        manual_select = bool(title_cfg.get("manual_select", True))
        max_candidates = max(1, int(title_cfg.get("max_candidates", 6)))
        candidate_threshold = float(
            title_cfg.get("candidate_threshold", min(0.35, max(match_threshold, match_threshold * 1.25)))
        )
        lock_miss_tolerance = max(1, int(title_cfg.get("lock_miss_tolerance", 15)))
        manual_track_radius = max(20, int(title_cfg.get("manual_track_radius", prev_search_radius)))
        manual_track_threshold = float(
            title_cfg.get("manual_track_threshold", min(0.4, max(match_threshold + 0.05, match_threshold * 1.25)))
        )

        # Periodically drop cached title lock so the tracker can recover from a wrong person.
        if (self.loc_title is not None and cache_ttl > 0
                and self._title_last_seen_frame >= 0
                and self.frame_count - self._title_last_seen_frame > cache_ttl):
            self.loc_title = None

        had_lock = self.loc_title is not None

        def _prepare_title_match_images():
            if mode == "grayscale":
                img_pat = self.img_title_gray
                img_roi = self.img_frame_gray
                if blur_k > 1:
                    img_pat = cv2.GaussianBlur(img_pat, (blur_k, blur_k), 0)
                    img_roi = cv2.GaussianBlur(img_roi, (blur_k, blur_k), 0)
                return img_roi, img_pat

            # white_mask mode (legacy default)
            img_pat = cv2.inRange(self.img_title_gray, lower_white, upper_white)
            img_roi = cv2.inRange(self.img_frame_gray, lower_white, upper_white)
            if blur_k > 1:
                img_pat = cv2.GaussianBlur(img_pat, (blur_k, blur_k), 0)
                img_roi = cv2.GaussianBlur(img_roi, (blur_k, blur_k), 0)
            return img_roi, img_pat

        def _biased_sqdiff_match(img_roi, img_pat, anchor, radius=None, bias_weight=0.08):
            """Pick among top-K SQDIFF matches using a distance-to-anchor bias."""
            h_img, w_img = img_roi.shape[:2]
            h_pat, w_pat = img_pat.shape[:2]
            if h_pat > h_img or w_pat > w_img:
                return None

            x_off = 0
            y_off = 0
            search_img = img_roi

            if radius is not None:
                x0 = max(0, int(anchor[0] - radius))
                y0 = max(0, int(anchor[1] - radius))
                x1 = min(w_img, int(anchor[0] + radius + w_pat))
                y1 = min(h_img, int(anchor[1] + radius + h_pat))
                if x1 - x0 >= w_pat and y1 - y0 >= h_pat:
                    search_img = img_roi[y0:y1, x0:x1]
                    x_off, y_off = x0, y0

            result = cv2.matchTemplate(search_img, img_pat, cv2.TM_SQDIFF_NORMED)
            result = np.nan_to_num(result, nan=1.0, posinf=1.0, neginf=1.0)

            flat = result.reshape(-1)
            if flat.size == 0:
                return None

            k = min(top_k, flat.size)
            idxs = np.argpartition(flat, k - 1)[:k]
            out_h, out_w = result.shape[:2]
            diag = max((w_img ** 2 + h_img ** 2) ** 0.5, 1.0)

            best = None
            for idx in idxs:
                y, x = divmod(int(idx), out_w)
                score = float(result[y, x])
                gx = x + x_off
                gy = y + y_off
                dist = ((gx - anchor[0]) ** 2 + (gy - anchor[1]) ** 2) ** 0.5 / diag
                cost = score + bias_weight * dist
                # 默认选相似度最高（SQDIFF score 最小）的候选，不做距离偏置。
                if best is None or score < best["score"]:
                    best = {
                        "loc": (gx, gy),
                        "score": score,
                        "cost": cost,
                        "center_dist_px": ((gx - center_anchor[0]) ** 2 + (gy - center_anchor[1]) ** 2) ** 0.5,
                    }
            return best

        try:
            img_roi_proc, img_title_proc = _prepare_title_match_images()
            th, tw = img_title_proc.shape[:2]
            fh, fw = img_roi_proc.shape[:2]

            center_anchor = (max(0, fw // 2 - tw // 2), max(0, fh // 2 - th // 2))
            candidate_pool = self._collect_title_candidates(
                img_roi_proc, img_title_proc,
                max_candidates=max_candidates,
                score_threshold=candidate_threshold,
            )
            self._update_title_candidates_snapshot(candidate_pool, (tw, th))
            # 始终记录称号最佳匹配相似度（右下角显示用）。
            self.title_sim = (
                max(0.0, min(1.0, 1.0 - candidate_pool[0]["score"]))
                if candidate_pool else None
            )

            with self._title_state_lock:
                if self._pending_title_select_loc is not None:
                    self._selected_title_loc = self._pending_title_select_loc
                    self._manual_title_locked = True
                    self.loc_title = self._pending_title_select_loc
                    self._title_lock_miss_count = 0
                    self._pending_title_select_loc = None
                    logger.info(f"[Title] Manual target armed at {self.loc_title}")
                manual_locked = manual_select and self._manual_title_locked
                selected_loc = self._selected_title_loc

            if manual_locked and selected_loc is not None:
                tracked = []
                for cand in candidate_pool:
                    dist = ((cand["loc"][0] - selected_loc[0]) ** 2 + (cand["loc"][1] - selected_loc[1]) ** 2) ** 0.5
                    cand["manual_dist_px"] = dist
                    cand["src"] = cand.get("src", "manual_pool")
                    if dist <= manual_track_radius:
                        tracked.append(cand)

                manual_best = min(
                    tracked,
                    key=lambda c: (c["score"], c.get("manual_dist_px", 0.0)),
                    default=None,
                )

                if manual_best is not None and manual_best["score"] <= manual_track_threshold:
                    loc = manual_best["loc"]
                    score = manual_best["score"]
                    match_src = "manual"
                    rejected = []
                    with self._title_state_lock:
                        self._selected_title_loc = loc
                        self._title_lock_miss_count = 0
                    self._update_title_candidates_snapshot(candidate_pool, (tw, th))
                else:
                    with self._title_state_lock:
                        self._title_lock_miss_count += 1
                        miss_count = self._title_lock_miss_count
                    if self.loc_title is not None and miss_count <= lock_miss_tolerance:
                        loc = self.loc_title
                        score = 1.0
                        match_src = "manual_hold"
                        rejected = []
                    else:
                        with self._title_state_lock:
                            self._manual_title_locked = False
                            self._selected_title_loc = None
                            self._title_lock_miss_count = 0
                        manual_locked = False

            candidates = []

            # Candidate 1: around previous lock (if any), weak bias for temporal stability.
            if not manual_locked and self.loc_title is not None:
                prev_cand = _biased_sqdiff_match(
                    img_roi_proc, img_title_proc,
                    anchor=self.loc_title,
                    radius=prev_search_radius,
                    bias_weight=0.04,
                )
                if prev_cand is not None:
                    prev_cand["src"] = "prev"
                    prev_cand["jump_px"] = 0.0
                    candidates.append(prev_cand)

            # Candidate 2: around center (primary anti-confusion strategy).
            if not manual_locked:
                center_cand = _biased_sqdiff_match(
                    img_roi_proc, img_title_proc,
                    anchor=center_anchor,
                    radius=center_search_radius,
                    bias_weight=center_bias_weight,
                )
                if center_cand is not None:
                    center_cand["src"] = "center"
                    center_cand["jump_px"] = (
                        ((center_cand["loc"][0] - self.loc_title[0]) ** 2
                         + (center_cand["loc"][1] - self.loc_title[1]) ** 2) ** 0.5
                        if self.loc_title is not None else 0.0
                    )
                    candidates.append(center_cand)

            # Candidate 3: global fallback only for recovery when there is no active lock.
            best_local_score = min([c["score"] for c in candidates], default=1.0)
            if not manual_locked and not had_lock and best_local_score > match_threshold * 0.9:
                global_cand = _biased_sqdiff_match(
                    img_roi_proc, img_title_proc,
                    anchor=center_anchor,
                    radius=None,
                    bias_weight=center_bias_weight,
                )
                if global_cand is not None:
                    global_cand["src"] = "global"
                    global_cand["jump_px"] = 0.0
                    candidates.append(global_cand)

            # 把全屏候选也纳入最终选择（candidate_pool 已按相似度从高到低），
            # 保证「相似度最高」的全屏匹配能被选到，即使它离屏幕中心较远。
            if not manual_locked:
                for cp in candidate_pool:
                    cx, cy = cp["loc"]
                    candidates.append({
                        "loc": (cx, cy),
                        "score": cp["score"],
                        "cost": cp["score"],
                        "center_dist_px": ((cx - center_anchor[0]) ** 2 + (cy - center_anchor[1]) ** 2) ** 0.5,
                        "jump_px": (((cx - self.loc_title[0]) ** 2 + (cy - self.loc_title[1]) ** 2) ** 0.5
                                     if self.loc_title is not None else 0.0),
                        "src": "pool",
                    })

            if not manual_locked and not candidates:
                self._fallback_player_location()
                return

            if not manual_locked:
                filtered = []
                rejected = []
                for cand in candidates:
                    src = cand.get("src", "unknown")
                    center_dist_px = cand.get("center_dist_px", 0.0)
                    jump_px = cand.get("jump_px", 0.0)

                    if src != "prev" and center_dist_px > center_accept_radius:
                        rejected.append((src, "far_from_center", round(center_dist_px, 1), round(cand["score"], 4)))
                        continue
                    if had_lock and src != "prev" and jump_px > max_jump_px:
                        rejected.append((src, "jump_too_large", round(jump_px, 1), round(cand["score"], 4)))
                        continue
                    filtered.append(cand)

                if not filtered and had_lock:
                    loc = self.loc_title
                    score = 1.0
                    match_src = "prev_keep"
                else:
                    if not filtered:
                        self._fallback_player_location()
                        return

                    # 默认定位：识别率最高（SQDIFF score 最小，即相似度最高）的候选。
                    best = min(filtered, key=lambda c: c["score"])

                    loc = best["loc"]
                    score = best["score"]
                    match_src = best.get("src", "unknown")
        except cv2.error as e:
            logger.warning(
                f"[Title] OpenCV error, skip frame: {e}. "
                f"mode={mode}, blur_kernel={blur_k}, white=({lower_white},{upper_white})"
            )
            self._fallback_player_location()
            return
        except Exception as e:
            logger.warning(f"[Title] Detection error, skip frame: {e}")
            self._fallback_player_location()
            return

        # Log score every frame for first 100 frames, then every 30 frames
        if self.frame_count <= 100 or self.frame_count % 30 == 0:
            status = "HIT" if (score < match_threshold or match_src in ("manual", "manual_hold", "prev_keep")) else "MISS"
            logger.info(
                f"[Title] frame={self.frame_count} score={score:.4f} "
                f"thresh={match_threshold:.2f} {status} src={match_src} mode={mode}"
            )
            if 'rejected' in locals() and rejected:
                logger.info(f"[Title] Rejected candidates: {rejected[:3]}")

        if score < match_threshold or match_src in ("prev_keep", "manual", "manual_hold"):
            # Anti-jump guard: reject sudden large jumps unless score is very confident.
            if self.loc_title is not None and match_src not in ("prev_keep", "manual_hold"):
                jump = ((loc[0] - self.loc_title[0]) ** 2 + (loc[1] - self.loc_title[1]) ** 2) ** 0.5
                if jump > max_jump_px and score > max(0.04, match_threshold * 0.45):
                    loc = self.loc_title
            self.loc_title = loc
            self._title_last_seen_frame = self.frame_count
            self.title_sim = max(0.0, min(1.0, 1.0 - score))
            if 'candidate_pool' in locals() and 'tw' in locals() and 'th' in locals():
                self._update_title_candidates_snapshot(candidate_pool, (tw, th))
            tw, th = self.img_title.shape[1], self.img_title.shape[0]
            self.loc_player = (
                loc[0] + tw // 2 + title_offset[0],
                loc[1] + th // 2 + title_offset[1]
            )
            if self.img_frame_debug is not None:
                draw_rectangle(
                    self.img_frame_debug,
                    loc, (th, tw), (255, 0, 0),
                    f"TITLE {self.title_sim:.2f}", thickness=3, text_height=1.0
                )
        elif self.loc_title is None:
            self._fallback_player_location()

    def _detect_minimap(self):
        """
        Detect minimap via white border (reference project pattern).
        Falls back to config if detection fails.

        When minimap.fixed is true, skip auto-detection and always use the
        manually-calibrated config coordinates. This is the more reliable
        choice when the game window size/position never changes (the minimap
        sits at a fixed pixel location).
        """
        # Manual drag-selection (real-time calibration) has the highest priority.
        if self._manual_minimap_rect is not None:
            return self._manual_minimap_rect

        mm_cfg = self.cfg.get("minimap", {})

        # Fixed mode: trust the calibrated coordinates (no auto-detect drift).
        if mm_cfg.get("fixed", False):
            if mm_cfg.get("width", 0) > 10 and mm_cfg.get("height", 0) > 10:
                return (mm_cfg["x"], mm_cfg["y"], mm_cfg["width"], mm_cfg["height"])
            return None

        # Try auto-detection first (white border method)
        result = get_minimap_loc_size(self.img_frame)
        if result:
            x, y, w_m, h_m = result
            if self.frame_count == 1:
                logger.info(f"[Minimap] Auto-detected: ({x},{y}) {w_m}x{h_m}")
            # Keep a 1px inset from border, but reject degenerate auto-detects.
            x_i, y_i, w_i, h_i = (x + 1, y + 1, w_m - 2, h_m - 2)
            if w_i > 0 and h_i > 0:
                return (x_i, y_i, w_i, h_i)

        # Fallback to config
        if mm_cfg.get("width", 0) > 10 and mm_cfg.get("height", 0) > 10:
            return (mm_cfg["x"], mm_cfg["y"], mm_cfg["width"], mm_cfg["height"])
        return None

    def toggle_minimap_select(self):
        """Toggle the drag-to-select minimap mode (for real-time calibration)."""
        self._minimap_select_mode = not self._minimap_select_mode
        self._mm_drag_start = None
        self._mm_drag_end = None
        self._mm_hover = None
        logger.info(f"[Minimap] Select mode {'ON' if self._minimap_select_mode else 'OFF'}")
        return self._minimap_select_mode

    def minimap_select_mouse(self, event, display_x, display_y, flags, param):
        """
        Handle drag-to-select for the minimap. Coordinates are display-space
        (scaled by display_scale); we convert back to full-frame on release.
        Also tracks the hover position for the magnifier.
        """
        if not self._minimap_select_mode:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            self._mm_hover = (display_x, display_y)
            if self._mm_drag_start is not None:
                self._mm_drag_end = (display_x, display_y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mm_drag_start = (display_x, display_y)
            self._mm_drag_end = (display_x, display_y)
        elif event == cv2.EVENT_LBUTTONUP and self._mm_drag_start is not None:
            scale = float(self.cfg.get("monster_detect", {}).get("display_scale", 0.55))
            scale = max(scale, 0.01)
            x0 = int(round(min(self._mm_drag_start[0], display_x) / scale))
            y0 = int(round(min(self._mm_drag_start[1], display_y) / scale))
            x1 = int(round(max(self._mm_drag_start[0], display_x) / scale))
            y1 = int(round(max(self._mm_drag_start[1], display_y) / scale))
            w, h = x1 - x0, y1 - y0
            if w > 5 and h > 5:
                self._manual_minimap_rect = (x0, y0, w, h)
                logger.info(
                    f"[Minimap] Selected rect: x={x0} y={y0} width={w} height={h}"
                )
                logger.info(
                    f"[Minimap] → 写进 config:  x: {x0}, y: {y0}, width: {w}, height: {h}"
                )
            self._mm_drag_start = None
            self._mm_drag_end = None

    def _draw_minimap_magnifier(self):
        """
        Draw a zoomed view of the area around the mouse cursor in the top-right
        corner, so the minimap border can be selected pixel-precisely.
        """
        if self.img_frame_debug is None:
            return
        if not self._minimap_select_mode or self._mm_hover is None:
            return

        scale = float(self.cfg.get("monster_detect", {}).get("display_scale", 0.55))
        scale = max(scale, 0.01)
        cx = int(round(self._mm_hover[0] / scale))
        cy = int(round(self._mm_hover[1] / scale))

        fh, fw = self.img_frame.shape[:2]
        crop_half = 50       # full-frame px each side → 100x100 window
        zoom = 3
        x0 = max(0, cx - crop_half)
        y0 = max(0, cy - crop_half)
        x1 = min(fw, cx + crop_half)
        y1 = min(fh, cy + crop_half)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return

        crop = self.img_frame[y0:y1, x0:x1]
        crop = cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom),
                          interpolation=cv2.INTER_NEAREST)
        ch, cw = crop.shape[:2]

        px_paste = fw - cw - 10
        py_paste = 10
        if not (py_paste >= 0 and px_paste >= 0
                and py_paste + ch <= fh and px_paste + cw <= fw):
            return
        self.img_frame_debug[py_paste:py_paste + ch, px_paste:px_paste + cw] = crop
        cv2.rectangle(self.img_frame_debug, (px_paste, py_paste),
                      (px_paste + cw, py_paste + ch), (255, 255, 255), 2)

        # Short thick crosshair centered on the cursor pixel (accounting for edge clamping).
        cxx = px_paste + int((cx - x0) * zoom)
        cyy = py_paste + int((cy - y0) * zoom)
        arm = 12
        cross_color = (0, 0, 255)  # 红色
        cv2.line(self.img_frame_debug, (cxx - arm, cyy), (cxx + arm, cyy), cross_color, 3)
        cv2.line(self.img_frame_debug, (cxx, cyy - arm), (cxx, cyy + arm), cross_color, 3)
        cv2.circle(self.img_frame_debug, (cxx, cyy), 2, cross_color, -1)

    def _draw_minimap_selection(self):
        """Draw the manual minimap selection (and in-progress drag) on the debug frame."""
        if self.img_frame_debug is None:
            return
        scale = float(self.cfg.get("monster_detect", {}).get("display_scale", 0.55))
        scale = max(scale, 0.01)

        # In-progress drag rectangle (display-space → full-frame).
        if (self._minimap_select_mode
                and self._mm_drag_start is not None
                and self._mm_drag_end is not None):
            x0 = int(round(min(self._mm_drag_start[0], self._mm_drag_end[0]) / scale))
            y0 = int(round(min(self._mm_drag_start[1], self._mm_drag_end[1]) / scale))
            x1 = int(round(max(self._mm_drag_start[0], self._mm_drag_end[0]) / scale))
            y1 = int(round(max(self._mm_drag_start[1], self._mm_drag_end[1]) / scale))
            cv2.rectangle(self.img_frame_debug, (x0, y0), (x1, y1), (255, 0, 255), 2)

        # Confirmed manual rect.
        if self._manual_minimap_rect is not None:
            x, y, w, h = self._manual_minimap_rect
            draw_rectangle(
                self.img_frame_debug, (x, y), (h, w),
                (255, 0, 255), "MM manual", thickness=2, text_height=0.5,
            )

    # -------------------------------------------------------------------
    # Route following
    # -------------------------------------------------------------------

    def load_map_and_routes(self, map_name):
        """Load map.png and route*.png for the given map."""
        import glob
        map_path = f"minimaps/{map_name}/map.png"
        if not os.path.exists(map_path):
            logger.warning(f"[Route] Map not found: {map_path}")
            return False

        self.img_map = load_image(map_path, cv2.IMREAD_COLOR)
        logger.info(f"[Route] Loaded map: {map_path} ({self.img_map.shape[1]}x{self.img_map.shape[0]})")

        # Load route images
        route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
        route_files = [p for p in route_files if "route_rest" not in p]
        self.img_routes = []
        for rf in route_files:
            img = cv2.cvtColor(load_image(rf), cv2.COLOR_BGR2RGB)
            # Strip map-resident colors matching route codes from route image
            img = mask_route_colors(self.img_map.copy(), img, self.cfg["route"]["color_code"])
            img = mask_route_colors(self.img_map.copy(), img, self.cfg["route"]["color_code_up_down"])
            self.img_routes.append(img)
            logger.info(f"[Route] Loaded: {rf}")

        if self.img_routes:
            self.img_route = self.img_routes[0]

        # Parse color→command dictionaries
        for k, v in self.cfg["route"]["color_code"].items():
            self.color_code[tuple(map(int, k.split(',')))] = v
        for k, v in self.cfg["route"]["color_code_up_down"].items():
            self.color_code_up_down[tuple(map(int, k.split(',')))] = v

        logger.info(f"[Route] Ready: {len(self.img_routes)} routes")
        return True

    def get_player_global_position(self):
        """Track global position via template matching (handles X and Y, incl. climbing)."""
        if self.img_map is None or self.img_minimap is None:
            return
        if self.loc_player_minimap is None:
            return

        # Create mask: ignore black background pixels of the minimap.
        mask = np.any(self.img_minimap != [0, 0, 0], axis=2).astype(np.uint8) * 255

        # Template match current minimap against global map (local search only — fast).
        last_result = None
        if self.loc_minimap_global != (0, 0):
            last_result = (int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1]))
        loc, score, _ = find_pattern_sqdiff(
            self.img_map, self.img_minimap,
            last_result=last_result,
            mask=mask,
            global_threshold=1.0,  # Always accept local search (never full-image scan)
        )
        self._last_match_score = score
        self.loc_minimap_global = loc
        self.loc_player_global = (
            loc[0] + self.loc_player_minimap[0],
            loc[1] + self.loc_player_minimap[1],
        )

    def get_nearest_route_color(self):
        """Scan route image around player for nearest color-coded pixel."""
        if self.img_route is None or self.loc_player_global == (0.0, 0.0):
            return None, None

        return self._get_nearest_route_color_from_image(self.img_route)

    def _get_nearest_route_color_from_image(self, route_img, search_range=None):
        """Scan a specific route image around player for nearest route pixels."""
        if route_img is None or self.loc_player_global == (0.0, 0.0):
            return None, None

        x0, y0 = int(self.loc_player_global[0]), int(self.loc_player_global[1])
        h, w = route_img.shape[:2]
        if search_range is None:
            search_range = self.cfg["route"].get("search_range", 10)

        x_min = max(0, x0 - search_range)
        x_max = min(w, x0 + search_range)
        y_min = max(0, y0 - search_range)
        y_max = min(h, y0 + search_range)

        nearest = None
        nearest_ud = None
        min_dist = float('inf')
        min_dist_ud = float('inf')

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(route_img[y, x])
                dist = abs(x - x0) + abs(y - y0)
                if pixel in self.color_code and dist < min_dist:
                    nearest = {"pixel": (x, y), "command": self.color_code[pixel], "distance": dist}
                    min_dist = dist
                if pixel in self.color_code_up_down and dist < min_dist_ud:
                    nearest_ud = {"pixel": (x, y), "command": self.color_code_up_down[pixel], "distance": dist}
                    min_dist_ud = dist

        return nearest, nearest_ud

    def _select_best_route(self):
        """Select the route image nearest to the current global player position."""
        if not self.img_routes or self.loc_player_global == (0.0, 0.0):
            return

        search_range = int(self.cfg.get("route", {}).get("search_range", 10))
        auto_select_range = int(self.cfg.get("route", {}).get("auto_select_range", max(30, search_range * 4)))

        best = None
        for idx, route_img in enumerate(self.img_routes):
            nearest, nearest_ud = self._get_nearest_route_color_from_image(route_img, auto_select_range)
            distances = []
            if nearest is not None:
                distances.append(nearest["distance"])
            if nearest_ud is not None:
                distances.append(nearest_ud["distance"])
            if not distances:
                continue
            dist = min(distances)
            if best is None or dist < best["dist"]:
                best = {"idx": idx, "dist": dist}

        if best is not None and best["idx"] != self.idx_routes:
            self.idx_routes = best["idx"]
            self.img_route = self.img_routes[self.idx_routes]
            logger.info(
                f"[Route] Auto-selected route {self.idx_routes + 1}/{len(self.img_routes)} "
                f"for global=({int(self.loc_player_global[0])},{int(self.loc_player_global[1])}) "
                f"dist={best['dist']}"
            )

    def update_cmd_by_route(self):
        """Convert nearest route color to movement commands."""
        self._select_best_route()
        nearest, nearest_ud = self.get_nearest_route_color()

        if nearest or nearest_ud:
            # Found a route color → reset the miss counter.
            self._route_miss_frames = 0
            comp_dist = int(self.cfg.get("route", {}).get("climb_complement_dist", 3))
            if nearest and nearest_ud:
                if nearest["distance"] < nearest_ud["distance"]:
                    self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest["command"].split()
                    _, cmd, _ = nearest_ud["command"].split()
                    # 已上梯子 → 补垂直方向，保证爬梯流畅；
                    # 横向跳跃且附近是「向上爬」→ 跳的同时按住 up，抓住绳子。
                    # 仅当另一颜色紧邻(距离<=comp_dist)才互补，避免把远处无关方向注入。
                    if self.cmd_move_y == "none" and nearest_ud["distance"] <= comp_dist and (
                        self.is_on_ladder or (self.cmd_action == "jump" and cmd == "up")
                    ):
                        self.cmd_move_y = cmd
                else:
                    self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest_ud["command"].split()
                    cmd, _, _ = nearest["command"].split()
                    # 竖直爬梯时不要被远处横向颜色带偏方向；只有紧邻的横向才互补。
                    if self.cmd_move_x == "none" and self.is_on_ladder and nearest["distance"] <= comp_dist:
                        self.cmd_move_x = cmd
            elif nearest:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest["command"].split()
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest_ud["command"].split()
        else:
            # No route color nearby: keep the last movement for a short grace
            # period to bridge small gaps in the recorded route, then stop.
            self._route_miss_frames += 1
            grace = int(self.cfg.get("route", {}).get("miss_grace_frames", 15))
            if self._route_miss_frames > grace:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = "none", "none", "none"

        # Goal → cycle to next route
        if self.cmd_action == "goal" and self.img_routes:
            self.idx_routes = (self.idx_routes + 1) % len(self.img_routes)
            self.img_route = self.img_routes[self.idx_routes]
            if self.frame_count % 30 == 0:
                logger.info(f"[Route] Goal reached → route {self.idx_routes + 1}/{len(self.img_routes)}")

    def _draw_player_marker(self):
        """Draw the player position marker on the debug frame."""
        if self.img_frame_debug is None:
            return
        cv2.circle(
            self.img_frame_debug,
            self.loc_player, radius=5,
            color=(0, 0, 255), thickness=-1
        )
        cv2.putText(
            self.img_frame_debug, "Player",
            (self.loc_player[0] - 25, self.loc_player[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    def _draw_appearance_debug(self):
        """Draw the appearance match box (magenta = primary anchor active)."""
        if self.img_frame_debug is None or self.appearance_result is None:
            return
        ap = self.appearance_result
        x, y = ap["loc"]
        h, w = ap["size"]
        draw_rectangle(
            self.img_frame_debug,
            (x, y), (h, w),
            (255, 0, 255),
            f"AP {ap['score']:.2f}",
            thickness=3, text_height=1.0,
        )

    def _get_attack_facing(self) -> str:
        """Return left/right facing toward the nearest detected monster."""
        if not self.monsters:
            return "none"

        px, _ = self.loc_player
        nearest_dx = None
        for m in self.monsters:
            mx, _ = m.get("position", (0, 0))
            mh, mw = m.get("size", (0, 0))
            cx = mx + (mw // 2)
            dx = cx - px
            if nearest_dx is None or abs(dx) < abs(nearest_dx):
                nearest_dx = dx

        if nearest_dx is None:
            return "none"

        deadzone = int(self.cfg.get("directional_attack", {}).get("face_deadzone_px", 8))
        facing = "none" if abs(nearest_dx) <= deadzone else ("right" if nearest_dx > 0 else "left")
        if self.frame_count % 30 == 0:
            logger.info(
                f"[Face] player_x={px} monster_x={px + nearest_dx} dx={nearest_dx} → {facing}"
            )
        return facing

    def _reset_hunt_search_state(self):
        """Reset no-monster search state once a target is found."""
        self._hunt_no_monster_frames = 0
        self._hunt_search_flip_frame = 0
        self._hunt_search_sweeps = 0
        self._hunt_search_pause_until = 0
        self._hunt_search_anchor_x = None
        self._hunt_search_last_progress_x = None
        self._hunt_search_stall_frames = 0
        self._hunt_search_last_reason = "target"

    def _get_hunt_search_progress_x(self):
        """Return stable X progress for search logic, preferring map/minimap coordinates.
        Screen-space coords are NOT used because the camera follows the player."""
        if self.cfg.get("bot", {}).get("enable_route") and self.loc_player_global != (0.0, 0.0):
            return float(self.loc_player_global[0]), "global"
        if self.loc_player_minimap is not None:
            return float(self.loc_player_minimap[0]), "minimap"
        return None, "none"

    def _get_hunt_search_command(self):
        """Return (move_x, move_y, action) for left/right sweeping while searching monsters."""
        cfg = self.cfg.get("directional_attack", {})
        if not bool(cfg.get("search_enabled", True)):
            return "none", "none", "none"

        # Keep ladder handling to route logic; avoid lateral sweep while climbing.
        if self.is_on_ladder:
            return "none", "none", "none"

        trigger_frames = max(1, int(cfg.get("search_trigger_frames", 4)))
        if self._hunt_no_monster_frames < trigger_frames:
            return "none", "none", "none"

        if self.frame_count < self._hunt_search_pause_until:
            return "none", "none", "none"

        flip_frames = max(4, int(cfg.get("search_flip_frames", 18)))
        progress_x, progress_src = self._get_hunt_search_progress_x()
        if progress_x is not None and self._hunt_search_anchor_x is None:
            self._hunt_search_anchor_x = progress_x
        if progress_x is not None and self._hunt_search_last_progress_x is None:
            self._hunt_search_last_progress_x = progress_x

        flip_reason = None
        stall_frames_limit = int(cfg.get("search_stall_frames", 8))
        if stall_frames_limit > 0:
            stall_min_delta_px = float(cfg.get("search_stall_min_delta_px", 1.0))
            if progress_x is not None and self._hunt_search_last_progress_x is not None:
                progress_delta = abs(progress_x - self._hunt_search_last_progress_x)
                if progress_delta <= stall_min_delta_px:
                    self._hunt_search_stall_frames += 1
                else:
                    self._hunt_search_stall_frames = 0
                self._hunt_search_last_progress_x = progress_x
                if progress_src in ("global", "minimap") and self._hunt_search_stall_frames >= stall_frames_limit:
                    flip_reason = "stall"

        max_offset_px = float(cfg.get("search_max_offset_px", 0))
        if (
            flip_reason is None
            and max_offset_px > 0
            and progress_src in ("global", "minimap")
            and progress_x is not None
            and self._hunt_search_anchor_x is not None
        ):
            offset_x = progress_x - self._hunt_search_anchor_x
            if self._hunt_search_dir == "left" and offset_x <= -max_offset_px:
                flip_reason = "bound"
            elif self._hunt_search_dir == "right" and offset_x >= max_offset_px:
                flip_reason = "bound"

        if self._hunt_search_flip_frame <= 0:
            self._hunt_search_flip_frame = self.frame_count + flip_frames
        elif flip_reason is None and self.frame_count >= self._hunt_search_flip_frame:
            flip_reason = "timer"

        if flip_reason is not None:
            self._hunt_search_dir = "right" if self._hunt_search_dir == "left" else "left"
            self._hunt_search_flip_frame = self.frame_count + flip_frames
            self._hunt_search_sweeps += 1
            self._hunt_search_stall_frames = 0
            self._hunt_search_last_reason = flip_reason
            if progress_x is not None:
                self._hunt_search_anchor_x = progress_x
                self._hunt_search_last_progress_x = progress_x

        max_sweeps = int(cfg.get("search_max_sweeps", 0))
        if max_sweeps > 0 and self._hunt_search_sweeps >= max_sweeps:
            pause_frames = max(0, int(cfg.get("search_pause_frames", 12)))
            self._hunt_search_pause_until = self.frame_count + pause_frames
            self._hunt_search_sweeps = 0
            self._hunt_search_last_reason = "pause"
            return "none", "none", "none"

        action = "attack" if bool(cfg.get("search_attack_while_move", True)) else "none"

        # Random jump during search (looks more human-like)
        jump_chance = float(cfg.get("search_random_jump_chance", 0.0))
        if jump_chance > 0:
            jump_cd = max(1, int(cfg.get("search_random_jump_cd", 30)))
            if not hasattr(self, '_hunt_next_jump_frame'):
                self._hunt_next_jump_frame = 0
            if self.frame_count >= self._hunt_next_jump_frame:
                import random
                if random.random() < jump_chance:
                    action = "jump"
                    self._hunt_next_jump_frame = self.frame_count + jump_cd

        # 80% 宽度边界：角色只能在中部这个比例内移动，到边界就掉头。
        boundary_pct = float(self.cfg.get("minimap", {}).get("width_boundary_pct", 0.0))
        if 0.0 < boundary_pct < 1.0 and self.loc_player_minimap is not None and self.img_minimap is not None:
            mm_w = self.img_minimap.shape[1]
            if mm_w > 0:
                margin = mm_w * (1.0 - boundary_pct) / 2.0
                left_b = margin
                right_b = mm_w - margin
                mx = self.loc_player_minimap[0]
                if mx <= left_b and self._hunt_search_dir == "left":
                    self._hunt_search_dir = "right"
                elif mx >= right_b and self._hunt_search_dir == "right":
                    self._hunt_search_dir = "left"

        return self._hunt_search_dir, "none", action

    def _get_in_place_command(self):
        """
        In-place attack with periodic small back-and-forth, to avoid MapleStory's
        attack lockout when standing still too long.

        Walks backward (opposite of facing) first, then forward (facing dir)
        to return to roughly the original spot.

        State machine:
          attack -> (after interval) -> walk_away(backward) ->
          walk_back(forward) -> (after walk_frames) -> attack
        Returns (move_x, move_y, action).
        """
        cfg = self.cfg.get("directional_attack", {})
        fps = max(1.0, float(self.cfg.get("system", {}).get("fps_limit_main", 15)))
        interval = max(1, int(float(cfg.get("in_place_walk_interval_seconds", 2.0)) * fps))
        walk_frames = max(1, int(float(cfg.get("in_place_walk_seconds", 0.5)) * fps))
        walk_back_frames = max(1, int(float(cfg.get("in_place_walk_back_seconds", 0.5)) * fps))
        stop_frames = max(1, int(float(cfg.get("in_place_stop_attack_seconds", 0.5)) * fps))

        # Advance state machine
        if self._in_place_state == "attack":
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= interval:
                # Fixed walk-away direction from config (default right).
                dir_behavior = cfg.get("in_place_walk_dir", "right")
                if dir_behavior == "left":
                    self._in_place_backward_dir = "left"
                else:
                    self._in_place_backward_dir = "right"
                self._in_place_walk_dir = self._in_place_backward_dir
                self._in_place_state = "pre_walk"
                self._in_place_state_frame = 0
        elif self._in_place_state == "pre_walk":
            # Stop attacking before moving (avoids attacking while walking).
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= stop_frames:
                self._in_place_state = "walk_away"
                self._in_place_state_frame = 0
        elif self._in_place_state == "walk_away":
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= walk_frames:
                self._in_place_state = "mid_pause"
                self._in_place_state_frame = 0
        elif self._in_place_state == "mid_pause":
            # Pause between walking right and walking left.
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= stop_frames:
                self._in_place_state = "walk_back"
                self._in_place_state_frame = 0
        elif self._in_place_state == "walk_back":
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= walk_back_frames:
                self._in_place_state = "post_walk"
                self._in_place_state_frame = 0
        elif self._in_place_state == "post_walk":
            # Pause after moving, before resuming attack.
            self._in_place_state_frame += 1
            if self._in_place_state_frame >= stop_frames:
                self._in_place_state = "attack"
                self._in_place_state_frame = 0

        # Produce command per state
        if self._in_place_state == "walk_away":
            return self._in_place_backward_dir, "none", "none"   # 往后走
        elif self._in_place_state == "mid_pause":
            # On the first frame of the mid pause, fire a single attack once
            # (attempt to break attack lockout after moving right).
            if self._in_place_state_frame == 1:
                return "none", "none", "attack_once"
            return "none", "none", "none"
        elif self._in_place_state == "walk_back":
            forward_dir = "right" if self._in_place_backward_dir == "left" else "left"
            return forward_dir, "none", "none"                    # 往前走

        # "attack" / "pre_walk" / "post_walk": stand still, no movement.
        # Attack only during "attack" state (not during pre/post_walk pause).
        if self._in_place_state == "attack":
            if self.monsters:
                self._in_place_no_monster_frames = 0
                aoe_threshold = int(
                    self.cfg.get("directional_attack", {}).get("aoe_monster_count", 3)
                )
                action = "aoe" if len(self.monsters) >= aoe_threshold else "attack"
                # Turn to face the nearest monster before attacking.
                facing = self._get_attack_facing()
                if facing in ("left", "right"):
                    return f"face_{facing}", "none", action
                return "none", "none", action
            # No monster detected: after a configurable idle time, fire AoE once
            # as a scout shot (does not move, so it never conflicts with walk-about).
            self._in_place_no_monster_frames += 1
            aoe_idle_seconds = float(
                self.cfg.get("directional_attack", {}).get("in_place_aoe_idle_seconds", 5.0)
            )
            aoe_idle_frames = max(1, int(aoe_idle_seconds * fps))
            if self._in_place_no_monster_frames >= aoe_idle_frames:
                self._in_place_no_monster_frames = 0
                return "none", "none", "aoe"  # one-shot scout shot
            return "none", "none", "none"
        return "none", "none", "none"

    def _update_debug_overlay(self):
        """Add FPS, HP/MP/EXP, and status info to the debug frame."""

        # --- HP / MP / EXP display (top-right) ---
        if self.health_monitor is not None:
            hm = self.health_monitor
            bar_count = self.cfg.get("health_monitor", {}).get("bar_count", 3)
            if bar_count == 2:
                bar_labels = [("HP", hm.hp_percent, (0, 0, 255), hm.enable_hp),
                              ("MP", hm.mp_percent, (255, 0, 0), hm.enable_mp)]
            else:
                bar_labels = [("HP", hm.hp_percent, (0, 0, 255), hm.enable_hp),
                              ("MP", hm.mp_percent, (255, 0, 0), hm.enable_mp),
                              ("EXP", hm.exp_percent, (0, 255, 0), True)]
            fw = self.img_frame.shape[1]
            x_s = fw - 220  # Right-aligned
            y_s = 30
            for i, (label, pct, color, enabled) in enumerate(bar_labels):
                status = "ON" if enabled else "OFF"
                text = f"{label}: {pct:.1f}% [{status}]"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(
                    self.img_frame_debug,
                    text,
                    (x_s, y_s + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )
                # Bar preview to the left of text
                if i < len(hm.loc_size_bars):
                    bx, by, bw, bh = hm.loc_size_bars[i]
                    if bw > 0 and bh > 0:
                        # loc_size_bars 现在存的是完整画面坐标，直接用完整帧预览
                        bar_src = self.img_frame
                        h_bar, w_bar = bar_src.shape[:2]
                        if by + bh <= h_bar and bx + bw <= w_bar:
                            preview = bar_src[by:by + bh, bx:bx + bw].copy()
                            # 预览条缩小到固定宽度，避免太占屏幕
                            preview_w = 100
                            preview_h = max(1, int(bh * preview_w / bw))
                            preview = cv2.resize(preview, (preview_w, preview_h),
                                                 interpolation=cv2.INTER_NEAREST)
                            px = x_s - preview_w - 10  # Preview to left of text
                            py = y_s + 30 * i - preview_h
                            ph, pw = preview.shape[:2]
                            if (py >= 0 and px >= 0
                                    and py + ph <= self.img_frame_debug.shape[0]
                                    and px + pw <= self.img_frame_debug.shape[1]):
                                self.img_frame_debug[py:py + ph, px:px + pw] = preview

        # --- Bot mode + title-fix indicator (top-right, below HP/MP) ---
        bot_mode = self.cfg.get("bot", {}).get("mode", "detect")
        mode_color = (0, 255, 0) if bot_mode == "hunt" else (0, 200, 255)
        mode_text = f"Bot: {bot_mode.upper()} [F11]"
        fix_text = f"Box: {'FIXED' if self._title_fixed else 'FOLLOW'} [F12]"
        fix_color = (0, 0, 255) if self._title_fixed else (0, 255, 0)

        for i, (text, color) in enumerate([(mode_text, mode_color), (fix_text, fix_color)]):
            (mw, mh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            x = self.img_frame.shape[1] - mw - 12
            y = 30 + 30 * 3 + 20 * i  # below HP/MP/EXP rows
            cv2.rectangle(self.img_frame_debug,
                          (x - 4, y - mh - 4),
                          (x + mw + 4, y + 4),
                          (0, 0, 0), -1)
            cv2.putText(self.img_frame_debug, text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # --- Status texts (bottom-left) ---
        diff_thres = self.cfg["monster_detect"]["diff_thres"]
        route_on = self.cfg["bot"].get("enable_route", False)
        texts = [
            f"FPS(P/D): {self.fps}/{self.display_fps} | KB: {'ON' if self._kb and self._kb.is_enable else 'OFF'}",
            f"Monsters: {len(self.monsters)} | Best: {self.best_score:.4f}/{diff_thres}",
            f"Mode: {self.cfg['monster_detect']['mode']} | Scale: {self.cfg.get('monster_detect',{}).get('display_scale',1)}",
            f"Frame: {self.img_frame.shape[1]}x{self.img_frame.shape[0]}",
        ]

        if self.cfg.get("bot", {}).get("mode", "detect") == "hunt":
            texts.append(
                f"Search: noMob={self._hunt_no_monster_frames} dir={self._hunt_search_dir} sweeps={self._hunt_search_sweeps} stall={self._hunt_search_stall_frames} last={self._hunt_search_last_reason}"
            )

        if route_on:
            gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
            mx, my = int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1])
            texts += [
                f"Route: {self.idx_routes+1}/{len(self.img_routes)} Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                f"Global: ({gx},{gy}) MM:({mx},{my}) PMM:{self.loc_player_minimap or '?'} Match:{getattr(self,'_last_match_score',1):.3f}",
                f"Ladder: {self.is_on_ladder} | Map: {self.img_map.shape[1]}x{self.img_map.shape[0]}" if self.img_map is not None else "No map loaded",
            ]
        if self.appearance_result is not None:
            ap = self.appearance_result
            texts.append(
                f"Appear(主): {ap['score']:.2f} feet=({ap['feet'][0]},{ap['feet'][1]})"
            )
        else:
            texts.append("Appear: 丢失(用称号)")
        texts += ["S=shot C/RClick=clear-title Q=quit F1=toggle"]

        # Bottom-left positioning (matches reference project)
        text_y_interval = 25
        text_y_start = self.img_frame.shape[0] - len(texts) * text_y_interval - 10
        font_scale = 0.8  # +2 sizes from 0.6

        for i, text in enumerate(texts):
            x, y = 10, text_y_start + text_y_interval * i
            # 黑底矩形 + 文字（比 4 次描边更快，也更清晰）
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(self.img_frame_debug,
                          (x - 2, y - th - 2),
                          (x + tw + 2, y + 4),
                          (0, 0, 0), -1)
            cv2.putText(
                self.img_frame_debug, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA
            )

        # --- Appearance / Title match scores (bottom-right) ---
        ap_score = getattr(self, "appearance_score", None)
        t_sim = getattr(self, "title_sim", None)
        ap_used = bool(getattr(self, "appearance_override", False))
        ap_txt = f"AP: {ap_score:.2f}" if ap_score is not None else "AP: N/A"
        title_txt = f"Title: {t_sim:.2f}" if t_sim is not None else "Title: N/A"
        score_rows = [
            (f"{ap_txt} {'[v]' if ap_used else '[x]'}", (255, 0, 255)),
            (f"{title_txt} {'[x]' if ap_used else '[v]'}", (0, 0, 255)),
        ]
        fw2 = self.img_frame.shape[1]
        fh2 = self.img_frame.shape[0]
        for i, (text, color) in enumerate(score_rows):
            (tw2, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x2 = fw2 - tw2 - 12
            y2 = fh2 - 12 - 26 * (len(score_rows) - 1 - i)
            cv2.rectangle(self.img_frame_debug,
                          (x2 - 4, y2 - th2 - 4),
                          (x2 + tw2 + 4, y2 + 4),
                          (0, 0, 0), -1)
            cv2.putText(self.img_frame_debug, text,
                        (x2, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Monster count in top-center (small text)
        if self.monsters:
            count_text = f"Monsters: {len(self.monsters)}"
            (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(
                self.img_frame_debug, count_text,
                (self.img_frame.shape[1] // 2 - tw // 2, th + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (42, 42, 165), 2, cv2.LINE_AA
            )

        self._draw_title_candidates()
        self._draw_appearance_debug()

    # -------------------------------------------------------------------
    # Main processing loop
    # -------------------------------------------------------------------

    def run_once(self) -> int:
        """
        Process a single game frame.

        Returns:
            0 on success, -1 if frame capture failed.
        """
        self.frame_count += 1
        self.is_frame_done = False

        # Capture frame
        self.img_frame = self.get_img_frame()
        if self.img_frame is None:
            logger.warning("Failed to capture game frame. Waiting...")
            if is_windows():
                activate_game_window(self.capture.window_title)
            return -1

        # Convert to grayscale (for title detection + grayscale matching)
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Update health monitor with the full frame; HealthMonitor crops the
        # bottom UI strip itself (fixed-coordinate mode needs full-frame coords).
        if self.health_monitor is not None:
            self.health_monitor.update_frame(self.img_frame)

        # Detect minimap (adapted for MapleStoryGo: bright fill + dark borders)
        minimap_result = self._detect_minimap()
        self.img_minimap = None
        if minimap_result:
            x, y, w, h = minimap_result
            fh, fw = self.img_frame.shape[:2]
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(fw, int(x + w))
            y1 = min(fh, int(y + h))
            self.loc_minimap = (x0, y0)
            if x1 > x0 and y1 > y0:
                self.img_minimap = self.img_frame[y0:y1, x0:x1]
            elif self.frame_count % 60 == 0:
                logger.warning(
                    f"[Minimap] Empty ROI after clamp: raw=({x},{y},{w},{h}) "
                    f"frame=({fw}x{fh})"
                )

            # 绿点卡住超时重定位：超过 dot_stuck_seconds 没动，重置重新定位。
            stuck_seconds = float(self.cfg.get("minimap", {}).get("dot_stuck_seconds", 3.0))
            if (stuck_seconds > 0
                    and self.loc_player_minimap is not None
                    and time.time() - self._dot_last_move_time >= stuck_seconds):
                logger.info("[Dot] 绿点卡住超时，重置重新定位")
                self.loc_player_minimap = None
                self._dot_last_pos = None
                self._dot_last_move_time = time.time()

            # Find player dot — connected components approach
            # Player dot is a small compact cluster; terrain yellow is large/scattered.
            # Prefer the candidate CLOSEST to the last known position: the player dot
            # moves, static terrain doesn't, so "closest" tracks the player reliably.
            if self.img_minimap is not None and self.img_minimap.size > 0:
                player_color = self.cfg.get("minimap", {}).get("player_color", (136, 255, 255))
                base_tol = self.cfg.get("minimap", {}).get("player_tolerance", 15)
                candidates = []
                for tol in [base_tol, base_tol + 10, base_tol + 20, base_tol + 30]:
                    lower = tuple(max(0, c - tol) for c in player_color)
                    upper = tuple(min(255, c + tol) for c in player_color)
                    mask = cv2.inRange(self.img_minimap, lower, upper)
                    num_lbl, lbl, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    for i in range(1, num_lbl):
                        area = stats[i, cv2.CC_STAT_AREA]
                        # Player dot: compact cluster（放宽上限，适配较大尺寸的小地图）
                        if 2 <= area <= 50:
                            cx, cy = centroids[i]
                            dist = 0
                            if self.loc_player_minimap is not None:
                                dist = abs(cx - self.loc_player_minimap[0]) + abs(cy - self.loc_player_minimap[1])
                            candidates.append((dist, area, (int(round(cx)), int(round(cy)))))
                    if candidates:
                        break
                if candidates:
                    # 离上次位置最近的优先；距离相同时选面积更大的。
                    candidates.sort(key=lambda x: (x[0], -x[1]))
                    cx, cy = candidates[0][2]
                    max_jump = int(self.cfg.get("minimap", {}).get("player_max_jump_px", 30))
                    if self.loc_player_minimap is not None:
                        jump = abs(cx - self.loc_player_minimap[0]) + abs(cy - self.loc_player_minimap[1])
                        # <=0 表示不做跳变限制
                        if max_jump <= 0 or jump <= max_jump:
                            self.loc_player_minimap = (cx, cy)
                    else:
                        self.loc_player_minimap = (cx, cy)
                    # 记录绿点是否移动（卡住超时重定位用）。
                    if self.loc_player_minimap != self._dot_last_pos:
                        self._dot_last_pos = self.loc_player_minimap
                        self._dot_last_move_time = time.time()
                    if self.frame_count % 30 == 0:
                        cand_str = [(round(a), round(x), round(y)) for d, a, (x, y) in candidates[:5]]
                        logger.info(f"[Dot] picked=({cx},{cy}) prev={self.loc_player_minimap} cands={cand_str}")

            # Global positioning (layer 3)
            if self.cfg["bot"].get("enable_route") and self.img_map is not None:
                self.get_player_global_position()

        # Route following
        route_action = "none"
        if self.cfg["bot"].get("enable_route") and self.img_route is not None:
            # Ladder detection
            prev = getattr(self, '_prev_loc', None)
            if prev and self.loc_player != (0, 0):
                dx = abs(self.loc_player[0] - prev[0])
                dy = abs(self.loc_player[1] - prev[1])
                if self.is_on_ladder:
                    if dx > 3:
                        self.is_on_ladder = False
                else:
                    if dx < 3 and dy != 0:
                        self.is_on_ladder = True
            self._prev_loc = self.loc_player

            self.update_cmd_by_route()
            route_action = self.cmd_action

        # Detect player via title template matching
        self._detect_player_by_title()

        # Refine with character appearance (handles title loss / same-title conflicts)
        if self.cfg.get("appearance", {}).get("enable", False):
            self._apply_appearance_anchor()

        # Compute search region (same formula for detection & display)
        margin = self.cfg["monster_detect"].get("search_box_margin", 50)
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
        else:
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin

        x0 = max(0, self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0, self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Prepare debug frame
        if self.is_show_debug:
            self.img_frame_debug = self.img_frame.copy()

        # ---------------------------------------------------------------
        # Monster detection — frame skipping for performance
        # ---------------------------------------------------------------
        interval = self.cfg["monster_detect"].get("detection_interval", 2)
        if self.frame_count % interval == 0:
            # Detect monsters on this frame
            self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))

            if self.monsters:
                names = set(m["name"] for m in self.monsters)
                logger.debug(
                    f"Detected {len(self.monsters)} monsters: {names}"
                )

        # Dispatch movement/action command after detection so attack can react this frame.
        if self._kb:
            move_x, move_y = "none", "none"
            action = "none"
            route_move_x, route_move_y = "none", "none"

            if self.cfg["bot"].get("enable_route") and self.img_route is not None:
                route_move_x, route_move_y = self.cmd_move_x, self.cmd_move_y
                move_x, move_y = route_move_x, route_move_y
                action = route_action

            # Attack switch: in hunt mode, combat takes priority over route actions.
            mode = self.cfg.get("bot", {}).get("mode", "detect")
            attack_style = self.cfg.get("bot", {}).get("attack", "directional")
            attack_in_place = bool(
                self.cfg.get("directional_attack", {}).get("attack_in_place", False)
            )
            follow_route = bool(
                self.cfg["bot"].get("enable_route") and self.img_route is not None
            )

            if mode == "hunt" and attack_in_place and not follow_route:
                # 原地攻击 + 定期左右走动（防攻击失效）——仅在不跟路线时
                self._reset_hunt_search_state()
                move_x, move_y, action = self._get_in_place_command()
            elif mode == "hunt" and self.monsters:
                self._reset_hunt_search_state()
                # AoE when monster count >= threshold, else single-target attack.
                aoe_threshold = int(
                    self.cfg.get("directional_attack", {}).get("aoe_monster_count", 3)
                )
                action = "aoe" if len(self.monsters) >= aoe_threshold else "attack"
                if attack_style == "directional":
                    facing = self._get_attack_facing()
                    if facing in ("left", "right"):
                        # 按住方向键朝怪物走（走动能可靠地让角色面向怪物），
                        # 而不是轻点转身（轻点有时被攻击动画吞掉，导致打反方向）。
                        move_x = facing
                        if not self.is_on_ladder:
                            move_y = "none"
                    elif route_move_x in ("left", "right") and self.is_on_ladder:
                        move_x = route_move_x
                    else:
                        # 怪物已在攻击范围内（deadzone 内）：停下攻击，不跟路线漂。
                        if not self.is_on_ladder:
                            move_x = "none"
            elif mode == "hunt" and not follow_route:
                # 没路线时才左右搜索找怪
                self._hunt_no_monster_frames += 1
                if attack_style == "directional":
                    search_x, search_y, search_action = self._get_hunt_search_command()
                    if search_x != "none" or search_y != "none" or search_action != "none":
                        move_x, move_y, action = search_x, search_y, search_action

            # 记录最后一次实际移动的方向（卡住反方向用）。
            if move_x in ("left", "face_left"):
                self._last_walk_dir = "left"
            elif move_x in ("right", "face_right"):
                self._last_walk_dir = "right"

            # 卡墙角/卡住恢复：原地没动超时 → 反方向走2秒脱困。
            # 跟路线用全局坐标；无路线(找怪)用小地图坐标判断是否卡住。
            now = time.time()
            scfg = self.cfg.get("stuck_recovery", {})
            if scfg.get("enable", True):
                if follow_route:
                    cur_pos = self.loc_player_global
                elif self.loc_player_minimap is not None:
                    cur_pos = (float(self.loc_player_minimap[0]), float(self.loc_player_minimap[1]))
                else:
                    cur_pos = None

                if cur_pos is not None and cur_pos != (0.0, 0.0):
                    if self._stuck_last_pos is not None and self._stuck_last_pos != (0.0, 0.0):
                        moved = abs(cur_pos[0] - self._stuck_last_pos[0]) + abs(cur_pos[1] - self._stuck_last_pos[1])
                        if moved >= 1.0:
                            self._stuck_last_move_time = now
                    self._stuck_last_pos = cur_pos

                    if (not self.monsters
                            and now - self._stuck_last_move_time >= float(scfg.get("timeout_seconds", 3.0))):
                        # 反方向：跟最后一次实际移动方向相反。
                        unstick_dir = "right" if self._last_walk_dir == "left" else "left"
                        self._unstick_until = now + float(scfg.get("unstick_seconds", 2.0))
                        self._unstick_dir = unstick_dir
                        self._stuck_last_move_time = now
                        logger.info(f"[Stuck] 卡住超时 → 往{unstick_dir}走2秒")

            if time.time() < self._unstick_until:
                move_x, move_y, action = self._unstick_dir, "none", "none"

            if mode == "hunt" and self.frame_count % 30 == 0:
                logger.info(
                    f"[Hunt] monsters={len(self.monsters)} inPlace={attack_in_place} "
                    f"dispatch=({move_x},{move_y},{action}) on_ladder={self.is_on_ladder} "
                    f"search=(noMob:{self._hunt_no_monster_frames},dir:{self._hunt_search_dir})"
                )

            # 宽度边界：任何移动模式下，绿点都不许超出黄色虚线（左右边界）。
            boundary_pct = float(self.cfg.get("minimap", {}).get("width_boundary_pct", 0.0))
            if (0.0 < boundary_pct < 1.0
                    and self.loc_player_minimap is not None
                    and self.img_minimap is not None):
                mm_w = self.img_minimap.shape[1]
                if mm_w > 0:
                    margin = mm_w * (1.0 - boundary_pct) / 2.0
                    left_b = margin
                    right_b = mm_w - margin
                    mx = self.loc_player_minimap[0]
                    if mx <= left_b and move_x in ("left", "face_left"):
                        move_x = "right"
                        if not self.is_on_ladder:
                            move_y = "none"
                    elif mx >= right_b and move_x in ("right", "face_right"):
                        move_x = "left"
                        if not self.is_on_ladder:
                            move_y = "none"

            self._kb.set_command(f"{move_x} {move_y} {action}")

        # Always draw decorations on debug frame
        if self.is_show_debug:
            self._update_debug_overlay()        # Text first (bottom layer)
            self._draw_search_region(x0, y0, x1, y1)  # Search border
            # Draw minimap rectangle (matches reference project)
            if self.img_minimap is not None:
                draw_rectangle(
                    self.img_frame_debug,
                    self.loc_minimap,
                    self.img_minimap.shape[:2],
                    (0, 0, 255), "minimap", thickness=2
                )
                # Draw player dot position on minimap
                if self.loc_player_minimap:
                    mx, my = self.loc_player_minimap
                    px = self.loc_minimap[0] + mx
                    py = self.loc_minimap[1] + my
                    # Bright green dot + label (BGR: 0,255,0 = green, not yellow)
                    cv2.circle(self.img_frame_debug, (px, py), 6, (0, 255, 0), -1)
                    cv2.putText(self.img_frame_debug, "P", (px+8, py+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # 宽度边界：画黄色虚线（两条竖线，标记可移动范围）
                boundary_pct = float(self.cfg.get("minimap", {}).get("width_boundary_pct", 0.0))
                if 0.0 < boundary_pct < 1.0:
                    mm_h, mm_w = self.img_minimap.shape[:2]
                    margin = mm_w * (1.0 - boundary_pct) / 2.0
                    y0 = self.loc_minimap[1]
                    y1 = self.loc_minimap[1] + mm_h
                    dash_len, gap_len = 8, 5
                    for bx in (self.loc_minimap[0] + margin, self.loc_minimap[0] + mm_w - margin):
                        y = y0
                        while y < y1:
                            cv2.line(self.img_frame_debug, (int(bx), int(y)),
                                     (int(bx), int(min(y + dash_len, y1))), (0, 255, 255), 2)
                            y += dash_len + gap_len
                    if self.frame_count % 60 == 0:
                        logger.info(
                            f"[Boundary] minimap=({self.loc_minimap[0]},{self.loc_minimap[1]}) "
                            f"size={mm_w}x{mm_h} pct={boundary_pct} "
                            f"left_x={self.loc_minimap[0] + margin:.1f} right_x={self.loc_minimap[0] + mm_w - margin:.1f}"
                        )

            # Route minimap preview on right-middle; keep clear of top-right stats.
            if self.img_route is not None and self.loc_player_global != (0.0, 0.0):
                route_bgr = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)
                gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
                crop_w, crop_h = 80, 80
                # Draw player dot on route preview (yellow)
                cv2.circle(route_bgr, (gx, gy), 2, (0, 255, 255), -1)
                # Draw minimap matching rectangle
                if self.loc_minimap_global != (0, 0):
                    mmx, mmy = int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1])
                    mmw = self.img_minimap.shape[1] if self.img_minimap is not None else 0
                    mmh = self.img_minimap.shape[0] if self.img_minimap is not None else 0
                    if mmw > 0 and mmh > 0:
                        cv2.rectangle(route_bgr, (mmx, mmy), (mmx + mmw, mmy + mmh), (0, 255, 255), 1)
                x0 = max(0, gx - crop_w // 2)
                y0 = max(0, gy - crop_h // 2)
                x1 = min(route_bgr.shape[1], x0 + crop_w)
                y1 = min(route_bgr.shape[0], y0 + crop_h)
                if x1 > x0 and y1 > y0:
                    crop = route_bgr[y0:y1, x0:x1]
                    crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                                      interpolation=cv2.INTER_NEAREST)
                    ch, cw = crop.shape[:2]
                    fh, fw = self.img_frame_debug.shape[:2]
                    px_paste = fw - cw - 10
                    # Reserve top-right area for HP/MP + mode labels + monster count.
                    hm_bar_count = int(self.cfg.get("health_monitor", {}).get("bar_count", 3))
                    hm_rows = 2 if hm_bar_count == 2 else 3
                    top_reserved = 30 + 30 * hm_rows + 40
                    if self.monsters:
                        top_reserved += 35

                    # Prefer right-middle to avoid overlapping top-right HUD text.
                    py_preferred = (fh - ch) // 2
                    py_paste = max(top_reserved, py_preferred)
                    py_paste = min(py_paste, max(0, fh - ch - 10))
                    if (py_paste >= 0 and px_paste >= 0
                            and py_paste + ch <= fh and px_paste + cw <= fw):
                        self.img_frame_debug[py_paste:py_paste + ch, px_paste:px_paste + cw] = crop
                        cv2.rectangle(self.img_frame_debug, (px_paste, py_paste),
                                      (px_paste + cw, py_paste + ch), (255, 255, 255), 2)

            self._draw_player_marker()          # Player dot
            self._draw_minimap_selection()      # Manual minimap drag-select box
            self._draw_minimap_magnifier()      # Zoomed cursor view (select mode)
            for monster in self.monsters:       # Monster boxes (top layer)
                if monster["name"] == "Health Bar":
                    color = (0, 255, 255)
                    sim = 1.0
                else:
                    color = (42, 42, 165)  # 棕色
                    # 怪物用 SQDIFF（0=完美，越小越好），转成 0~1 相似度（越大越好）。
                    sim = max(0.0, min(1.0, 1.0 - monster['score']))
                draw_rectangle(
                    self.img_frame_debug,
                    monster["position"], monster["size"],
                    color,
                    f"{monster['name']}:{sim:.2f}",
                    thickness=3, text_height=1.0
                )

        # Pre-scale display frame in background thread
            scale = self.cfg["monster_detect"].get("display_scale", 0.55)
            if scale != 1.0:
                self.img_display = cv2.resize(
                    self.img_frame_debug, None,
                    fx=scale, fy=scale,
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                self.img_display = self.img_frame_debug

            # Signal a new renderable frame for the UI thread.
            self.frame_seq += 1

        self.is_frame_done = True
        return 0

    def toggle_mode(self):
        """Toggle bot mode between "detect" (only visualize) and "hunt" (attack)."""
        current = self.cfg.get("bot", {}).get("mode", "detect")
        new_mode = "hunt" if current == "detect" else "detect"
        self.cfg["bot"]["mode"] = new_mode
        logger.info(f"[Bot] Mode switched to: {new_mode}")
        return new_mode

    def toggle_title_fixed(self):
        """Toggle whether the title/player box is frozen at its current position."""
        self._title_fixed = not self._title_fixed
        logger.info(f"[Title] Box {'FROZEN' if self._title_fixed else 'FOLLOWING'}")
        return self._title_fixed

    def start(self):
        """
        Start the background processing thread and health monitor.
        Matches reference: MapleStoryAutoBot.start() → thread_auto_bot
        """
        # Start health monitor if enabled
        if (self.cfg.get("health_monitor", {}).get("enable", False)
                and self._kb is not None):
            self.health_monitor = HealthMonitor(self.cfg, self._kb)
            self.health_monitor.start()

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("[MonsterDetector] Background processing started.")

    def _process_loop(self):
        """
        Background processing loop. Captures frames and runs detection.
        Display is handled by the main thread in main.py.
        Matches reference: MapleStoryAutoBot.loop()
        """
        if is_windows() and self.capture and not self.test_image:
            activate_game_window(self.capture.window_title)

        logger.info("[MonsterDetector] Processing loop started...")

        while not self.is_terminated:
            t_start = time.time()

            # Process one frame (capture + optional detection)
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"[MonsterDetector] Frame processing error: {e}")
                logger.error(traceback.format_exc())
                continue

            # Cap FPS (matches reference project pattern)
            frame_duration = time.time() - t_start
            try:
                fps_limit = float(self.cfg["system"]["fps_limit_main"])
            except (ValueError, TypeError, KeyError):
                fps_limit = 15.0
            target_duration = 1.0 / max(fps_limit, 0.001)
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

            # FPS shown on overlay should reflect full loop time (including cap sleep).
            loop_duration = time.time() - t_start
            self.fps = round(1.0 / max(loop_duration, 0.001))

    def cleanup(self):
        """Clean up resources."""
        logger.info("[MonsterDetector] Cleaning up...")
        if self.health_monitor:
            self.health_monitor.stop()
        if self.capture:
            self.capture.stop()
        cv2.destroyAllWindows()
        logger.info("[MonsterDetector] Terminated.")

    def save_screenshot(self):
        """Save the current debug frame to disk."""
        if self.img_frame_debug is not None:
            screenshot(self.img_frame_debug, "detection")
        else:
            screenshot(self.img_frame, "raw_frame")
