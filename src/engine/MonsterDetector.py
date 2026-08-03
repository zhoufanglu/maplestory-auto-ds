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

        # Minimap
        self.loc_minimap = (0, 0)
        self.img_minimap = None
        self.loc_player_minimap = None

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
        self._kb = None  # Keyboard controller (set externally)

        # Load title template for player detection
        self.img_title = None
        self.img_title_gray = None
        if os.path.exists("character/title.png"):
            self.img_title = load_image("character/title.png")
            self.img_title_gray = load_image("character/title.png", cv2.IMREAD_GRAYSCALE)
            logger.info(f"  Loaded title template: {self.img_title.shape[1]}x{self.img_title.shape[0]}")
        else:
            logger.warning("  No character/title.png found, player will stay at frame center")

        # Timers
        self.t_last_frame = time.time()
        self.frame_count = 0
        self.fps = 0
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
        # Load specified monsters from config, or scan all directories
        specified = self.cfg["monster_detect"].get("monsters", [])
        if specified:
            monster_dirs = [os.path.join("monster", m) for m in specified
                            if os.path.isdir(os.path.join("monster", m))]
            if not monster_dirs:
                logger.error(f"Specified monster dirs not found: {specified}")
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
        diff_thres = self.cfg["monster_detect"]["diff_thres"]
        h, w = img_monster.shape[:2]

        # Multi-scale matching (configured via monster_detect.scales)
        scales = self.cfg["monster_detect"].get("scales", [1.0])

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
            count = 0
            for pt in zip(*match_locations[::-1]):
                if count >= 50:
                    break
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
        mask = cv2.inRange(img_roi, hp_color, hp_color)

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

    def _detect_player_by_title(self):
        """Detect player location by matching the title template (at character feet)."""
        if self.img_title is None:
            h, w = self.img_frame.shape[:2]
            self.loc_player = (w // 2, h // 2)
            return

        h, w = self.img_frame.shape[:2]
        title_cfg = self.cfg.get("title", {})
        title_offset = title_cfg.get("offset", [0, -60])

        # White-mask binarization: extract only bright pixels from template + frame
        # (参考项目 nametag white_mask 模式，对文字类模板更准)
        lower_white = title_cfg.get("white_lower", 150)
        upper_white = title_cfg.get("white_upper", 255)
        img_nametag_bin = cv2.inRange(self.img_title_gray, lower_white, upper_white)
        img_roi_bin = cv2.inRange(self.img_frame_gray, lower_white, upper_white)

        # Gaussian blur to soften edges
        blur_k = title_cfg.get("blur_kernel", 3)
        img_nametag_bin = cv2.GaussianBlur(img_nametag_bin, (blur_k, blur_k), 0)
        img_roi_bin = cv2.GaussianBlur(img_roi_bin, (blur_k, blur_k), 0)

        # Match binary masks
        match_threshold = title_cfg.get("match_threshold", 0.3)
        loc, score, is_cached = find_pattern_sqdiff(
            img_roi_bin, img_nametag_bin,
            last_result=self.loc_title,
            global_threshold=match_threshold
        )

        # Log score every frame for first 100 frames, then every 30 frames
        if self.frame_count <= 100 or self.frame_count % 30 == 0:
            status = "HIT" if score < match_threshold else "MISS"
            logger.info(
                f"[Title] frame={self.frame_count} score={score:.4f} "
                f"thresh={match_threshold:.2f} {status} cached={is_cached}"
            )

        if score < match_threshold:
            self.loc_title = loc
            tw, th = self.img_title.shape[1], self.img_title.shape[0]
            self.loc_player = (
                loc[0] + tw // 2 + title_offset[0],
                loc[1] + th // 2 + title_offset[1]
            )
            if self.img_frame_debug is not None:
                draw_rectangle(
                    self.img_frame_debug,
                    loc, (th, tw), (255, 0, 0),
                    f"TITLE {score:.2f}", thickness=3, text_height=0.7
                )
        elif self.loc_title is None:
            h, w = self.img_frame.shape[:2]
            self.loc_player = (w // 2, h // 2)

    def _detect_minimap(self):
        """
        Detect minimap via white border (reference project pattern).
        Falls back to config if detection fails.
        """
        # Try auto-detection first (white border method)
        result = get_minimap_loc_size(self.img_frame)
        if result:
            x, y, w_m, h_m = result
            if self.frame_count == 1:
                logger.info(f"[Minimap] Auto-detected: ({x},{y}) {w_m}x{h_m}")
            return (x + 1, y + 1, w_m - 2, h_m - 2)

        # Fallback to config
        mm_cfg = self.cfg.get("minimap", {})
        if mm_cfg.get("width", 0) > 10:
            return (mm_cfg["x"], mm_cfg["y"], mm_cfg["width"], mm_cfg["height"])
        return None

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
        """Track global position: X=player dot, Y=manual offset from climbing."""
        if self.img_map is None or self.img_minimap is None:
            return
        if self.loc_player_minimap is None:
            return

        # Initial positioning: template match to find starting point
        if self.loc_player_global == (0, 0):
            loc, score, _ = find_pattern_sqdiff(self.img_map, self.img_minimap)
            self._last_match_score = score
            self.loc_minimap_global = loc
            self.loc_player_global = (
                loc[0] + self.loc_player_minimap[0],
                loc[1] + self.loc_player_minimap[1],
            )
            self._last_pmm = self.loc_player_minimap
            return

        # X + Y: track via player dot displacement
        dx = self.loc_player_minimap[0] - self._last_pmm[0]
        dy = self.loc_player_minimap[1] - self._last_pmm[1]
        if abs(dx) < 20 and abs(dy) < 20:
            self.loc_player_global = (
                self.loc_player_global[0] + dx,
                self.loc_player_global[1] + dy,
            )

        # Fallback: Y offset when climbing (supports 0.5 etc)
        offset = float(self.cfg["route"].get("climb_y_offset", 1))
        interval = int(self.cfg["route"].get("climb_y_interval", 15))
        if self.frame_count % interval == 0:
            if self.cmd_move_y == "up":
                self.loc_player_global = (self.loc_player_global[0], self.loc_player_global[1] - offset)
            elif self.cmd_move_y == "down":
                self.loc_player_global = (self.loc_player_global[0], self.loc_player_global[1] + offset)

        self._last_pmm = self.loc_player_minimap
        self.loc_minimap_global = (
            self.loc_player_global[0] - self.loc_player_minimap[0],
            self.loc_player_global[1] - self.loc_player_minimap[1],
        )

        # Periodic drift correction (every 60 frames)
        if self.frame_count % 60 == 0:
            loc, score, _ = find_pattern_sqdiff(self.img_map, self.img_minimap)
            self._last_match_score = score
            if score < 0.15:
                self.loc_minimap_global = loc
                self.loc_player_global = (
                    loc[0] + self.loc_player_minimap[0],
                    loc[1] + self.loc_player_minimap[1],
                )

    def get_nearest_route_color(self):
        """Scan route image around player for nearest color-coded pixel."""
        if self.img_route is None or self.loc_player_global == (0.0, 0.0):
            return None, None

        x0, y0 = int(self.loc_player_global[0]), int(self.loc_player_global[1])
        h, w = self.img_route.shape[:2]
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
                pixel = tuple(self.img_route[y, x])
                dist = abs(x - x0) + abs(y - y0)
                if pixel in self.color_code and dist < min_dist:
                    nearest = {"pixel": (x, y), "command": self.color_code[pixel], "distance": dist}
                    min_dist = dist
                if pixel in self.color_code_up_down and dist < min_dist_ud:
                    nearest_ud = {"pixel": (x, y), "command": self.color_code_up_down[pixel], "distance": dist}
                    min_dist_ud = dist

        return nearest, nearest_ud

    def update_cmd_by_route(self):
        """Convert nearest route color to movement commands."""
        nearest, nearest_ud = self.get_nearest_route_color()

        if nearest and nearest_ud:
            if nearest["distance"] < nearest_ud["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest["command"].split()
                _, cmd, _ = nearest_ud["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest_ud["command"].split()
                cmd, _, _ = nearest["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd
        elif nearest:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest["command"].split()
        elif nearest_ud:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = nearest_ud["command"].split()
        else:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = "none", "none", "none"

        # Goal → cycle to next route
        if self.cmd_action == "goal" and self.img_routes:
            self.idx_routes = (self.idx_routes + 1) % len(self.img_routes)
            self.img_route = self.img_routes[self.idx_routes]
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

    def _update_debug_overlay(self):
        """Add FPS and status info to the debug frame."""
        self.fps = round(1.0 / max(time.time() - self.t_last_frame, 0.001))

        diff_thres = self.cfg["monster_detect"]["diff_thres"]
        route_on = self.cfg["bot"].get("enable_route", False)
        texts = [
            f"FPS: {self.fps} | KB: {'ON' if self._kb and self._kb.is_enable else 'OFF'}",
            f"Monsters: {len(self.monsters)} | Best: {self.best_score:.4f}/{diff_thres}",
            f"Mode: {self.cfg['monster_detect']['mode']} | Scale: {self.cfg.get('monster_detect',{}).get('display_scale',1)}",
            f"Frame: {self.img_frame.shape[1]}x{self.img_frame.shape[0]}",
        ]

        if route_on:
            gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
            mx, my = int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1])
            texts += [
                f"Route: {self.idx_routes+1}/{len(self.img_routes)} Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
                f"Global: ({gx},{gy}) MM:({mx},{my}) PMM:{self.loc_player_minimap or '?'} Match:{getattr(self,'_last_match_score',1):.3f}",
                f"Ladder: {self.is_on_ladder} | Map: {self.img_map.shape[1]}x{self.img_map.shape[0]}" if self.img_map is not None else "No map loaded",
            ]
        texts += ["S=shot Q=quit F1=toggle"]

        # Bottom-left positioning (matches reference project)
        text_y_interval = 25
        text_y_start = self.img_frame.shape[0] - len(texts) * text_y_interval - 10
        font_scale = 0.8  # +2 sizes from 0.6

        for i, text in enumerate(texts):
            x, y = 10, text_y_start + text_y_interval * i
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cv2.putText(
                    self.img_frame_debug, text,
                    (x + dx, y + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA
                )
            cv2.putText(
                self.img_frame_debug, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2, cv2.LINE_AA
            )

        # Monster count in top-right corner (big text, eye-catching)
        if self.monsters:
            count_text = f"Monsters: {len(self.monsters)}"
            (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(
                self.img_frame_debug, count_text,
                (self.img_frame.shape[1] - tw - 15, th + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
            )

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

        # Detect minimap (adapted for MapleStoryGo: bright fill + dark borders)
        minimap_result = self._detect_minimap()
        if minimap_result:
            x, y, w, h = minimap_result
            self.loc_minimap = (x, y)
            self.img_minimap = self.img_frame[y:y + h, x:x + w]

            # Find player dot — connected components approach
            # Player dot is a small compact cluster; terrain yellow is large/scattered
            player_color = self.cfg.get("minimap", {}).get("player_color", (136, 255, 255))
            base_tol = self.cfg.get("minimap", {}).get("player_tolerance", 15)
            for tol in [base_tol, base_tol + 10, base_tol + 20, base_tol + 30]:
                lower = tuple(max(0, c - tol) for c in player_color)
                upper = tuple(min(255, c + tol) for c in player_color)
                mask = cv2.inRange(self.img_minimap, lower, upper)
                # Find connected components, pick the best one (not terrain blob)
                num_lbl, lbl, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                best = None
                best_score = -1
                for i in range(1, num_lbl):
                    area = stats[i, cv2.CC_STAT_AREA]
                    # Player dot: 3-15 pixels, compact
                    if 3 <= area <= 15:
                        # Prefer larger clusters (more confident match)
                        if area > best_score:
                            best_score = area
                            best = centroids[i]
                if best is not None:
                    self.loc_player_minimap = (int(round(best[0])), int(round(best[1])))
                    break

            # Global positioning (layer 3)
            if self.cfg["bot"].get("enable_route") and self.img_map is not None:
                self.get_player_global_position()

        # Route following
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
            if self._kb:
                self._kb.set_command(
                    f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}"
                )

        # Detect player via title template matching
        self._detect_player_by_title()

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

            # Route minimap preview in top-right corner (matches reference project)
            if self.img_route is not None and self.loc_player_global != (0.0, 0.0):
                route_bgr = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)
                gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
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
                    py_paste = 10
                    self.img_frame_debug[py_paste:py_paste + ch, px_paste:px_paste + cw] = crop
                    cv2.rectangle(self.img_frame_debug, (px_paste, py_paste),
                                  (px_paste + cw, py_paste + ch), (255, 255, 255), 2)

            self._draw_player_marker()          # Player dot
            for monster in self.monsters:       # Monster boxes (top layer)
                color = (0, 255, 255) if monster["name"] == "Health Bar" else (0, 255, 0)
                draw_rectangle(
                    self.img_frame_debug,
                    monster["position"], monster["size"],
                    color,
                    f"{monster['name']}:{monster['score']:.2f}",
                    thickness=2, text_height=0.5
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

        # Update FPS timer
        self.t_last_frame = time.time()

        self.is_frame_done = True
        return 0

    def start(self):
        """
        Start the background processing thread.
        Matches reference: MapleStoryAutoBot.start() → thread_auto_bot
        """
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
            self.run_once()

            # Cap FPS (matches reference project pattern)
            frame_duration = time.time() - t_start
            target_duration = 1.0 / self.cfg["system"]["fps_limit_main"]
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

    def cleanup(self):
        """Clean up resources."""
        logger.info("[MonsterDetector] Cleaning up...")
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
