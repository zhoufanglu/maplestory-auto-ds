"""
Route Recorder for MapleStoryGo.
Generates map.png (stitched minimap) and route*.png (color-coded paths).

Usage:
    python tools/routeRecorder.py --new_map empty_house --cfg custom

Controls:
    Arrow keys + space = record movement
    F1 = toggle recording
    F2 = save screenshot
    F3 = save route*.png
    F4 = save map.png
    Q  = quit

Reference: MapleStoryAutoLevelUp-main/tools/routeRecorder.py
"""
import time
import argparse
import sys
import os
import shutil

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger
from src.utils.common import (
    load_yaml, override_cfg, load_image,
    find_pattern_sqdiff, draw_rectangle, screenshot,
    to_opencv_hsv,
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.KeyBoardListener import KeyBoardListener


class RouteRecorder:
    """
    Records minimap data to generate map.png and route*.png.
    """

    def __init__(self, args):
        self.args = args
        self.idx_routes = 0
        self.fps = 0
        self.frame_count = 0
        self.is_first_frame = True
        self.is_enable = True

        # Coordinates
        self.loc_minimap = (0, 0)
        self.loc_player_minimap = (0, 0)
        self.loc_minimap_global = (0, 0)
        self.loc_player_global = (0.0, 0.0)
        self.loc_player_global_last = None
        self._last_pmm = None  # Last player minimap position (for displacement tracking)
        self._last_minimap = None  # Previous frame's minimap (for frame-to-frame tracking)

        # Images
        self.frame = None
        self.img_frame = None
        self.img_frame_debug = None
        self.img_minimap = None
        self.img_map = None
        self.img_route = None
        self.img_route_debug = None

        # Timers
        self.t_last_frame = time.time()
        self.t_last_draw_blob = 0.0

        # ------------------------------------------------------------------
        # Load config
        # ------------------------------------------------------------------
        logger.info("Loading config for route recorder...")
        self.cfg = load_yaml("config/config_default.yaml")
        custom_path = f"config/config_{args.cfg}.yaml"
        if os.path.exists(custom_path):
            self.cfg = override_cfg(self.cfg, load_yaml(custom_path))
            logger.info(f"  Applied: {custom_path}")

        # Parse color codes: "R,G,B" string -> (R,G,B) int tuple -> action
        self.color_code = {}
        for k, v in self.cfg["route"]["color_code"].items():
            self.color_code[tuple(map(int, k.split(',')))] = v
        for k, v in self.cfg["route"]["color_code_up_down"].items():
            self.color_code[tuple(map(int, k.split(',')))] = v

        # Reverse mapping: action -> RGB color
        self.action_to_rgb = {v: k for k, v in self.color_code.items()}

        # FPS limit
        self.fps_limit = self.cfg["system"].get("fps_limit_route_recorder", 10)

        # ------------------------------------------------------------------
        # Create output directory
        # ------------------------------------------------------------------
        self.map_dir = os.path.join("minimaps", args.new_map)
        if os.path.exists(self.map_dir):
            ans = input(f"Directory '{self.map_dir}' already exists. Replace? [y/N]: ")
            if ans.lower() == 'y':
                shutil.rmtree(self.map_dir)
            else:
                logger.info("Exiting.")
                sys.exit(0)
        os.makedirs(self.map_dir)
        logger.info(f"Output directory: {self.map_dir}")

        # ------------------------------------------------------------------
        # Load existing map if provided
        # ------------------------------------------------------------------
        if args.map:
            self.img_map = load_image(args.map)
            logger.info(f"Loaded existing map: {args.map}")

        # ------------------------------------------------------------------
        # Start keyboard listener & game capture
        # ------------------------------------------------------------------
        self.kb = KeyBoardListener(self.cfg, is_autobot=False)
        self.capture = GameWindowCapturor(self.cfg)
        logger.info("[RouteRecorder] Ready. Press F1 to start recording.")

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def get_img_frame(self):
        """Get the latest game frame at original size."""
        return self.capture.get_frame()

    # ------------------------------------------------------------------
    # Minimap detection
    # ------------------------------------------------------------------

    def _detect_minimap_first_frame(self):
        """Auto-detect minimap via white border, fallback to config."""
        from src.utils.common import get_minimap_loc_size
        result = get_minimap_loc_size(self.img_frame)
        if result:
            x, y, w_m, h_m = result
            logger.info(f"[Minimap] Auto-detected: ({x},{y}) {w_m}x{h_m}")
            return (x + 1, y + 1, w_m - 2, h_m - 2)

        # Fallback to config
        mm = self.cfg.get("minimap", {})
        if mm.get("width", 0) > 10:
            logger.info(f"[Minimap] Using config fallback")
            return (mm["x"], mm["y"], mm["width"], mm["height"])

        logger.error("Cannot detect minimap!")
        return None

    # ------------------------------------------------------------------
    # Global map positioning
    # ------------------------------------------------------------------

    def get_player_location_on_global_map(self):
        """Track global position via template matching (handles X and Y, incl. climbing)."""
        if self.loc_player_minimap is None or self.img_map is None:
            return self.loc_player_global

        # Template match current minimap against global map (ignore black background).
        mask = getattr(self, '_minimap_mask', None)
        loc, score, _ = find_pattern_sqdiff(self.img_map, self.img_minimap, mask=mask)
        self.loc_minimap_global = loc
        self.loc_player_global = (
            loc[0] + self.loc_player_minimap[0],
            loc[1] + self.loc_player_minimap[1],
        )

        if self.frame_count % 30 == 0:
            logger.info(
                f"[Global] score={score:.4f} minimap_loc={loc} "
                f"pmm={self.loc_player_minimap} player_global={self.loc_player_global}"
            )

        # Draw on route debug
        if self.img_route_debug is not None:
            mmx, mmy = int(loc[0]), int(loc[1])
            hm, wm = self.img_minimap.shape[:2]
            cv2.rectangle(self.img_route_debug, (mmx, mmy), (mmx + wm, mmy + hm), (0, 255, 255), 1)
            gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
            cv2.circle(self.img_route_debug, (gx, gy), 2, (0, 255, 255), -1)

        return self.loc_player_global

    # ------------------------------------------------------------------
    # Map stitching
    # ------------------------------------------------------------------

    def ensure_img_map_capacity(self, x, y, h, w):
        """Expand img_map canvas in all directions as needed (X and Y)."""
        x, y = int(x), int(y)  # Float safety
        map_h, map_w = self.img_map.shape[:2]
        pad = self.cfg["route_recoder"]["map_padding"]

        expand_top = max(0, pad - y)
        expand_left = max(0, pad - x)
        expand_bottom = max(0, y + h + pad - map_h)
        expand_right = max(0, x + w + pad - map_w)

        if expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
            return

        new_h = map_h + expand_top + expand_bottom
        new_w = map_w + expand_left + expand_right
        new_map = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_map[expand_top:expand_top + map_h, expand_left:expand_left + map_w] = self.img_map
        self.img_map = new_map

        self.loc_minimap_global = (
            self.loc_minimap_global[0] + expand_left,
            self.loc_minimap_global[1] + expand_top,
        )

    def replace_color_on_map(self, lower_hsv, upper_hsv, replace_color=(0, 0, 0)):
        """Replace pixels in img_map matching HSV range, filtering by component size."""
        hsv_map = cv2.cvtColor(self.img_map, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_map, to_opencv_hsv(lower_hsv), to_opencv_hsv(upper_hsv))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 10:
                self.img_map[labels == i] = replace_color

    def remove_color_code_pixels(self, img):
        """Set any pixel matching a route color code to black."""
        for rgb in self.color_code.keys():
            bgr = (rgb[2], rgb[1], rgb[0])
            mask = np.all(img == bgr, axis=2)
            img[mask] = (0, 0, 0)
        return img

    # ------------------------------------------------------------------
    # Player dot detection on minimap
    # ------------------------------------------------------------------

    def _find_player_on_minimap(self):
        """Find player dot — prefer clusters near last known position."""
        mm = self.cfg.get("minimap", {})
        color = mm.get("player_color", (136, 255, 255))
        base_tol = mm.get("player_tolerance", 15)

        candidates = []
        for tol in [base_tol, base_tol + 10, base_tol + 20, base_tol + 30]:
            lower = tuple(max(0, c - tol) for c in color)
            upper = tuple(min(255, c + tol) for c in color)
            mask = cv2.inRange(self.img_minimap, lower, upper)
            num_lbl, lbl, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, num_lbl):
                area = stats[i, cv2.CC_STAT_AREA]
                if 2 <= area <= 10:  # Smaller range: player dot is tiny
                    cx, cy = centroids[i]
                    # Prefer close to last position
                    dist = 0
                    if self.loc_player_minimap and self.loc_player_minimap != (0, 0):
                        dist = abs(cx - self.loc_player_minimap[0]) + abs(cy - self.loc_player_minimap[1])
                    candidates.append((dist, area, (int(round(cx)), int(round(cy)))))
            if candidates:
                break  # Found candidates at this tolerance level

        if candidates:
            # Sort: closest to last position first, then largest area
            candidates.sort(key=lambda x: (x[0], -x[1]))
            return candidates[0][2]
        return None

    # ------------------------------------------------------------------
    # Keyboard -> action resolution
    # ------------------------------------------------------------------

    def _resolve_action(self):
        """Map keyboard state to an action string + draw_mode flag."""
        keys = self.kb.key_pressing
        action = ""
        is_blob = False

        if "alt" in keys:
            is_blob = True
            if "left" in keys:
                action = "left none jump"
            elif "right" in keys:
                action = "right none jump"
            elif "down" in keys:
                action = "none down jump"
            else:
                action = "none none jump"
        elif "e" in keys:
            is_blob = True
            if "left" in keys:
                action = "left none teleport"
            elif "right" in keys:
                action = "right none teleport"
            elif "down" in keys:
                action = "none down teleport"
            elif "up" in keys:
                action = "none up teleport"
        elif "up" in keys:
            action = "none up none"
        elif "down" in keys:
            action = "none down none"
        elif "left" in keys:
            action = "left none none"
        elif "right" in keys:
            action = "right none none"

        return action, is_blob

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------

    def run_once(self):
        """Process one game frame."""
        self.frame_count += 1

        # 1. Capture frame
        self.img_frame = self.get_img_frame()
        if self.img_frame is None:
            logger.warning("Waiting for game frame...")
            return -1

        self.img_frame_debug = self.img_frame.copy()

        # 2. Detect minimap
        if self.is_first_frame:
            result = self._detect_minimap_first_frame()
            if result is None:
                return -1
            x, y, w_m, h_m = result
            self.loc_minimap = (x, y)
            logger.info(f"[Minimap] Position: ({x},{y}) Size: {w_m}x{h_m}")
        else:
            x, y = self.loc_minimap
            w_m, h_m = self.img_minimap.shape[1], self.img_minimap.shape[0]

        self.img_minimap = self.img_frame[y:y + h_m, x:x + w_m].copy()

        # Create non-black mask BEFORE modifying black pixels
        non_black_mask = np.any(self.img_minimap != [0, 0, 0], axis=2).astype(np.uint8) * 255
        self._minimap_mask = non_black_mask

        # Replace black pixels (0,0,0) -> (1,1,1) to avoid stitching collision
        black_in_minimap = np.all(self.img_minimap == [0, 0, 0], axis=2)
        self.img_minimap[black_in_minimap] = (1, 1, 1)

        # 4. Find player on minimap
        player_loc = self._find_player_on_minimap()
        if player_loc:
            self.loc_player_minimap = player_loc
        if self.frame_count % 10 == 0:
            logger.info(f"[PlayerDot] PMM={player_loc} Global={self.loc_player_global}")

        # 5. Build/update global map
        if self.is_first_frame:
            # Initialize map from first minimap (pad on all sides, matches reference)
            self.img_map = self.img_minimap.copy()
            pad = self.cfg["route_recoder"]["map_padding"]
            self.img_map = cv2.copyMakeBorder(
                self.img_map, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # Remove player yellow dot
            self.replace_color_on_map((55, 40, 80), (60, 100, 100))
            # Remove other player red dot
            self.replace_color_on_map((0, 80, 80), (5, 100, 100))
            # Init route image
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.img_route_debug = self.img_route.copy()

            # Set initial global position
            if self.loc_player_minimap:
                pad = self.cfg["route_recoder"]["map_padding"]
                self.loc_player_global = (
                    pad + self.loc_player_minimap[0],
                    pad + self.loc_player_minimap[1],
                )
                self.loc_minimap_global = (pad, pad)
                self._last_pmm = self.loc_player_minimap
        else:
            # Get player global position FIRST (displacement tracking or match)
            self.loc_player_global = self.get_player_location_on_global_map()
            x_g, y_g = int(self.loc_minimap_global[0]), int(self.loc_minimap_global[1])

            # Expand canvas if needed
            self.ensure_img_map_capacity(x_g, y_g, h_m, w_m)

            # Mask player dot (5px radius) to avoid baking into map
            if self.loc_player_minimap:
                px, py = self.loc_player_minimap
                r = 5
                x1_m = max(0, px - r)
                x2_m = min(w_m, px + r)
                y1_m = max(0, py - r)
                y2_m = min(h_m, py + r)
                self.img_minimap[y1_m:y2_m, x1_m:x2_m] = (0, 0, 0)

            # Stitch: fill black pixels in map with minimap data
            if x_g >= 0 and y_g >= 0:
                mh, mw = self.img_minimap.shape[:2]
                real_h = min(mh, self.img_map.shape[0] - y_g)
                real_w = min(mw, self.img_map.shape[1] - x_g)
                if real_h > 0 and real_w > 0:
                    map_slice = self.img_map[y_g:y_g + real_h, x_g:x_g + real_w]
                    mm_slice = self.img_minimap[:real_h, :real_w]
                    black_in_map = np.all(map_slice == [0, 0, 0], axis=2)
                    map_slice[black_in_map] = mm_slice[black_in_map]

            # Remove other player red dots
            self.replace_color_on_map((0, 78, 78), (5, 100, 100))

        # 6. Show map window
        cv2.imshow("Map", cv2.resize(self.img_map, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST))
        self.img_route_debug = self.img_route.copy()

        # 7. Get player global position
        if not self.is_first_frame:
            self.get_player_location_on_global_map()

        # 8. Resolve keyboard action
        action, is_blob = self._resolve_action()

        # 9. Handle F-keys
        if self.kb.is_pressed_func_key[0]:  # F1
            self.is_enable = not self.is_enable
            self.kb.is_pressed_func_key[0] = False
            logger.info(f"[F1] Recording {'ENABLED' if self.is_enable else 'PAUSED'}")

        if self.kb.is_pressed_func_key[2]:  # F3 — save route (always, even if paused)
            self.kb.is_pressed_func_key[2] = False
            action = "none none goal"
            is_blob = True
            logger.info("[F3] Goal marker placed")
            # Draw goal marker
            rgb = self.action_to_rgb.get(action)
            if rgb:
                gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
                cv2.circle(self.img_route, (gx, gy), 2,
                          (rgb[2], rgb[1], rgb[0]), -1)
            # Save route AND map (same dimensions, avoids size mismatch)
            route_path = os.path.join(self.map_dir, f"route{self.idx_routes + 1}.png")
            map_path = os.path.join(self.map_dir, "map.png")
            cv2.imwrite(route_path, self.img_route)
            cv2.imwrite(map_path, self.img_map)
            logger.info(f"Saved: {route_path} + map.png ({self.img_map.shape[1]}x{self.img_map.shape[0]})")
            self.idx_routes += 1
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.loc_player_global_last = None
            action = ""  # Prevent double-processing below

        # 10. Draw on route image (only when recording is enabled)
        if self.is_enable and action:
            rgb = self.action_to_rgb.get(action)
            if rgb:
                color_bgr = (rgb[2], rgb[1], rgb[0])
                px, py = int(self.loc_player_global[0]), int(self.loc_player_global[1])

                if is_blob:
                    if time.time() - self.t_last_draw_blob > self.cfg["route_recoder"]["blob_cooldown"]:
                        cv2.circle(self.img_route, (px, py), 2, color_bgr, -1)
                        self.t_last_draw_blob = time.time()
                        self.loc_player_global_last = None
                else:
                    if self.loc_player_global_last:
                        dist = abs(px - self.loc_player_global_last[0]) + abs(py - self.loc_player_global_last[1])
                        if dist < 30:
                            cv2.line(self.img_route, self.loc_player_global_last, (px, py), color_bgr, 1)
                    self.loc_player_global_last = (px, py)

        # 11. Save map on F4 (always, even if paused)
        if self.kb.is_pressed_func_key[3]:
            self.kb.is_pressed_func_key[3] = False
            out = os.path.join(self.map_dir, "map.png")
            cv2.imwrite(out, self.img_map)
            logger.info(f"Saved: {out}")

        # 13. Debug display
        self._update_debug_display()

        # 14. Screenshot on F2
        if self.kb.is_pressed_func_key[1]:
            screenshot(self.img_frame, "route_recorder")
            self.kb.is_pressed_func_key[1] = False

        self.is_first_frame = False
        return 0

    def _update_debug_display(self):
        """Draw text overlays and show debug windows."""
        h, w = self.img_frame.shape[:2]

        texts = [
            f"FPS: {self.fps} | Rec: {'ON' if self.is_enable else 'OFF'} | Keys: {self.kb.key_pressing}",
            f"Route: {self.idx_routes + 1} | Action: {self._resolve_action()[0] or 'idle'}",
            f"Global: ({self.loc_player_global[0]},{self.loc_player_global[1]}) Map: {self.img_map.shape[1]}x{self.img_map.shape[0]}",
            "F1=toggle F3=save Q=quit",
        ]
        for i, text in enumerate(texts):
            cv2.putText(self.img_frame_debug, text, (10, 30 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw minimap rectangle
        x, y = self.loc_minimap
        h_m, w_m = self.img_minimap.shape[:2]
        cv2.rectangle(self.img_frame_debug, (x, y), (x + w_m, y + h_m), (0, 0, 255), 2)

        # Draw player dot on minimap overlay (green "P", same as detector)
        if self.loc_player_minimap:
            px = x + self.loc_player_minimap[0]
            py = y + self.loc_player_minimap[1]
            cv2.circle(self.img_frame_debug, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(self.img_frame_debug, "P", (px + 8, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Paste route minimap preview in top-right corner
        if self.img_route_debug is not None and self.loc_player_global != (0.0, 0.0):
            gx, gy = int(self.loc_player_global[0]), int(self.loc_player_global[1])
            crop_w, crop_h = 80, 80
            x0 = max(0, gx - crop_w // 2)
            y0 = max(0, gy - crop_h // 2)
            x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
            y1 = min(self.img_route_debug.shape[0], y0 + crop_h)
            if x1 > x0 and y1 > y0:
                crop = self.img_route_debug[y0:y1, x0:x1]
                crop = cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3),
                                  interpolation=cv2.INTER_NEAREST)
                ch, cw = crop.shape[:2]
                px_paste = w - cw - 10
                py_paste = 10
                self.img_frame_debug[py_paste:py_paste + ch, px_paste:px_paste + cw] = crop
                cv2.rectangle(self.img_frame_debug, (px_paste, py_paste),
                              (px_paste + cw, py_paste + ch), (255, 255, 255), 2)

        # Show debug window scaled to match main detector's display
        scale = self.cfg["monster_detect"].get("display_scale", 0.55)
        display = self.img_frame_debug
        if scale != 1.0:
            display = cv2.resize(display, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Game Window Debug", display)

        # Pin to bottom-right on first frame
        if self.is_first_frame:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
                rect = cv2.getWindowImageRect("Game Window Debug")
                if rect:
                    _, _, win_w, win_h = rect
                    cv2.moveWindow("Game Window Debug",
                                   screen_w - win_w - 20, screen_h - win_h - 80)
            except Exception:
                pass

        self.fps = round(1.0 / max(time.time() - self.t_last_frame, 0.001))
        self.t_last_frame = time.time()


# ======================================================================
# Main entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapleStoryGo Route Recorder")
    parser.add_argument("--new_map", type=str, default="empty_house",
                        help="Map name for output directory")
    parser.add_argument("--cfg", type=str, default="custom",
                        help="Config suffix to load")
    parser.add_argument("--map", type=str, default="",
                        help="Path to existing map.png to extend")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Route Recorder — new_map={args.new_map}")
    logger.info("=" * 60)

    recorder = RouteRecorder(args)

    try:
        while True:
            t_start = time.time()
            recorder.run_once()
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed.")
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        recorder.capture.stop()
        recorder.kb.stop()
        cv2.destroyAllWindows()
        logger.info("Done.")
