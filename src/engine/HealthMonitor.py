"""
Health Monitor — monitors HP/MP/EXP bars and auto-uses potions.

Runs in a separate thread, receiving frame updates from the main thread.
When HP or MP drops below configured thresholds, presses the corresponding
potion key.

Adapted from MapleStoryAutoLevelUp-main/src/engine/HealthMonitor.py
"""
import threading
import time

import cv2

from src.utils.logger import logger
from src.utils.common import get_bar_percent
from src.input.KeyBoardController import press_key


class HealthMonitor:
    """
    Independent health monitoring thread that can heal while other actions run.
    """

    def __init__(self, cfg: dict, kb_controller):
        self.cfg = cfg
        self.kb = kb_controller
        self.is_terminated = False
        self.thread = None

        # Separate HP/MP enable flags
        self.enable_hp = self.cfg.get("health_monitor", {}).get("enable_hp", True)
        self.enable_mp = self.cfg.get("health_monitor", {}).get("enable_mp", True)

        # Health monitoring state
        self.hp_percent = 100.0
        self.mp_percent = 100.0
        self.exp_percent = 100.0
        self.t_last_valid_read = 0.0
        # Edge-trigger latches: prevent repeated drinking while value stays low.
        self._hp_potion_latched = False
        self._mp_potion_latched = False

        # Timers
        self.t_last_heal = 0.0
        self.t_last_mp = 0.0
        self.t_last_hp_reduce = 0.0
        self.t_last_run = 0.0
        self.t_hp_watch_dog = time.time()

        # Frame data (updated by main thread)
        self.img_frame = None
        self.frame_lock = threading.Lock()

        # FPS
        self.fps_limit = self.cfg.get("health_monitor", {}).get("fps_limit", 20)
        self.fps = 0

        # Debug: bar bounding boxes [(x, y, w, h), ...]
        self.loc_size_bars = [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)]
        self._bar_diag_count = 0

        logger.info("[HealthMonitor] Initialized")

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def start(self):
        """Start the health monitoring thread."""
        if not self.is_terminated:
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("[HealthMonitor] Started")

    def stop(self):
        """Stop the health monitoring thread."""
        self.is_terminated = True
        if self.thread:
            self.thread.join(timeout=1.0)
            logger.info("[HealthMonitor] Stopped")
 
    def toggle_hp(self):
        """Toggle HP auto-heal on/off (F2 hotkey)."""
        self.enable_hp = not self.enable_hp
        logger.info(f"[HealthMonitor] HP auto-heal: {'ON' if self.enable_hp else 'OFF'}")

    def toggle_mp(self):
        """Toggle MP auto-recovery on/off (F3 hotkey)."""
        self.enable_mp = not self.enable_mp
        logger.info(f"[HealthMonitor] MP auto-recovery: {'ON' if self.enable_mp else 'OFF'}")

    # -------------------------------------------------------------------
    # Frame update (called from main thread)
    # -------------------------------------------------------------------

    def update_frame(self, img_frame):
        """Update the frame slice (UI bar area) from the main thread."""
        with self.frame_lock:
            if img_frame is None or getattr(img_frame, "size", 0) == 0:
                self.img_frame = None
                return
            self.img_frame = img_frame

    # -------------------------------------------------------------------
    # Bar percentage extraction
    # -------------------------------------------------------------------

    @staticmethod
    def _classify_bar(img_bar) -> str:
        """
        Classify a bar image as 'HP', 'MP', or 'EXP' by sampling the
        color of filled (non-white, non-gray) pixels in the center row.

        Returns: 'HP', 'MP', 'EXP', or 'unknown'.
        """
        h, w = img_bar.shape[:2]
        line = img_bar[h // 2, :]

        # Collect pixels that are not white (border) and not dark-gray (empty)
        filled = []
        for pixel in line:
            r, g, b = int(pixel[2]), int(pixel[1]), int(pixel[0])
            # Skip white/gray (border & empty)
            if r > 200 and g > 200 and b > 200:
                continue
            if abs(r - g) < 15 and abs(r - b) < 15:
                continue
            filled.append((r, g, b))

        if len(filled) < 5:
            return 'unknown'

        avg_r = sum(p[0] for p in filled) / len(filled)
        avg_g = sum(p[1] for p in filled) / len(filled)
        avg_b = sum(p[2] for p in filled) / len(filled)

        # Red-dominant → HP
        if avg_r > avg_g + 20 and avg_r > avg_b + 20:
            return 'HP'
        # Blue-dominant → MP
        if avg_b > avg_r + 20 and avg_b > avg_g + 20:
            return 'MP'
        # Yellow/green → EXP
        if avg_g > avg_b + 20:
            return 'EXP'

        return 'unknown'

    def get_hp_mp_exp_percent(self):
        """
        Extract HP, MP, and EXP bar fill ratios from the UI area.

        Strategy: find white-bordered rectangles, then classify each
        by fill color (red=HP, blue=MP, yellow/green=EXP).

        Returns:
            (hp_pct, mp_pct, exp_pct) each 0–100, or (None, None, None).
        """
        if self.img_frame is None:
            return None, None, None

        with self.frame_lock:
            img_frame = self.img_frame.copy()

        if img_frame is None or img_frame.size == 0:
            return None, None, None
        if len(img_frame.shape) < 2 or img_frame.shape[0] <= 0 or img_frame.shape[1] <= 0:
            return None, None, None

        hm_cfg = self.cfg.get("health_monitor", {})

        # 固定坐标模式：游戏 UI 更新后血条位置固定，直接按配置坐标裁剪。
        # 白框自动检测在条填充过半时会把白框切断、只抓到空槽部分，导致漏识别，
        # 所以这里优先走固定坐标（更可靠）。
        if hm_cfg.get("fixed_bars", False):
            return self._get_percent_fixed_bars(img_frame, hm_cfg)

        # 白框自动检测：裁剪底部 UI 条，减少白框搜索范围。
        ui_y = self.cfg.get("ui_coords", {}).get("ui_y_start", 610)
        if 0 <= ui_y < img_frame.shape[0]:
            img_frame = img_frame[ui_y:, :]
        else:
            return None, None, None

        bar_min_aspect = hm_cfg.get("bar_min_aspect", 2.5)
        bar_max_aspect = hm_cfg.get("bar_max_aspect", 12.0)
        bar_min_area = hm_cfg.get("bar_min_area", 800)
        bar_max_area = hm_cfg.get("bar_max_area", 8000)
        # Additional size guards reduce false positives from other white UI boxes.
        bar_min_width = hm_cfg.get("bar_min_width", 90)
        bar_max_width = hm_cfg.get("bar_max_width", 220)
        bar_min_height = hm_cfg.get("bar_min_height", 14)
        bar_max_height = hm_cfg.get("bar_max_height", 24)
        bar_count = hm_cfg.get("bar_count", 3)
        white_thresh = hm_cfg.get("bar_white_thresh", 240)

        img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(img_gray, white_thresh, 255)

        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_h, frame_w = img_frame.shape[:2]
        all_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            all_candidates.append((x, y, w, h, w / h if h > 0 else 0, w * h))

        loc_size_bars = []
        for x, y, w, h, asp, area in all_candidates:
            if (bar_min_aspect < asp < bar_max_aspect
                    and bar_min_area < area < bar_max_area
                    and bar_min_width <= w <= bar_max_width
                    and bar_min_height <= h <= bar_max_height):
                loc_size_bars.append((x, y, w, h))
        # Deterministic order helps fallback classification when colors are ambiguous.
        loc_size_bars = sorted(loc_size_bars, key=lambda b: (b[1], b[0]))

        # --- Color-based classification ---
        # Instead of position-guessing, sample each bar's fill color to decide type.
        hp_bar = mp_bar = exp_bar = None
        for x, y, w, h in loc_size_bars:
            bar_img = img_frame[y:y + h, x:x + w]
            if bar_img is None or bar_img.size == 0:
                continue
            bar_type = self._classify_bar(bar_img)
            if bar_type == 'HP' and hp_bar is None:
                hp_bar = (x, y, w, h)
            elif bar_type == 'MP' and mp_bar is None:
                mp_bar = (x, y, w, h)
            elif bar_type == 'EXP' and exp_bar is None:
                exp_bar = (x, y, w, h)

        # Fallback: if color classification failed, use position order
        remaining = [b for b in loc_size_bars
                     if b not in (hp_bar, mp_bar, exp_bar)]
        if hp_bar is None and remaining:
            hp_bar = remaining.pop(0)
        if mp_bar is None and remaining:
            mp_bar = remaining.pop(0)
        if exp_bar is None and remaining and bar_count >= 3:
            exp_bar = remaining.pop(0)

        # Build ordered list
        if bar_count == 2:
            ordered = [b for b in (hp_bar, mp_bar) if b is not None]
        else:
            ordered = [b for b in (hp_bar, mp_bar, exp_bar) if b is not None]

        # Diagnostic logging
        if not hasattr(self, '_bar_diag_count'):
            self._bar_diag_count = 0
        self._bar_diag_count += 1

        if self._bar_diag_count == 1 or self._bar_diag_count % 90 == 0:
            logger.info(
                f"[HealthMonitor] UI area: {frame_w}x{frame_h}, "
                f"white_thresh={white_thresh}, "
                f"total_white_contours={len(all_candidates)}, "
                f"bar_candidates={len(loc_size_bars)}, "
                f"need={bar_count}, "
                f"w={bar_min_width}-{bar_max_width}, "
                f"h={bar_min_height}-{bar_max_height}"
            )
            if all_candidates:
                sorted_cands = sorted(all_candidates, key=lambda c: c[5], reverse=True)[:5]
                for i, (x, y, w, h, asp, area) in enumerate(sorted_cands):
                    logger.info(
                        f"  #{i+1}: ({x},{y}) {w}x{h} "
                        f"aspect={asp:.1f} area={area}"
                    )
            # Log classification results
            bar_types = []
            if hp_bar: bar_types.append('HP')
            if mp_bar: bar_types.append('MP')
            if exp_bar: bar_types.append('EXP')
            logger.info(f"[HealthMonitor] Classified bars: {bar_types}")
            logger.info(
                f"[HealthMonitor] Selected boxes: HP={hp_bar} MP={mp_bar} EXP={exp_bar}"
            )

            if len(ordered) < bar_count:
                logger.warning(
                    f"[HealthMonitor] Found {len(ordered)}/{bar_count} bars. "
                    f"Try adjusting bar_min_aspect/bar_max_aspect/bar_min_area/bar_max_area "
                    f"or bar_white_thresh in config."
                )

        if len(ordered) < bar_count:
            return None, None, None

        self.loc_size_bars = ordered

        percents = []
        for x, y, w, h in ordered:
            bar_crop = img_frame[y:y + h, x:x + w]
            if bar_crop is None or bar_crop.size == 0:
                return None, None, None
            pct = get_bar_percent(bar_crop)
            percents.append(pct)

        if bar_count == 2:
            hp_pct, mp_pct = percents[0], percents[1]
            if self._bar_diag_count % 60 == 0:
                logger.info(
                    f"[HealthMonitor] HP/MP read: HP={hp_pct:.1f}% MP={mp_pct:.1f}% "
                    f"thres=({hm_cfg.get('add_hp_percent', 50)},{hm_cfg.get('add_mp_percent', 50)})"
                )
            return hp_pct, mp_pct, 0.0
        if self._bar_diag_count % 60 == 0:
            logger.info(
                f"[HealthMonitor] HP/MP read: HP={percents[0]:.1f}% MP={percents[1]:.1f}% "
                f"thres=({hm_cfg.get('add_hp_percent', 50)},{hm_cfg.get('add_mp_percent', 50)})"
            )
        return tuple(percents)

    def _get_percent_fixed_bars(self, img_frame, hm_cfg):
        """
        Fixed-coordinate mode: crop HP/MP/EXP bars from configured rects
        instead of auto-detecting white-bordered bars.

        Coordinate rects are [x, y, w, h]. Values may be absolute pixels
        (any value > 1.0) or normalized fractions of the frame (0.0 ~ 1.0),
        so the same config survives window resizing when using fractions.
        """
        frame_h, frame_w = img_frame.shape[:2]
        bar_count = int(hm_cfg.get("bar_count", 2))
        keys = ["hp_bar_rect", "mp_bar_rect", "exp_bar_rect"][:bar_count]

        rects = []
        for key in keys:
            rect = hm_cfg.get(key)
            if not rect or len(rect) != 4:
                logger.warning(f"[HealthMonitor] 缺少固定条坐标配置: {key}")
                return None, None, None
            x, y, w, h = [float(v) for v in rect]
            if max(x, y, w, h) <= 1.0:  # 比例坐标 (0~1)
                x, w = x * frame_w, w * frame_w
                y, h = y * frame_h, h * frame_h
            x, y, w, h = int(x), int(y), int(w), int(h)
            if (x < 0 or y < 0 or x + w > frame_w or y + h > frame_h):
                logger.warning(
                    f"[HealthMonitor] 条坐标越界: ({x},{y},{w},{h}) "
                    f"frame={frame_w}x{frame_h}"
                )
                return None, None, None
            rects.append((x, y, w, h))

        self.loc_size_bars = rects

        percents = []
        for x, y, w, h in rects:
            crop = img_frame[y:y + h, x:x + w]
            if crop is None or crop.size == 0:
                return None, None, None
            percents.append(get_bar_percent(crop))

        if bar_count == 2:
            hp_pct, mp_pct = percents[0], percents[1]
            if self._bar_diag_count % 60 == 0:
                logger.info(
                    f"[HealthMonitor] HP/MP fixed: HP={hp_pct:.1f}% MP={mp_pct:.1f}% "
                    f"thres=({hm_cfg.get('add_hp_percent', 50)},{hm_cfg.get('add_mp_percent', 50)})"
                )
            return hp_pct, mp_pct, 0.0
        return tuple(percents)

    # -------------------------------------------------------------------
    # Monitoring loop (runs in daemon thread)
    # -------------------------------------------------------------------

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.is_terminated:
            try:
                t_cur = time.time()

                # Read bar percentages
                result = self.get_hp_mp_exp_percent()
                if result[0] is not None:
                    hp_pct, mp_pct, exp_pct = result

                    if self.hp_percent > hp_pct:
                        self.t_last_hp_reduce = t_cur
                    self.hp_percent = hp_pct
                    self.mp_percent = mp_pct
                    self.exp_percent = exp_pct
                    self.t_last_valid_read = t_cur

                hm_cfg = self.cfg.get("health_monitor", {})
                hp_thres = hm_cfg.get("add_hp_percent", 50)
                mp_thres = hm_cfg.get("add_mp_percent", 50)
                hp_cd = hm_cfg.get("add_hp_cooldown", 0.5)
                mp_cd = hm_cfg.get("add_mp_cooldown", 0.5)
                watchdog_to = hm_cfg.get("return_home_watch_dog_timeout", 3)
                valid_timeout = hm_cfg.get("valid_read_timeout", 1.5)
                hp_rearm = hm_cfg.get("hp_rearm_percent", min(100, hp_thres + 5))
                mp_rearm = hm_cfg.get("mp_rearm_percent", min(100, mp_thres + 5))
                # If value stays low for too long (e.g., continuous attack drain),
                # release latch and allow one more potion after retry timeout.
                hp_retry_timeout = hm_cfg.get("hp_retry_timeout", 2.0)
                mp_retry_timeout = hm_cfg.get("mp_retry_timeout", 1.5)
                has_recent_read = (t_cur - self.t_last_valid_read) <= valid_timeout

                if not has_recent_read:
                    # No fresh bar read: skip potion actions to avoid stale values.
                    self._limit_fps()
                    continue

                # Rearm only after value recovers above threshold + margin.
                if self.hp_percent >= hp_rearm:
                    self._hp_potion_latched = False
                if self.mp_percent >= mp_rearm:
                    self._mp_potion_latched = False

                # Safety retry: avoid permanent latch when potion does not recover enough.
                if (self._hp_potion_latched and self.hp_percent <= hp_thres
                        and t_cur - self.t_last_heal >= hp_retry_timeout):
                    self._hp_potion_latched = False
                if (self._mp_potion_latched and self.mp_percent <= mp_thres
                        and t_cur - self.t_last_mp >= mp_retry_timeout):
                    self._mp_potion_latched = False

                potion_used_this_tick = False

                # --- Force heal (priority) ---
                if self.enable_hp and hm_cfg.get("force_heal", False):
                    if self.hp_percent < hp_thres:
                        if not self.kb.is_need_force_heal:
                            logger.info(
                                f"[HealthMonitor] Force heal triggered, "
                                f"HP: {self.hp_percent:.1f}%"
                            )
                        self.kb.is_need_force_heal = True
                    else:
                        self.kb.is_need_force_heal = False
                elif self.enable_hp:
                    # --- Normal HP heal ---
                    if (self.hp_percent <= hp_thres
                            and not self._hp_potion_latched
                            and t_cur - self.t_last_heal > hp_cd):
                        if self._heal():
                            logger.info(
                                f"[HealthMonitor] Auto heal, HP: {self.hp_percent:.1f}%"
                            )
                            self.t_last_heal = t_cur
                            self._hp_potion_latched = True
                            potion_used_this_tick = True

                # --- No-potion watchdog → return home ---
                if self.enable_hp and hm_cfg.get("return_home_if_no_potion", False):
                    if self.hp_percent >= hp_thres:
                        self.t_hp_watch_dog = t_cur
                    else:
                        if t_cur - self.t_hp_watch_dog > watchdog_to:
                            logger.warning(
                                f"[HealthMonitor] HP({self.hp_percent:.1f}%) "
                                f"< {hp_thres}% for "
                                f"{round(t_cur - self.t_hp_watch_dog, 1)}s. "
                                f"Returning home (potion used up)."
                            )
                            press_key(self.cfg["key"]["return_home"])
                            self.is_terminated = True
                            self.kb.is_terminated = True

                # --- MP potion ---
                if self.enable_mp and not potion_used_this_tick:
                    if (self.mp_percent <= mp_thres
                            and not self._mp_potion_latched
                            and t_cur - self.t_last_mp > mp_cd):
                        if self._add_mp():
                            logger.info(
                                f"[HealthMonitor] Auto MP, MP: {self.mp_percent:.1f}%"
                            )
                            self.t_last_mp = t_cur
                            self._mp_potion_latched = True

                self._limit_fps()

            except Exception as e:
                logger.error(f"[HealthMonitor] Error: {e}")
                self._limit_fps()

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def _heal(self):
        """Press HP potion key."""
        try:
            if hasattr(self.kb, "press_hp_potion"):
                ok = self.kb.press_hp_potion()
            elif hasattr(self.kb, "queue_hp_potion"):
                ok = self.kb.queue_hp_potion()
            else:
                ok = press_key(self.cfg["key"]["add_hp"], 0.05)
            if not ok:
                logger.warning("[HealthMonitor] Failed to send HP potion key")
            return ok
        except Exception as e:
            logger.error(f"[HealthMonitor] Heal failed: {e}")
            return False

    def _add_mp(self):
        """Press MP potion key."""
        try:
            if hasattr(self.kb, "press_mp_potion"):
                ok = self.kb.press_mp_potion()
            elif hasattr(self.kb, "queue_mp_potion"):
                ok = self.kb.queue_mp_potion()
            else:
                ok = press_key(self.cfg["key"]["add_mp"], 0.05)
            if not ok:
                logger.warning("[HealthMonitor] Failed to send MP potion key")
            return ok
        except Exception as e:
            logger.error(f"[HealthMonitor] MP failed: {e}")
            return False

    # -------------------------------------------------------------------
    # FPS limiting
    # -------------------------------------------------------------------

    def _limit_fps(self):
        target = 1.0 / self.fps_limit
        elapsed = time.time() - self.t_last_run
        if elapsed < target:
            time.sleep(target - elapsed)
        self.fps = round(1.0 / max(time.time() - self.t_last_run, 0.001))
        self.t_last_run = time.time()
