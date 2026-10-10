"""
MapleStoryGo — Monster Detection System

Entry point for the monster detection engine.
Processing runs in background thread, main thread handles display (no stutter).

Usage:
    python src/main.py                          # Live detection on MapleStoryGo
    python src/main.py --test_image test/xxx.png # Offline test on a screenshot
    python src/main.py --debug                   # Enable debug logging
    python src/main.py --disable_viz             # Disable visual debug window

Reference:
  MapleStoryAutoLevelUp-main/src/engine/MapleStoryAutoLevelUp.py
"""
import argparse
import logging
import os
import sys
import time
import threading

import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add warning directory to path for alarm detection
_warning_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "warning")
sys.path.insert(0, _warning_dir)

from src.utils.logger import logger
from src.utils.common import load_yaml, override_cfg, is_mac
from src.engine.MonsterDetector import MonsterDetector

try:
    import warning as warning_module
except ImportError:
    warning_module = None


def main():
    parser = argparse.ArgumentParser(
        description="MapleStoryGo — Monster Detection System"
    )
    parser.add_argument("--test_image", type=str, default="",
                        help="Path to a test screenshot for offline debugging")
    parser.add_argument("--cfg", type=str, default="custom",
                        help="Custom config suffix")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--disable_viz", action="store_true",
                        help="Disable the debug visualization window")
    args = parser.parse_args()

    if args.debug:
        logger.set_level(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("MapleStoryGo — Monster Detection System")
    logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------
    logger.info("Loading configuration...")
    cfg_path = "config/config_default.yaml"
    if not os.path.exists(cfg_path):
        logger.error(f"Default config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_yaml(cfg_path)
    logger.info(f"  Loaded: {cfg_path}")

    if is_mac():
        platform_cfg_path = "config/config_macOS.yaml"
        if os.path.exists(platform_cfg_path):
            cfg = override_cfg(cfg, load_yaml(platform_cfg_path))
            logger.info(f"  Applied: {platform_cfg_path}")

    custom_cfg_path = f"config/config_{args.cfg}.yaml"
    if os.path.exists(custom_cfg_path):
        cfg = override_cfg(cfg, load_yaml(custom_cfg_path))
        logger.info(f"  Applied: {custom_cfg_path}")
    else:
        logger.info(f"  No custom config found at {custom_cfg_path}")

    if args.disable_viz:
        cfg["disable_viz"] = True
        logger.info("  Debug visualization: disabled")

    # -------------------------------------------------------------------
    # Initialize the monster detector
    # -------------------------------------------------------------------
    logger.info("Initializing MonsterDetector...")
    detector = MonsterDetector(cfg, test_image=args.test_image)

    logger.info("Loading monster templates...")
    if not detector.load_monster_images():
        logger.error("Failed to load monster images. Exiting.")
        sys.exit(1)

    # Load map and routes if enabled
    if cfg["bot"].get("enable_route") and cfg["bot"].get("map"):
        logger.info(f"Loading map/route: {cfg['bot']['map']}...")
        detector.load_map_and_routes(cfg["bot"]["map"])

    # Create keyboard controller for movement + listener for F1 toggle
    from src.input.KeyBoardController import KeyBoardController
    from src.input.KeyBoardListener import KeyBoardListener
    kb = KeyBoardController(cfg)
    detector._kb = kb  # Wire to detector
    kb_listener = KeyBoardListener(cfg, is_autobot=True)
    kb_listener.register_func_key_handler('f1', kb.toggle)

    logger.info("Starting game window capture...")
    detector.start_capture()

    # -------------------------------------------------------------------
    # Start processing in background thread (matches reference: thread_auto_bot)
    # -------------------------------------------------------------------
    detector.start()

    # Register HP/MP toggle hotkeys (after HealthMonitor is created)
    if detector.health_monitor is not None:
        kb_listener.register_func_key_handler(
            'f2', detector.health_monitor.toggle_hp
        )
        kb_listener.register_func_key_handler(
            'f3', detector.health_monitor.toggle_mp
        )

    # Register F11 to toggle bot mode (detect <-> hunt)
    kb_listener.register_func_key_handler('f11', detector.toggle_mode)

    # Register F12 to freeze/unfreeze the title/player box
    kb_listener.register_func_key_handler('f12', detector.toggle_title_fixed)

    # -------------------------------------------------------------------
    # Warning (测谎/报警) detection — background thread, non-blocking
    # -------------------------------------------------------------------
    warning_cfg = cfg.get("warning_monitor", {})
    warning_enabled = bool(warning_cfg.get("enable", True))
    warning_interval = float(warning_cfg.get("interval_seconds", 2.0))
    if warning_interval <= 0:
        warning_interval = 2.0

    warning_thread = None
    if warning_module is None:
        logger.info("[Warning] 警报模块未加载，跳过监控")
    elif not warning_enabled:
        logger.info("[Warning] 警报监控已在配置中关闭")
    else:
        def _warning_monitor():
            logger.info("[Warning] 测谎报警监控已启动")
            while not detector.is_terminated:
                try:
                    if warning_module.check_for_warning():
                        logger.warning("[Warning] 检测到测谎报警弹窗！请及时处理！")
                except Exception as e:
                    logger.debug(f"[Warning] 检测异常: {e}")
                time.sleep(warning_interval)

        warning_thread = threading.Thread(target=_warning_monitor, daemon=True)
        warning_thread.start()

    # -------------------------------------------------------------------
    # Main thread: display loop (matches reference: main() while loop)
    # Processing runs in background, display never blocks.
    # -------------------------------------------------------------------
    win_name = "NewMaple"
    window_positioned = False
    last_frame_seq = -1
    t_last_display = time.time()

    cv2.namedWindow(win_name)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            detector.request_title_selection_from_display(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            detector.clear_title_selection()

    cv2.setMouseCallback(win_name, _on_mouse)

    try:
        while not detector.is_terminated:
            # Show debug image — only when a new frame is ready
            # (exact same pattern as reference: MapleStoryAutoLevelUp.py:1851-1864)
            if detector.is_frame_done and detector.frame_seq != last_frame_seq:
                if detector.img_display is not None:
                    cv2.imshow(win_name, detector.img_display)
                    now = time.time()
                    detector.display_fps = round(1.0 / max(now - t_last_display, 0.001))
                    t_last_display = now
                    last_frame_seq = detector.frame_seq

                    if not window_positioned:
                        try:
                            import ctypes
                            user32 = ctypes.windll.user32
                            get_system_metrics = getattr(user32, "GetSystemMetrics")
                            screen_w = get_system_metrics(0)
                            screen_h = get_system_metrics(1)
                            rect = cv2.getWindowImageRect(win_name)
                            if rect:
                                _, _, win_w, win_h = rect
                                cv2.moveWindow(win_name,
                                               screen_w - win_w - 20,
                                               screen_h - win_h - 100)
                                window_positioned = True
                        except Exception:
                            pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("Quit key pressed.")
                detector.is_terminated = True
            elif key == ord('s'):
                from src.utils.common import screenshot
                screenshot(detector.img_display, "detection")
                logger.info("Screenshot saved.")
            elif key == ord('c'):
                detector.clear_title_selection()

            # No extra fixed sleep: let waitKey pump events smoothly.

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()
        kb.stop()
        kb_listener.stop()

    logger.info("Done.")


if __name__ == "__main__":
    main()
