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

import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger
from src.utils.common import load_yaml, override_cfg, is_mac, is_windows
from src.engine.MonsterDetector import MonsterDetector


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

    logger.info("Starting game window capture...")
    detector.start_capture()

    # -------------------------------------------------------------------
    # Start processing in background thread (matches reference: thread_auto_bot)
    # -------------------------------------------------------------------
    detector.start()

    # -------------------------------------------------------------------
    # Main thread: display loop (matches reference: main() while loop)
    # Processing runs in background, display never blocks.
    # -------------------------------------------------------------------
    win_name = "Monster Detection — MapleStoryGo"
    window_positioned = False
    scale = cfg["monster_detect"].get("display_scale", 0.55)

    try:
        while not detector.is_terminated:
            # Show debug image — only when a new frame is ready
            # (exact same pattern as reference: MapleStoryAutoLevelUp.py:1851-1864)
            if detector.is_frame_done:
                if detector.img_display is not None:
                    cv2.imshow(win_name, detector.img_display)

                    if not window_positioned:
                        try:
                            import ctypes
                            user32 = ctypes.windll.user32
                            screen_w = user32.GetSystemMetrics(0)
                            screen_h = user32.GetSystemMetrics(1)
                            rect = cv2.getWindowImageRect(win_name)
                            if rect:
                                _, _, win_w, win_h = rect
                                cv2.moveWindow(win_name,
                                               screen_w - win_w - 20,
                                               screen_h - win_h - 80)
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

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()

    logger.info("Done.")


if __name__ == "__main__":
    main()
