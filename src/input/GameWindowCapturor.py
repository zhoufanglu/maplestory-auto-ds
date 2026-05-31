"""
Game window capturer for the "MapleStoryGo" game window on Windows.

Uses mss for screen capture + win32gui for window locating.
Pattern follows the macOS version in MapleStoryAutoLevelUp-main/src/input/GameWindowCapturorForMac.py
"""
import time
import threading

import mss
import cv2
import numpy as np

from src.utils.logger import logger
from src.utils.common import load_image, get_game_window_title_by_token


class GameWindowCapturor:
    """
    Captures frames from the "MapleStoryGo" game window.

    Uses mss (cross-platform screen capture) + win32gui (window location).
    Capture runs in a daemon thread. Pattern matches the macOS reference.
    """

    def __init__(self, cfg: dict, test_image_path: str = ""):
        self.cfg = cfg
        self.frame = None
        self.lock = threading.Lock()
        self.is_terminated = False

        if test_image_path:
            # Offline test mode: load a static image
            logger.info(f"[GameWindowCapturor] Test mode: loading {test_image_path}")
            self.frame = load_image(test_image_path)
            logger.info(f"[GameWindowCapturor] Test image loaded: {self.frame.shape}")
            self.fps = 0
            self.window_title = ""
            return

        # Live capture mode
        import win32gui
        self._win32gui = win32gui

        self.window_title = get_game_window_title_by_token(cfg["game_window"]["title"])
        if not self.window_title:
            logger.error(
                f"[GameWindowCapturor] Unable to find window containing "
                f"'{cfg['game_window']['title']}'"
            )
            self.frame = None
            return

        logger.info(f"[GameWindowCapturor] Found window: '{self.window_title}'")

        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0

        # Use mss to capture the screen region
        self.capture = mss.mss()

        # Get initial window region
        self.update_window_region()

        # Start capture in daemon thread (matches macOS reference exactly)
        threading.Thread(target=self.start_capture, daemon=True).start()

        # Busy-wait for first frame (matches macOS reference exactly)
        time.sleep(0.1)
        while self.frame is None:
            self.limit_fps()

    def start_capture(self):
        """Capture loop — runs in daemon thread."""
        while not self.is_terminated:
            self.update_window_region()
            self.capture_frame()
            self.limit_fps()

    def stop(self):
        """Stop the capture thread."""
        self.is_terminated = True
        logger.info("[GameWindowCapturor] Terminated")

    def update_window_region(self):
        """Update the game window's screen region."""
        try:
            hwnd = self._win32gui.FindWindow(None, self.window_title)
            if hwnd:
                left, top, right, bottom = self._win32gui.GetWindowRect(hwnd)
                self.region = {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top,
                }
            else:
                self.region = None
        except Exception:
            self.region = None

        if self.region is None:
            logger.error(f"Cannot find window: {self.window_title}")
            # Don't raise — keep trying on next iteration

    def capture_frame(self):
        """Capture the current game window region."""
        if self.region is None:
            return

        try:
            img = self.capture.grab(self.region)
            frame = np.array(img)
            with self.lock:
                self.frame = frame
        except Exception as e:
            logger.debug(f"[GameWindowCapturor] grab error: {e}")

    def get_frame(self):
        """
        Thread-safe getter for the latest game frame.

        Returns:
            BGR numpy array, or None if no frame is available.
        """
        with self.lock:
            if self.frame is None:
                return None
            # Only grab reference inside lock — convert outside
            frame_ref = self.frame

        # Convert BGRA (mss format) to BGR outside lock (avoids blocking capture)
        title_bar_height = self.cfg["game_window"].get("title_bar_height", 0)
        if title_bar_height > 0:
            return cv2.cvtColor(
                frame_ref[title_bar_height:, :, :], cv2.COLOR_BGRA2BGR
            )
        return cv2.cvtColor(frame_ref, cv2.COLOR_BGRA2BGR)

    def limit_fps(self):
        """Limit FPS to save system resources."""
        target_duration = 1.0 / self.fps_limit
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        self.fps = round(1.0 / max(time.time() - self.t_last_run, 0.001))
        self.t_last_run = time.time()
