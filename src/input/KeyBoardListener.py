"""
Simple keyboard listener for route recorder.
Tracks currently-held keys and F-key press state via pynput.
"""
import time
import threading

from pynput import keyboard

from src.utils.logger import logger


class KeyBoardListener:
    """Listens for physical keyboard input from the user."""

    def __init__(self, cfg=None, is_autobot=False):
        self.cfg = cfg or {}
        self.window_title = self.cfg.get("game_window", {}).get("title", "MapleStoryGo")

        self.key_pressing: list = []           # currently held movement keys
        self.is_pressed_func_key = [False] * 12  # pollable F1-F12 flags
        self.t_func_key = [0.0] * 12            # debounce timers
        self.debounce_interval = 0.5            # seconds between F-key triggers
        self.is_enable = True
        self.is_terminated = False

        # Movement key mapping (pynput Key -> string)
        self.movement_keys = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.space: "space",
            keyboard.Key.alt: "alt",
            keyboard.Key.alt_l: "alt",
            keyboard.Key.alt_r: "alt",
        }

        # Function key mapping (pynput Key -> index 0-11)
        self.func_keys = {}
        self.func_key_handlers = {}
        for i in range(1, 13):
            self.func_keys[getattr(keyboard.Key, f"f{i}")] = i - 1
            self.func_key_handlers[i - 1] = self._do_nothing

        # For route recorder mode: keep a lightweight daemon thread
        if not is_autobot:
            threading.Thread(target=self._run_idle, daemon=True).start()

        # Start pynput listener
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.start()

        logger.info("[KeyBoardListener] Started")

    def _on_press(self, key):
        """Handle key press."""
        try:
            # Regular character keys
            char = key.char.lower()
            if char and char not in self.key_pressing:
                self.key_pressing.append(char)
            return
        except AttributeError:
            pass

        # Function keys (debounced)
        idx = self.func_keys.get(key)
        if idx is not None:
            now = time.time()
            if now - self.t_func_key[idx] > self.debounce_interval:
                self.is_pressed_func_key[idx] = True
                self.t_func_key[idx] = now
                self.func_key_handlers[idx]()
            return

        # Movement keys
        name = self.movement_keys.get(key)
        if name and name not in self.key_pressing:
            self.key_pressing.append(name)

    def _on_release(self, key):
        """Handle key release."""
        try:
            char = key.char.lower()
            if char in self.key_pressing:
                self.key_pressing.remove(char)
            return
        except AttributeError:
            pass

        name = self.movement_keys.get(key)
        if name and name in self.key_pressing:
            self.key_pressing.remove(name)

    def _run_idle(self):
        """Idle loop for route recorder mode (keeps thread alive)."""
        while not self.is_terminated:
            time.sleep(0.1)

    def _do_nothing(self):
        """Placeholder handler for unregistered F-keys."""
        pass

    def register_func_key_handler(self, key_name, handler):
        """Register a callback for a function key. key_name like 'f1', 'f2', ... 'f12'."""
        idx = int(key_name[1:]) - 1
        if 0 <= idx < 12:
            self.func_key_handlers[idx] = handler

    def stop(self):
        """Stop the listener."""
        self.is_terminated = True
        if self._listener:
            self._listener.stop()
        logger.info("[KeyBoardListener] Stopped")
