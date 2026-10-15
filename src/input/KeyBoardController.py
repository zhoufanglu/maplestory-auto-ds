"""
Keyboard controller for sending movement commands.
Uses pyautogui for movement keys, pynput for jump/teleport (better game compat).
Runs in daemon thread, consumes commands from set_command().
"""
import time
import threading

import pyautogui
from pynput.keyboard import Key, Controller as KBController

from src.utils.logger import logger

pyautogui.FAILSAFE = False
_pynput_kb = KBController()


def _resolve_pynput_key(key):
    """Resolve string key names like 'home'/'end' to pynput Key constants."""
    if isinstance(key, str):
        normalized = key.strip().lower()
        if hasattr(Key, normalized):
            return getattr(Key, normalized)
    return key


def key_down(key):
    """Press and hold a key. Try pynput first, then pyautogui."""
    try:
        _pynput_kb.press(_resolve_pynput_key(key))
        return True
    except Exception:
        pass
    try:
        pyautogui.keyDown(key)
        return True
    except Exception:
        return False


def key_up(key):
    """Release a held key. Try pynput first, then pyautogui."""
    try:
        _pynput_kb.release(_resolve_pynput_key(key))
        return True
    except Exception:
        pass
    try:
        pyautogui.keyUp(key)
        return True
    except Exception:
        return False


def press_key(key: str, duration=0.2) -> bool:
    """Press and release a key. Returns True on success."""
    # Try pynput first (works better for many in-game keys).
    try:
        resolved_key = _resolve_pynput_key(key)
        _pynput_kb.press(resolved_key)
        time.sleep(duration)
        _pynput_kb.release(resolved_key)
        return True
    except Exception:
        pass

    # Fallback to pyautogui for keys that are not accepted by pynput/game.
    try:
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)
        return True
    except Exception:
        return False


class KeyBoardController:
    """
    Sends movement commands to the game via pyautogui.
    Runs in a daemon thread, polling for new commands.

    Usage:
        kb = KeyBoardController(cfg)
        kb.set_command("left none none")   # walk left
        kb.set_command("none none jump")   # jump
        kb.set_command("none none none")   # stop
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.is_enable = True
        self.is_terminated = False

        # Key bindings
        key_cfg = cfg.get("key", {})
        self.key_jump = key_cfg.get("jump", "space")
        self.key_teleport = key_cfg.get("teleport", "e")
        self.key_directional_attack = key_cfg.get("directional_attack", "w")
        self.key_aoe_skill = key_cfg.get("aoe_skill", "q")
        self.key_pickup = key_cfg.get("pickup", "z")
        self.jump_hold_up_seconds = float(key_cfg.get("jump_hold_up_seconds", 2.0))
        self.attack_mode = cfg.get("bot", {}).get("attack", "directional")
        self._t_last_attack = 0.0
        self._t_last_pickup = 0.0
        self._attack_pause_until = 0.0
        self._face_direction_until = 0.0
        self._hold_up_until = 0.0
        self._t_last_face_tap = 0.0
        self._last_face_dir = None
        self._attack_count = 0            # 连续攻击计数
        self._force_walk_until = 0.0      # 连击后强制移动（朝向怪物）的截止时间
        self._action_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending_hp_potion = False
        self._pending_mp_potion = False

        # Force heal flag (set by HealthMonitor)
        self.is_need_force_heal = False

        # Current command state
        self.cmd_left_right = "none"  # "left", "right", "stop", "none"
        self.cmd_up_down = "none"     # "up", "down", "stop", "none"
        self.cmd_action = "none"      # "jump", "teleport", "goal", "attack", "none", "add_hp", "add_mp"

        # Tracking to avoid redundant key_down/key_up
        self._last_left_right = "none"
        self._last_up_down = "none"
        self._last_action = "none"

        # FPS limit
        self._fps_limit = cfg["system"].get("fps_limit_main", 15)
        self._t_last = time.time()

        # Start daemon thread
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[KeyBoardController] Started")

    def set_command(self, cmd_str):
        """Set movement command from string like 'left none jump'."""
        parts = cmd_str.strip().split()
        if len(parts) >= 3:
            self.cmd_left_right = parts[0]
            self.cmd_up_down = parts[1]
            self.cmd_action = parts[2]

    def _tap_direction(self, key: str, duration=0.04) -> bool:
        """Tap a direction briefly to turn without visibly walking."""
        with self._send_lock:
            if not key_down(key):
                return False
            time.sleep(duration)
            key_up(key)
        return True

    def disable(self):
        self.is_enable = False
        self.release_all()

    def enable(self):
        self.is_enable = True

    def toggle(self):
        if self.is_enable:
            self.disable()
        else:
            self.enable()
        logger.info(f"[KB] {'ENABLED' if self.is_enable else 'DISABLED'}")

    def release_all(self):
        """Release all held movement keys."""
        for key in ["left", "right"]:
            key_up(key)
        for key in ["up", "down"]:
            try:
                _pynput_kb.release(getattr(Key, key, key))
            except Exception:
                pass
        self._last_left_right = "none"
        self._last_up_down = "none"
        self._last_action = "none"
        self._hold_up_until = 0.0

    def stop(self):
        """Stop the controller and release all keys."""
        self.is_terminated = True
        self.release_all()
        logger.info("[KeyBoardController] Stopped")

    def queue_hp_potion(self):
        """Queue one HP potion action (consumed by keyboard thread)."""
        with self._action_lock:
            self._pending_hp_potion = True
        return True

    def queue_mp_potion(self):
        """Queue one MP potion action (consumed by keyboard thread)."""
        with self._action_lock:
            self._pending_mp_potion = True
        return True

    def _press_key_serialized(self, key: str, duration=0.05) -> bool:
        """Serialize one-shot key sends across keyboard and health-monitor threads."""
        with self._send_lock:
            return press_key(key, duration)

    def press_hp_potion(self) -> bool:
        """Immediately send the HP potion key and pause attack briefly."""
        ok = self._press_key_serialized(self.cfg["key"].get("add_hp", "1"), 0.05)
        if ok:
            self._attack_pause_until = max(self._attack_pause_until, time.time() + 0.12)
            self._last_action = "add_hp"
        return ok

    def press_mp_potion(self) -> bool:
        """Immediately send the MP potion key and pause attack briefly."""
        ok = self._press_key_serialized(self.cfg["key"].get("add_mp", "2"), 0.05)
        if ok:
            self._attack_pause_until = max(self._attack_pause_until, time.time() + 0.12)
            self._last_action = "add_mp"
        return ok

    def _run(self):
        """Daemon loop — applies left/right, up/down, action, and pickup commands."""
        while not self.is_terminated:
            if self.is_enable:
                self._apply_movement()
                self._apply_action()
                self._apply_pickup()
            self._limit_fps()

    def _apply_pickup(self):
        """Tap the pickup key periodically while the character is moving."""
        pickup_cfg = self.cfg.get("pickup", {})
        if not pickup_cfg.get("enable", True):
            return
        moving = (
            self.cmd_left_right in ("left", "right")
            or self.cmd_up_down in ("up", "down")
        )
        if not moving:
            return
        pickup_cd = float(pickup_cfg.get("cooldown", 0.3))
        t_now = time.time()
        if t_now - self._t_last_pickup >= pickup_cd:
            self._press_key_serialized(self.key_pickup, 0.03)
            self._t_last_pickup = t_now

    def _apply_movement(self):
        """Apply left/right and up/down key state. Up/down uses pynput for game compatibility."""
        # Left/Right: support both walking (hold) and face-only taps.
        lr = self.cmd_left_right
        # 连击后强制移动：把「转身」临时变成「走」，朝怪物方向走一小段重新面向。
        if time.time() < self._force_walk_until and lr in ("face_left", "face_right"):
            lr = "left" if lr == "face_left" else "right"
        if lr in ("face_left", "face_right"):
            # Face commands: tap only when the facing direction changes,
            # so repeated face commands don't cause the character to creep.
            if self._last_left_right in ("left", "right"):
                key_up(self._last_left_right)

            if lr != self._last_face_dir:
                tap_duration = float(self.cfg.get("directional_attack", {}).get("face_tap_duration", 0.07))
                key = "left" if lr == "face_left" else "right"
                if self._tap_direction(key, duration=max(0.03, tap_duration)):
                    self._last_face_dir = lr
                    # 转身后多停一会儿再攻击，确保转身（可能被攻击动画吞掉）先完成。
                    self._face_direction_until = time.time() + tap_duration + 0.3
                    if time.time() - getattr(self, '_last_face_log', 0) >= 2.0:
                        logger.info(f"[KB] 转身 {lr} → 按 {key} 键 ({tap_duration:.2f}s)")
                        self._last_face_log = time.time()
            self._last_left_right = lr
        elif lr != self._last_left_right:
            if self._last_left_right in ("left", "right"):
                key_up(self._last_left_right)
            if lr in ("left", "right"):
                key_down(lr)
                self._face_direction_until = time.time() + 0.09
                # Walking turns the character: keep the face-direction cache
                # in sync so the next face_* command actually taps to turn.
                self._last_face_dir = "face_left" if lr == "left" else "face_right"
            elif lr == "stop":
                key_up("left")
                key_up("right")
            self._last_left_right = lr

        # Up/Down (pynput — same as jump, better game compatibility)
        ud = self.cmd_up_down
        # 抓绳：跳跃后短时间内强制按住 up（除非正往下）。
        if time.time() < self._hold_up_until and ud != "down":
            ud = "up"
        if ud != self._last_up_down:
            if self._last_up_down in ("up", "down"):
                _pynput_kb.release(getattr(Key, self._last_up_down, self._last_up_down))
            if ud in ("up", "down"):
                _pynput_kb.press(getattr(Key, ud, ud))
            elif ud == "stop":
                _pynput_kb.release(Key.up)
                _pynput_kb.release(Key.down)
            self._last_up_down = ud

    def _apply_action(self):
        """Apply one-shot actions only once per command change."""
        # Force heal overrides normal action
        if self.is_need_force_heal:
            self.press_hp_potion()
            return

        # Potions are high priority and are serialized in this same thread.
        pending_hp = False
        pending_mp = False
        with self._action_lock:
            if self._pending_hp_potion:
                pending_hp = True
                self._pending_hp_potion = False
            elif self._pending_mp_potion:
                pending_mp = True
                self._pending_mp_potion = False

        if pending_hp:
            self.press_hp_potion()
            return
        if pending_mp:
            self.press_mp_potion()
            return

        # Attack is level-triggered: keep firing on cooldown while command stays "attack"/"aoe".
        if self.cmd_action in ("attack", "aoe"):
            if time.time() < self._attack_pause_until:
                return
            if time.time() < self._face_direction_until:
                return
            if self.cmd_action == "aoe":
                # Explicit AoE action (monster count >= threshold)
                atk_key = self.key_aoe_skill
                atk_cd = self.cfg.get("aoe_skill", {}).get("cooldown", 1.0)
            elif self.attack_mode == "aoe_skill":
                atk_key = self.key_aoe_skill
                atk_cd = self.cfg.get("aoe_skill", {}).get("cooldown", 1.0)
            else:
                atk_key = self.key_directional_attack
                atk_cd = self.cfg.get("directional_attack", {}).get("cooldown", 0.9)

            t_now = time.time()
            if t_now - self._t_last_attack >= float(atk_cd):
                self._press_key_serialized(atk_key, 0.05)
                self._t_last_attack = t_now
                # 连续攻击达到阈值后，短暂朝怪物方向移动一下（重新面向/贴近）。
                self._attack_count += 1
                interval = int(self.cfg.get("directional_attack", {}).get("attack_move_interval", 3))
                if self._attack_count >= interval:
                    move_seconds = float(self.cfg.get("directional_attack", {}).get("attack_move_seconds", 0.3))
                    self._force_walk_until = time.time() + move_seconds
                    self._attack_count = 0
            return

        if self.cmd_action != self._last_action:
            self._last_action = self.cmd_action
            if self.cmd_action == "jump":
                time.sleep(0.05)
                self._press_key_serialized(self.key_jump)
                # 抓绳优化：按跳的同时按住 up 一段时间（向下跳除外）。
                if self.cmd_up_down != "down" and self.jump_hold_up_seconds > 0:
                    self._hold_up_until = time.time() + self.jump_hold_up_seconds
            elif self.cmd_action == "teleport":
                time.sleep(0.05)
                self._press_key_serialized(self.key_teleport)
            elif self.cmd_action == "attack_once":
                # One-shot single attack (ignores cooldown), then reset.
                self._press_key_serialized(self.key_directional_attack, 0.05)
                self.cmd_action = "none"
            elif self.cmd_action == "add_hp":
                self.press_hp_potion()
                self.cmd_action = "none"
            elif self.cmd_action == "add_mp":
                self.press_mp_potion()
                self.cmd_action = "none"

    def _limit_fps(self):
        """Cap the loop rate."""
        elapsed = time.time() - self._t_last
        target = 1.0 / self._fps_limit
        if elapsed < target:
            time.sleep(target - elapsed)
        self._t_last = time.time()
