import os
import time

import cv2
import numpy as np
import pygame
import winsound
import mss

# ==========================================
# 🛠️ 告警配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WARNING_IMAGE = os.path.join(BASE_DIR, 'warning.png')  # 告警图片
CONFIDENCE_LEVEL = 0.5  # 匹配置信度 0~1（越大越严格）
ALARM_FILE = os.path.join(BASE_DIR, 'warning.mp3')  # 告警音频

# 初始化 Pygame 混音器
try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
except Exception as e:
    print(f"⚠️ 音频初始化失败: {e}")


def play_alarm():
    """
    内部函数：负责播放声音，包含兜底逻辑
    """
    try:
        if not pygame.mixer.music.get_busy():
            if os.path.exists(ALARM_FILE):
                pygame.mixer.music.load(ALARM_FILE)
                pygame.mixer.music.play()
            else:
                winsound.MessageBeep(winsound.MB_ICONHAND)
    except Exception as e:
        winsound.MessageBeep(winsound.MB_ICONHAND)
        print(f"🔊 播放异常: {e}")


def check_for_warning(region=None):
    """
    供主程序调用：截屏并用模板匹配检测测谎告警弹窗。

    用 mss 截屏 + cv2.matchTemplate（比 pyautogui 更快更可控），
    匹配成功返回 True 并报警；失败返回 False。

    :param region: 限制检测区域 (x, y, w, h)，不传则检测主显示器全屏
    """
    try:
        needle = cv2.imread(WARNING_IMAGE, cv2.IMREAD_COLOR)
        if needle is None:
            print("[WARNING] warning.png 加载失败，请检查文件是否存在")
            return False

        # 截屏
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # 主显示器
            shot = np.array(sct.grab(monitor))
        screen = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

        if region is not None:
            x, y, w, h = region
            screen = screen[y:y + h, x:x + w]

        # 模板匹配（归一化相关系数，0~1，越大越像）
        result = cv2.matchTemplate(screen, needle, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= CONFIDENCE_LEVEL:
            print(f"\n[🚨 ALERT] 发现告警图！score={max_val:.2f} 位置={max_loc}")
            play_alarm()
            return True

        # 接近阈值时打印分数，方便调 CONFIDENCE_LEVEL
        if max_val >= CONFIDENCE_LEVEL - 0.15:
            print(f"[WARNING] 接近匹配 score={max_val:.2f}（阈值 {CONFIDENCE_LEVEL}）")
        return False
    except Exception as e:
        print(f"[WARNING] 检测异常: {e}")
        return False


# ==========================================
# 🧪 独立测试逻辑
# ==========================================
if __name__ == "__main__":
    print("🔔 正在进行 warning.py 独立功能测试...")
    print("请在屏幕上打开告警图，或者手动触发声音测试。")

    play_alarm()

    while True:
        if check_for_warning():
            print("检测到目标！")
        time.sleep(1)
