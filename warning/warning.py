import pyautogui
import pygame
import winsound
import os
import time

# ==========================================
# 🛠️ 告警配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WARNING_IMAGE = os.path.join(BASE_DIR, 'warning.png')  # 告警图片
CONFIDENCE_LEVEL = 0.5  # 匹配精度
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
        # 如果当前没有声音在播放，才开始播放新声音（防止鬼畜）
        if not pygame.mixer.music.get_busy():
            if os.path.exists(ALARM_FILE):
                pygame.mixer.music.load(ALARM_FILE)
                pygame.mixer.music.play()
            else:
                # 找不到文件时的兜底：系统警告音
                winsound.MessageBeep(winsound.MB_ICONHAND)
    except Exception as e:
        # 播放过程出错时的兜底
        winsound.MessageBeep(winsound.MB_ICONHAND)
        print(f"🔊 播放异常: {e}")


def check_for_warning(region=None):
    """
    供主程序 hunter.py 调用：检查屏幕并报警
    :param region: 限制检测区域 (x, y, w, h)，不传则全屏
    """
    try:
        # 查找图片
        location = pyautogui.locateOnScreen(WARNING_IMAGE, confidence=CONFIDENCE_LEVEL, region=region)

        if location:
            print(f"\n[🚨 ALERT] 发现告警图！位置: {location}")
            play_alarm()
            return True
        return False
    except Exception:
        # 可能是因为没装 opencv-python 导致的信心值搜索失败
        return False


# ==========================================
# 🧪 独立测试逻辑
# ==========================================
if __name__ == "__main__":
    print("🔔 正在进行 warning.py 独立功能测试...")
    print("请在屏幕上打开告警图，或者手动触发声音测试。")

    # 强制响一声确认模块没问题
    play_alarm()

    # 进入一个简单的测试循环
    while True:
        if check_for_warning():
            print("检测到目标！")
        time.sleep(1)