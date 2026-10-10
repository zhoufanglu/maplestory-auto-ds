"""
测试声音播放 — 直接播放 warning.mp3，验证报警音是否正常。

用法:
    python warning/test_waring.py
"""
import os
import time

import pygame

# 用绝对路径，避免工作目录不同导致找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_FILE = os.path.join(BASE_DIR, 'warning.mp3')


def main():
    if not os.path.exists(ALARM_FILE):
        print(f"❌ 找不到文件: {ALARM_FILE}")
        return

    print(f"🎵 音频文件: {ALARM_FILE}")

    # 初始化 pygame 混音器
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"❌ 混音器初始化失败: {e}")
        return

    try:
        print("▶ 开始播放...")
        pygame.mixer.music.load(ALARM_FILE)
        pygame.mixer.music.play()

        # 等待播放结束（最长等 10 秒，防止卡死）
        wait_seconds = 10
        t_start = time.time()
        while pygame.mixer.music.get_busy() and time.time() - t_start < wait_seconds:
            time.sleep(0.1)

        if pygame.mixer.music.get_busy():
            print("⚠️ 播放仍在进行（可能音频较长），手动停止")
            pygame.mixer.music.stop()

        print("✅ 播放完成，声音正常！")
    except Exception as e:
        print(f"❌ 播放失败: {e}")
    finally:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
