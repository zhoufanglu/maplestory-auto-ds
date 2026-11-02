"""
Lie Detector — 鼠标跟随 Demo

当检测到绿色目标点时，控制鼠标光标跟随移动。
此脚本为概念验证，实际部署需要针对游戏窗口坐标做校准。

Usage:
    python mouse_follower.py --video assets/2026-09-29_16-52-17_lie_detector.mp4
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

try:
    import pyautogui
except ImportError:
    pyautogui = None


class LieDetectorMouseFollower:
    """
    测谎小游戏目标点检测 + 鼠标跟随控制器。

    注意：
      - 仅在 Windows / macOS / Linux 桌面环境下可用。
      - 运行前请确保游戏窗口可见，并且鼠标不会被误控造成危险。
    """

    def __init__(self, move_mouse: bool = False, smooth: float = 0.5):
        self.move_mouse = move_mouse
        self.smooth = max(0.0, min(1.0, smooth))
        self.last_pos = None

    def detect_green_target(self, frame: np.ndarray, roi: tuple = None):
        """
        检测测谎小游戏画面中的绿色圆圈目标点。

        Returns:
            (cx, cy) 或 None
        """
        if roi is not None:
            x, y, w, h = roi
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame))
            y = max(0, min(y, h_frame))
            x2 = min(w_frame, x + w)
            y2 = min(h_frame, y + h)
            frame = frame[y:y2, x:x2]
            roi_offset = (x, y)
        else:
            roi_offset = (0, 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 20:
            return None

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None

        cx = int(moments["m10"] / moments["m00"]) + roi_offset[0]
        cy = int(moments["m01"] / moments["m00"]) + roi_offset[1]
        return cx, cy

    def move_cursor(self, x: int, y: int):
        """
        移动鼠标到指定位置。

        Args:
            x: 屏幕 X 坐标
            y: 屏幕 Y 坐标
        """
        if not self.move_mouse or pyautogui is None:
            return

        # 平滑移动：使用当前位置和目标位置的加权平均
        if self.last_pos is not None:
            lx, ly = self.last_pos
            x = int(lx + (x - lx) * self.smooth)
            y = int(ly + (y - ly) * self.smooth)

        pyautogui.moveTo(x, y, duration=0.01)
        self.last_pos = (x, y)

    def process_video(self, video_path: str, headless: bool = False, roi: tuple = None):
        """处理视频并可选地移动鼠标。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return

        if headless:
            print("Headless mode: processing video without GUI...")
        else:
            print("按 'q' 退出。")
        if self.move_mouse:
            print("⚠️  已启用鼠标移动。请确保目标窗口正确，按 'q' 立即停止。")

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            target = self.detect_green_target(frame, roi=roi)
            debug = frame.copy()

            if target:
                cx, cy = target
                cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    debug,
                    f"Target: ({cx}, {cy})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                if self.move_mouse:
                    self.move_cursor(cx, cy)
            else:
                cv2.putText(
                    debug,
                    "No target",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            if headless:
                # 每 10 帧保存一帧调试图像
                if frame_index % 10 == 0:
                    os.makedirs("output", exist_ok=True)
                    out_path = os.path.join("output", f"mouse_follower_{frame_index:04d}.jpg")
                    cv2.imwrite(out_path, debug)
                    print(f"Saved {out_path}")
            else:
                cv2.imshow("Lie Detector Mouse Follower", debug)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break

            frame_index += 1

        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_index}")


def main():
    parser = argparse.ArgumentParser(description="Lie detector mouse follower demo")
    parser.add_argument("--video", type=str, required=True, help="Path to the lie detector video")
    parser.add_argument(
        "--move-mouse",
        action="store_true",
        help="Actually move the mouse cursor (default: just visualize)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="Smoothing factor for mouse movement (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI, save output images",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=[330, 110, 640, 520],
        help="Detection ROI: x y w h (default: 330 110 640 520)",
    )
    args = parser.parse_args()

    follower = LieDetectorMouseFollower(move_mouse=args.move_mouse, smooth=args.smooth)
    follower.process_video(args.video, headless=args.headless, roi=tuple(args.roi))


if __name__ == "__main__":
    main()
