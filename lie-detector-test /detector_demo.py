"""
Lie Detector — 目标形状检测 Demo

检测《冒险岛》测谎小游戏中的白色/浅色目标图形。
绿色圆点是玩家鼠标位置，真正的目标是把鼠标移到这个白色图形上。

Usage:
    python detector_demo.py --image frames/frame_10s.jpg
    python detector_demo.py --video assets/2026-09-29_16-52-17_lie_detector.mp4
"""
import argparse
import math
import os

import cv2
import numpy as np


def ensure_output_dir():
    """创建输出目录。"""
    os.makedirs("output", exist_ok=True)
    return "output"


def detect_mouse_cursor(hsv: np.ndarray):
    """
    检测绿色鼠标光标位置。
    """
    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 10:
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy), area


def detect_target_shape(frame: np.ndarray, debug: bool = False, roi: tuple = None):
    """
    检测测谎小游戏画面中的白色/浅色目标图形。

    策略：
      1. 在 ROI 内检测绿色圆点（鼠标位置）
      2. 以绿色圆点为中心，提取周围的高亮连通区域（目标图形）
      3. 通过亮度阈值 + 连通域分析，找到目标形状
      4. 返回目标图形的轮廓、中心、面积

    Args:
        frame: BGR 输入图像。
        debug: 是否返回调试图像。
        roi: (x, y, w, h) 限制检测区域，None 表示全图。

    Returns:
        (center_x, center_y, target_info, debug_image)
    """
    # 裁剪 ROI
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ------------------- 步骤 1：检测绿色圆点（鼠标） -------------------
    mouse_result = detect_mouse_cursor(hsv)
    if mouse_result is None:
        if not debug:
            return None, None, None, None
        debug_img = frame.copy()
        cv2.putText(
            debug_img,
            "No mouse cursor",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return None, None, None, debug_img

    (mx, my), mouse_area = mouse_result

    # ------------------- 步骤 2：以鼠标为中心提取目标图形 -------------------
    # 目标图形比周围岩石更亮。我们用自适应阈值：
    # 以鼠标点为中心，取周围区域，用 Otsu 或高阈值分割出目标图形

    # 方法：从鼠标点开始向外扩展，找到亮度高于背景的连通区域
    # 先用一个相对低的阈值提取所有可能的亮区域
    _, mask_bright = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

    # 形态学连接
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_bright = cv2.morphologyEx(mask_bright, cv2.MORPH_CLOSE, kernel)

    # 找到包含鼠标点的亮区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bright, connectivity=8)

    target_info = None
    for i in range(1, num_labels):
        # 检查鼠标点是否在该连通区域内
        if labels[my, mx] == i:
            x, y_b, w_b, h_b, area = stats[i]
            cx, cy = centroids[i]
            target_info = {
                "contour": None,  # connectedComponents 不直接返回轮廓
                "bbox": (x, y_b, w_b, h_b),
                "area": area,
                "center": (int(cx) + roi_offset[0], int(cy) + roi_offset[1]),
                "mouse": (mx + roi_offset[0], my + roi_offset[1]),
            }
            break

    if not debug:
        if target_info is None:
            return None, None, None, None
        return target_info["center"][0], target_info["center"][1], target_info, None

    # 绘制调试图像
    debug_img = frame.copy()
    if target_info is not None:
        x, y_b, w_b, h_b = target_info["bbox"]
        cx, cy = target_info["center"]
        mx, my = target_info["mouse"]

        # 绘制边界框
        cv2.rectangle(debug_img, (x, y_b), (x + w_b, y_b + h_b), (0, 255, 0), 2)
        # 绘制中心和鼠标点
        cv2.circle(debug_img, (mx - roi_offset[0], my - roi_offset[1]), 5, (0, 0, 255), -1)
        cv2.circle(debug_img, (cx - roi_offset[0], cy - roi_offset[1]), 5, (255, 0, 0), -1)

        cv2.putText(
            debug_img,
            f"Target center: ({cx}, {cy}) area={target_info['area']:.0f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            debug_img,
            "No target detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if target_info is None:
        return None, None, None, debug_img
    return target_info["center"][0], target_info["center"][1], target_info, debug_img


def process_image(image_path: str, headless: bool = False):
    """处理单张图片。"""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return

    x, y, info, debug = detect_target_shape(frame, debug=True, roi=(350, 230, 600, 400))
    print(f"Detected target center: ({x}, {y})")

    if headless:
        output_dir = ensure_output_dir()
        out_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_shape_result.jpg"))
        cv2.imwrite(out_path, debug)
        print(f"Saved result to: {out_path}")
    else:
        cv2.imshow("Lie Detector Target Shape", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(video_path: str, headless: bool = False):
    """处理视频流。"""
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    if headless:
        print("Headless mode: processing video without GUI...")
    else:
        print("按 'q' 退出，空格键暂停。")
    paused = False

    frame_index = 0
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            x, y, info, debug = detect_target_shape(frame, debug=True, roi=(350, 230, 600, 400))
            if x is not None:
                print(f"Frame {frame_index}: Target center: ({x}, {y})")

            if headless:
                output_dir = ensure_output_dir()
                out_path = os.path.join(output_dir, f"shape_{frame_index:04d}.jpg")
                cv2.imwrite(out_path, debug)
            else:
                cv2.imshow("Lie Detector Target Shape", debug)

            frame_index += 1

        if not headless:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
        else:
            if frame_index % 10 == 0:
                print(f"Processed {frame_index} frames...")

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"Total frames processed: {frame_index}")


def main():
    parser = argparse.ArgumentParser(description="Lie detector target shape detection demo")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--video", type=str, help="Path to a video file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI, save output images")
    args = parser.parse_args()

    if args.image:
        process_image(args.image, headless=args.headless)
    elif args.video:
        process_video(args.video, headless=args.headless)
    else:
        default_frame = "frames/frame_10s.jpg"
        print(f"No input provided, using default: {default_frame}")
        process_image(default_frame, headless=True)


if __name__ == "__main__":
    main()
