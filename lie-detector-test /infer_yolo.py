"""
使用训练好的 YOLOv8 模型检测测谎小游戏目标图形。

Usage:
    python infer_yolo.py --image dataset/images/val/frame_0015.jpg
    python infer_yolo.py --video assets/2026-09-29_16-52-17_lie_detector.mp4
"""
import argparse
import os

import cv2

from ultralytics import YOLO


def draw_boxes(img, results):
    """在图像上绘制检测结果。"""
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"target {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return img


def process_image(model, image_path, output_dir="output"):
    """对单张图片推理。"""
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    results = model(img, verbose=False)
    img = draw_boxes(img, results)

    out_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_yolo.jpg"))
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


def process_video(model, video_path, output_dir="output"):
    """对视频推理。"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = os.path.join(output_dir, "lie_detector_yolo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        frame = draw_boxes(frame, results)
        writer.write(frame)

        frame_index += 1
        if frame_index % 10 == 0:
            print(f"Processed {frame_index} frames...")

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 inference for lie detector target")
    parser.add_argument("--model", type=str, default="lie-detector-test/runs/detect/lie_detector_target/weights/best.pt",
                        help="Path to trained YOLOv8 model")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--video", type=str, help="Path to a video file")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"模型不存在: {args.model}")
        print("请先运行 train_yolo.py 训练模型")
        return

    model = YOLO(args.model)

    if args.image:
        process_image(model, args.image)
    elif args.video:
        process_video(model, args.video)
    else:
        print("请提供 --image 或 --video 参数")


if __name__ == "__main__":
    main()
