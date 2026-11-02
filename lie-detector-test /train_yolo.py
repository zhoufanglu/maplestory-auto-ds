"""
训练 YOLOv8 模型检测测谎小游戏中的目标图形。

Usage:
    python train_yolo.py
"""
import os

from ultralytics import YOLO


def main():
    # 加载预训练的 YOLOv8n 模型
    model = YOLO("yolov8n.pt")

    # 训练参数
    results = model.train(
        data="lie-detector-test/dataset/lie_detector.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="lie_detector_target",
        project="lie-detector-test/runs",
        patience=20,
    )

    print(f"训练完成，最佳模型: {results.best}")


if __name__ == "__main__":
    main()
