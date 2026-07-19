"""Скачивание YOLO модели"""
from ultralytics import YOLO

print("Скачивание YOLOv8n...")
model = YOLO('yolov8n.pt')
print(f"✅ Модель скачана: {model.ckpt_path}")
