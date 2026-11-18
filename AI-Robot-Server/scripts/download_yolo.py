"""Скачивание YOLO модели в models/yolov8n.pt"""
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
target = MODELS_DIR / "yolov8n.pt"

if target.exists():
    print(f"✓ {target.name} уже существует")
else:
    print("Скачивание YOLOv8n...")
    model = YOLO("yolov8n.pt")
    downloaded = Path(model.ckpt_path)
    if downloaded.resolve() != target.resolve():
        shutil.move(str(downloaded), str(target))
    print(f"✅ Модель скачана: {target}")
