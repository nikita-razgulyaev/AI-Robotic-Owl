"""Проверка GGUF модели"""
import os
import sys
from pathlib import Path

def check_model():
    model_path = Path("./models/qwen2.5-7b-instruct-q4_k_m.gguf")

    print("=== Проверка модели ===")
    print(f"Путь: {model_path.absolute()}")
    print(f"Существует: {model_path.exists()}")

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"Размер: {size_gb:.2f} GB")

        # Проверяем заголовок GGUF
        with open(model_path, 'rb') as f:
            header = f.read(4)
            print(f"Заголовок (первые 4 байта): {header}")
            print(f"Ожидается: b'GGUF' -> {header == b'GGUF'}")

        if size_gb < 4.0:
            print("\n⚠️  Размер слишком мал! Модель повреждена или неполная.")
            print("   Нужно: ~4.5 GB")
    else:
        print("\n❌ Файл не найден!")
        print("   Проверь папку models/")

        models_dir = Path("./models")
        if models_dir.exists():
            print(f"\nФайлы в {models_dir}:")
            for f in models_dir.iterdir():
                size = f.stat().st_size / (1024**3)
                print(f"  {f.name} ({size:.2f} GB)")

if __name__ == "__main__":
    check_model()
