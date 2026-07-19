"""Скрипт для автоматической загрузки всех моделей"""
import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path, desc: str = ""):
    """Скачивает файл с прогресс-баром"""
    if dest.exists():
        print(f"  ✓ {dest.name} уже существует")
        return

    print(f"  ↓ Скачивание {desc}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
        print(f"  ✓ {dest.name} загружен")
    except Exception as e:
        print(f"  ✗ Ошибка загрузки {dest.name}: {e}")
        if dest.exists():
            dest.unlink()


def download_whisper_model():
    """Whisper модели скачиваются автоматически при первом запуске faster-whisper"""
    print("\n🎤 Whisper (STT)")
    print("  Модели скачаются автоматически при первом запуске сервера.")
    print("  Или скачай вручную с: https://huggingface.co/Systran")
    print("  Рекомендуемая: faster-whisper-small (~500MB)")


def download_silero_model():
    """Silero TTS модель"""
    print("\n🔊 Silero TTS - Русский голос")
    print("  Модель скачается автоматически при первом запуске сервера.")
    print("  Или скачай вручную:")
    print("  https://models.silero.ai/models/tts/ru/v4_ru.pt")
    print("  Положи в папку models/silero_v4_ru.pt")


def download_llm_model():
    """Инструкции по загрузке LLM"""
    print("\n🧠 LLM (Qwen2.5 7B)")
    print("  Скачай GGUF модель вручную:")
    print("  1. Перейди на https://huggingface.co")
    print("  2. Найди: Qwen/Qwen2.5-7B-Instruct-GGUF")
    print("  3. Скачай: qwen2.5-7b-instruct-q4_k_m.gguf (~4.5GB)")
    print("  4. Положи в папку models/")
    print("\n  Альтернативы:")
    print("  - unsloth/Llama-3.2-3B-Instruct-GGUF (меньше, ~2GB)")
    print("  - bartowski/Mistral-7B-Instruct-v0.3-GGUF")
    print("  - TheBloke/NeuralBeagle14-7B-GGUF")

    # Попробуем скачать через huggingface-cli если установлен
    print("\n  Попытка автоматической загрузки через huggingface-cli...")
    try:
        import subprocess
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  huggingface-cli найден!")
            print("  Запускаю загрузку Qwen2.5-7B-Instruct-GGUF...")
            subprocess.run([
                "huggingface-cli", "download",
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
                "qwen2.5-7b-instruct-q4_k_m.gguf",
                "--local-dir", str(MODELS_DIR),
                "--local-dir-use-symlinks", "False"
            ])
        else:
            print("  huggingface-cli не найден. Установи: pip install huggingface-hub")
    except Exception as e:
        print(f"  Автоматическая загрузка не удалась: {e}")


def download_yolo_model():
    """YOLO модель скачивается автоматически через ultralytics"""
    print("\n👁 YOLO Vision")
    print("  Модель скачается автоматически при первом запуске.")
    print("  Или скачай вручную:")
    print("  - yolov8n.pt (~6MB): https://github.com/ultralytics/assets/releases")
    print("  - yolov8s.pt (~23MB) - точнее, но медленнее")


def main():
    print("=" * 60)
    print("  🤖 Robot AI Server - Загрузка моделей")
    print("=" * 60)

    download_whisper_model()
    download_silero_model()
    download_llm_model()
    download_yolo_model()

    print("\n" + "=" * 60)
    print("  ✅ Готово! Проверь папку models/")
    print("=" * 60)
    print("\n  Следующие шаги:")
    print("  1. pip install -r requirements.txt")
    print("  2. python download_models.py (для ручной загрузки)")
    print("  3. Скопируй .env.example → .env и настрой")
    print("  4. python websocket_server.py")


if __name__ == "__main__":
    main()
