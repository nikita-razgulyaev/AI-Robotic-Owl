"""Скрипт для скачивания Qwen GGUF модели с HuggingFace"""
import os
import sys
from pathlib import Path

def download_with_hf_hub():
    """Скачивает модель через hf_hub_download (автоматически собирает части)"""
    try:
        from huggingface_hub import hf_hub_download
        print("Скачивание Qwen2.5-7B-Instruct-GGUF...")
        print("Это займет 10-30 минут в зависимости от интернета.\n")

        # hf_hub_download автоматически обрабатывает разбитые файлы
        downloaded_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            local_dir="./models",
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\n✅ Модель скачана: {downloaded_path}")
        size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
        print(f"   Размер: {size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def download_parts_manually():
    """Ручное скачивание частей и сборка"""
    import urllib.request
    from tqdm import tqdm

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    base_url = "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/"

    files = [
        "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
    ]

    print("Скачивание частей модели...\n")

    for filename in files:
        filepath = models_dir / filename
        if filepath.exists():
            print(f"✓ {filename} уже существует")
            continue

        url = base_url + filename
        print(f"↓ Скачивание {filename}...")

        try:
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
            print(f"✓ {filename} скачан\n")

        except Exception as e:
            print(f"✗ Ошибка скачивания {filename}: {e}")
            return False

    # Собираем части
    print("Сборка модели из частей...")
    output_file = models_dir / "qwen2.5-7b-instruct-q4_k_m.gguf"

    with open(output_file, 'wb') as outfile:
        for filename in files:
            filepath = models_dir / filename
            with open(filepath, 'rb') as infile:
                outfile.write(infile.read())
            print(f"  Добавлен: {filename}")

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n✅ Модель собрана: {output_file}")
    print(f"   Размер: {size_mb:.1f} MB")

    # Удаляем части (опционально)
    # for filename in files:
    #     (models_dir / filename).unlink()
    # print("Части удалены")

    return True


def main():
    print("=" * 60)
    print("  🤖 Загрузка Qwen2.5-7B-Instruct-GGUF")
    print("=" * 60)
    print()

    # Пробуем через hf_hub_download
    if download_with_hf_hub():
        return

    print("\nПробуем ручное скачивание...\n")
    download_parts_manually()


if __name__ == "__main__":
    main()
