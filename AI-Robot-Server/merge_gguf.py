"""Скрипт для сборки GGUF модели из частей"""
import os
import sys
from pathlib import Path

def merge_gguf_parts():
    """Собирает разбитые части GGUF в один файл"""

    models_dir = Path("./models")

    # Ищем части
    parts = sorted([
        f for f in models_dir.iterdir() 
        if f.name.startswith("qwen2.5-7b-instruct-q4_k_m-") 
        and f.name.endswith(".gguf")
        and "of" in f.name
    ])

    if not parts:
        print("❌ Части не найдены в ./models/")
        print("   Сначала скачай части:")
        print('   hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_k_m*.gguf" --local-dir ./models')
        return False

    print(f"Найдено частей: {len(parts)}")
    for p in parts:
        size_gb = os.path.getsize(p) / (1024**3)
        print(f"  {p.name} ({size_gb:.2f} GB)")

    output_file = models_dir / "qwen2.5-7b-instruct-q4_k_m.gguf"

    if output_file.exists():
        print(f"\n⚠️  Файл уже существует: {output_file}")
        overwrite = input("Перезаписать? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Отменено.")
            return False

    print(f"\nСборка: {output_file.name}")
    print("Это может занять несколько минут...\n")

    with open(output_file, 'wb') as outfile:
        for i, part in enumerate(parts):
            size_gb = os.path.getsize(part) / (1024**3)
            print(f"  [{i+1}/{len(parts)}] {part.name} ({size_gb:.2f} GB)...", end=" ", flush=True)

            with open(part, 'rb') as infile:
                outfile.write(infile.read())

            print("✓")

    total_size = os.path.getsize(output_file) / (1024**3)
    print(f"\n✅ Готово!")
    print(f"   Файл: {output_file}")
    print(f"   Размер: {total_size:.2f} GB")

    # Удаляем части (опционально)
    delete = input("\nУдалить исходные части? (y/n): ").lower().strip()
    if delete == 'y':
        for part in parts:
            part.unlink()
        print("Части удалены.")

    return True


if __name__ == "__main__":
    merge_gguf_parts()
