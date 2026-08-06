"""Конвертация LoRA адаптера в GGUF формат для llama.cpp"""
import os
import sys
import argparse
import torch
from pathlib import Path


def convert_lora_to_gguf(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    out_type: str = "q4_k_m"
):
    """
    Конвертирует LoRA адаптер в GGUF формат.

    Требует:
    1. llama.cpp скомпилированный с поддержкой convert
    2. python convert-lora-to-ggml.py из llama.cpp

    Args:
        base_model_path: Путь к базовой модели (HuggingFace формат)
        adapter_path: Путь к LoRA адаптеру
        output_path: Путь для сохранения GGUF
        out_type: Тип квантизации (q4_k_m, q5_k_m, q8_0, f16)
    """

    print(f"🔄 Конвертация LoRA → GGUF")
    print(f"   Базовая модель: {base_model_path}")
    print(f"   Адаптер: {adapter_path}")
    print(f"   Выход: {output_path}")
    print()

    # Проверяем наличие llama.cpp
    llama_cpp_dir = Path(os.getenv("LLAMA_CPP_DIR", "./llama.cpp"))
    if not llama_cpp_dir.exists():
        print(f"❌ llama.cpp не найден в {llama_cpp_dir}")
        print("   Скачай: git clone https://github.com/ggerganov/llama.cpp")
        print("   Собери: cmake -B build && cmake --build build --config Release")
        return False

    convert_script = llama_cpp_dir / "convert_lora_to_ggml.py"
    if not convert_script.exists():
        convert_script = llama_cpp_dir / "convert-lora-to-ggml.py"

    if not convert_script.exists():
        print(f"❌ Скрипт конвертации не найден: {convert_script}")
        print("   Убедись, что llama.cpp обновлён до последней версии")
        return False

    # Способ 1: Через merge_and_unload + конвертация в GGUF
    print("📦 Способ 1: Merge LoRA + конвертация...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("   Загрузка базовой модели...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        print("   Загрузка LoRA адаптера...")
        model = PeftModel.from_pretrained(model, adapter_path)

        print("   Слияние весов...")
        model = model.merge_and_unload()

        # Сохраняем merged модель
        merged_dir = Path(adapter_path) / "merged"
        merged_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

        print(f"   Слитая модель сохранена: {merged_dir}")
        print()
        print("   Теперь конвертируй в GGUF:")
        print(f"   python {llama_cpp_dir}/convert_hf_to_gguf.py {merged_dir} --outfile {output_path} --outtype {out_type}")

        return True

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        print()

    # Способ 2: Прямая конвертация через llama.cpp
    print("📦 Способ 2: Прямая конвертация через llama.cpp...")
    print(f"   python {convert_script} {adapter_path} --outtype {out_type}")
    print()
    print("   Или используй llama-quantize:")
    print(f"   {llama_cpp_dir}/build/bin/llama-quantize {output_path} {output_path}.q4_k_m.bin q4_k_m")

    return False


def main():
    parser = argparse.ArgumentParser(description="Конвертация LoRA в GGUF")
    parser.add_argument("--base", required=True, help="Путь к базовой модели")
    parser.add_argument("--adapter", required=True, help="Путь к LoRA адаптеру")
    parser.add_argument("--out", default="./models/soren_lora.gguf", help="Выходной файл")
    parser.add_argument("--outtype", default="q4_k_m", help="Тип квантизации")

    args = parser.parse_args()
    convert_lora_to_gguf(args.base, args.adapter, args.out, args.outtype)


if __name__ == "__main__":
    main()