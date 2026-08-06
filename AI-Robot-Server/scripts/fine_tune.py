"""Fine-tuning Сорена через LoRA на CPU (без квантизации)"""
import os
import json
import torch
import argparse
from pathlib import Path

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установи: pip install transformers peft datasets accelerate")
    exit(1)


def load_dataset(dataset_path: Path):
    """Загружает датасет из JSONL"""
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    formatted = []
    for ex in examples:
        messages = ex.get("messages", [])
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text += f"<|im_start|>system\n{content}\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}\n"
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def find_all_linear_names(model):
    """Находит все linear слои для LoRA"""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def train_lora_cpu(
    base_model_path: str,
    dataset_path: Path,
    output_dir: Path,
    lora_r: int = 4,
    lora_alpha: int = 8,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 256,
):
    """Обучает LoRA адаптер на CPU (без квантизации)"""

    print(f"🦉 Fine-tuning Сорена (CPU, LoRA)")
    print(f"   Базовая модель: {base_model_path}")
    print(f"   Датасет: {dataset_path}")
    print(f"   LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"   Epochs: {epochs}, LR: {learning_rate}")
    print(f"   ⚠️ CPU-only — будет медленно!")
    print()

    # 1. Загружаем токенизатор
    print("📖 Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Загружаем модель в fp16 (без квантизации)
    print("🧠 Загрузка модели (fp16)...")
    print("   ⏳ Это займёт время и ~14 GB RAM...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print(f"   ✅ Модель загружена")
    print(f"   📊 Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # 3. Настройка LoRA
    print("⚙️ Настройка LoRA...")
    target_modules = find_all_linear_names(model)
    print(f"   Target modules: {target_modules[:5]}...")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Загружаем датасет
    print("📊 Загрузка датасета...")
    dataset = load_dataset(dataset_path)
    print(f"   Примеров: {len(dataset)}")

    # 5. Токенизация
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 6. Настройка обучения (CPU-оптимизировано)
    print("🏋️ Начало обучения...")
    print("   ⏳ Это займёт время...")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 7. Обучение
    trainer.train()

    # 8. Сохранение
    print("💾 Сохранение адаптера...")
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"✅ Готово! Адаптер сохранён в: {adapter_dir}")

    # Показываем размер
    import glob
    files = glob.glob(str(adapter_dir / "*"))
    total = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    print(f"   📦 Размер: {total / 1024**2:.1f} MB")

    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Сорена на CPU")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="./datasets/curated_v1_fixed.jsonl")
    parser.add_argument("--output", default="./models/lora_output")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_lora_cpu(
        base_model_path=args.base_model,
        dataset_path=Path(args.dataset),
        output_dir=output_dir,
        lora_r=args.r,
        lora_alpha=args.alpha,
        epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()