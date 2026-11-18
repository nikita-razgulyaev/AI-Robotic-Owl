"""Парсит curated_v1.jsonl через регулярки (не JSON.parse)"""
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_curated_format(input_path: Path, output_jsonl: Path, output_txt: Path):
    """Парсит специальный формат curated_v1 через регулярки"""

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Исходный размер: {len(content)} символов")

    pattern = r'"messages":\s*\[(.*?)\]\s*\}'
    matches = re.findall(pattern, content, re.DOTALL)

    print(f"Найдено блоков messages: {len(matches)}")

    examples = []
    for i, block in enumerate(matches):
        msg_pattern = r'\{\s*"role":\s*"([^"]+)"\s*,\s*"content":\s*"(.*?)"\s*\}'
        msg_matches = re.findall(msg_pattern, block, re.DOTALL)

        messages = []
        for role, content in msg_matches:
            content = content.replace('\n', ' ').replace('\r', ' ')
            content = ' '.join(content.split())
            messages.append({"role": role, "content": content})

        if len(messages) >= 2:
            examples.append({"messages": messages})
            print(f"   Пример {i+1}: {len(messages)} сообщений")

    print(f"Успешно распарсено: {len(examples)} примеров")

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"JSONL сохранен: {output_jsonl}")

    with open(output_txt, 'w', encoding='utf-8') as f:
        for i, ex in enumerate(examples):
            messages = ex["messages"]

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "").strip()

                if role == "system":
                    f.write(f"### System:\n{content}\n\n")
                elif role == "user":
                    f.write(f"### User:\n{content}\n\n")
                elif role == "assistant":
                    f.write(f"### Assistant:\n{content}\n\n")

            if i < len(examples) - 1:
                f.write("<|endoftext|>\n\n")

    print(f"TXT сохранен: {output_txt}")

    if examples:
        print(f"\nПервый пример ({len(examples[0]['messages'])} сообщений):")
        for msg in examples[0]['messages'][:2]:
            role = msg.get("role", "?")
            content = msg.get("content", "")[:80]
            print(f"   [{role}]: {content}...")

    return len(examples)


if __name__ == "__main__":
    input_file = ROOT_DIR / "datasets" / "curated_v1.jsonl"
    output_jsonl = ROOT_DIR / "datasets" / "curated_v1_fixed.jsonl"
    output_txt = ROOT_DIR / "datasets" / "train_llamacpp.txt"

    if not input_file.exists():
        print(f"Файл не найден: {input_file}")
        exit(1)

    count = parse_curated_format(input_file, output_jsonl, output_txt)
    print(f"\nГотово! {count} примеров обработано.")