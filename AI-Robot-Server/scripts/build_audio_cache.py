"""Офлайн-генерация WAV для всех 'responses[].text' из character/quick_answers.json.

Идея: TTS (Silero/edge-tts) вызывается ОДИН РАЗ на этапе сборки, а не в
рантайме на каждый вопрос пользователя. В рантайме quick_answers просто
читает готовый файл с диска и отправляет байты — TTS-движок вообще не
трогается для этого пути (см. modules/quick_answers.py, поле 'audio').

Идемпотентен: если файл для конкретного id/варианта уже существует на
диске — пропускается. Добавили новый вариант в JSON → перезапустили
скрипт → сгенерировался только новый файл, остальные не трогаются.

Запуск:
    python scripts/build_audio_cache.py            # только недостающие
    python scripts/build_audio_cache.py --force     # перегенерировать всё
    python scripts/build_audio_cache.py --speaker X # конкретный голос TTS
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CHARACTER_DIR
from modules.tts import TTSEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUICK_ANSWERS_PATH = CHARACTER_DIR / "quick_answers.json"
AUDIO_CACHE_DIR = CHARACTER_DIR.parent / "audio_cache"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="перегенерировать даже существующие файлы")
    parser.add_argument("--speaker", default=None, help="голос TTS (см. TTSEngine.get_available_speakers())")
    args = parser.parse_args()

    if not QUICK_ANSWERS_PATH.exists():
        logger.error(f"Не найден {QUICK_ANSWERS_PATH}")
        sys.exit(1)

    with open(QUICK_ANSWERS_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tts = TTSEngine()

    total = 0
    generated = 0
    skipped_existing = 0
    missing_audio_field = 0

    for entry in entries:
        entry_id = entry.get("id", "?")
        responses = entry.get("responses")
        if not responses:
            # старая форма ('response' без 'responses') — офлайн-кэш под неё
            # не строим: смысл миграции на 'responses' как раз в том, чтобы
            # явно проставить путь 'audio' для каждой записи, которую хотим кэшировать
            continue

        for i, resp in enumerate(responses):
            total += 1
            audio_rel = resp.get("audio")
            if not audio_rel:
                missing_audio_field += 1
                logger.warning(f"[{entry_id}][{i}] нет поля 'audio' в JSON — пропущено (не будет закэшировано)")
                continue

            out_path = CHARACTER_DIR / audio_rel
            if out_path.exists() and not args.force:
                skipped_existing += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            text = resp["text"]
            logger.info(f"[{entry_id}][{i}] синтез: {text[:60]!r} → {out_path}")
            try:
                tts.synthesize_to_wav(text, output_path=out_path, speaker=args.speaker)
                generated += 1
            except Exception as e:
                logger.error(f"[{entry_id}][{i}] ошибка синтеза: {e}")

    logger.info(
        f"Готово: {generated} сгенерировано, {skipped_existing} уже были на диске "
        f"(используйте --force для перегенерации), {missing_audio_field} без поля 'audio' из {total} вариантов"
    )


if __name__ == "__main__":
    main()
