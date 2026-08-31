"""Будильник (wake word) — команды принимаются только после слова "Сорен".

Отдельного always-on голосового wake-word движка (Porcupine/openWakeWord) нет
и не нужен: STT (faster-whisper) и так гоняет через себя каждую фразу,
которую нарезал VAD (см. RobotBrain.process_audio_chunk), поэтому дешевле
и надёжнее искать будильник уже в распознанном тексте, а не городить
отдельный аудио-пайплайн специально под одно слово.

Матчинг — fuzzy по SequenceMatcher (тот же приём, что и в quick_answers.py),
т.к. Whisper на коротком имени может услышать что угодно: "сорен" -> "сарен",
"сорена", "серен" и т.п. Проверяем только первые несколько слов фразы —
будильник у пользователя всегда в начале обращения к роботу, а не в середине
или конце ("... и передай привет Сорену" не должно случайно сработать).
"""
import re
from difflib import SequenceMatcher
from typing import Optional

WAKE_WORD = "сорен"

# Порог ниже, чем FUZZY_THRESHOLD в quick_answers.py (0.85): слово короткое,
# и даже одна спутанная буква в 5-6-буквенном имени резко проседает по ratio.
# Но не ниже 0.75 — иначе будильник начнёт ложно матчить случайные короткие
# слова ("да", "нет" и т.п. дают низкий, но не нулевой score с "сорен").
FUZZY_THRESHOLD = 0.75

# Ищем будильник среди первых N слов фразы, а не только самого первого —
# Whisper иногда добавляет спереди мусорное слово-паразит ("ну сорен включи
# свет", "э сорен потухни"). Намеренно не больше 2: с 3+ слов начинают
# ложно матчиться фразы НЕ к роботу, где имя просто упомянуто дальше в
# предложении ("передай привет сорену коллеге" не должно включать будильник).
MAX_LOOKAHEAD = 2

_EDGE_PUNCT_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def _clean_word(word: str) -> str:
    """Слово без окружающей пунктуации, в нижнем регистре — только для
    сравнения с эталоном; сам текст при вырезании остаётся как был."""
    return _EDGE_PUNCT_RE.sub("", word).lower()


def _matches_wake_word(cleaned_word: str) -> bool:
    if not cleaned_word:
        return False
    if cleaned_word == WAKE_WORD:
        return True
    return SequenceMatcher(None, cleaned_word, WAKE_WORD).ratio() >= FUZZY_THRESHOLD


def strip_wake_word(text: str) -> Optional[str]:
    """Ищет "Сорен" среди первых слов фразы.

    Возвращает:
      - остаток фразы без будильника (может быть пустой строкой, если
        пользователь просто позвал робота по имени без команды);
      - None, если будильник в фразе не найден вообще.

    Регистр и пунктуация остальной части текста не трогаются — отрезается
    только сам токен с именем, чтобы дальше по пайплайну (fuzzy_matcher,
    quick_answers, LLM) шёл текст, максимально похожий на то, что реально
    сказал пользователь.
    """
    if not text or not text.strip():
        return None

    tokens = text.split()
    lookahead = min(MAX_LOOKAHEAD, len(tokens))

    for i in range(lookahead):
        if _matches_wake_word(_clean_word(tokens[i])):
            rest_tokens = tokens[:i] + tokens[i + 1:]
            return " ".join(rest_tokens).strip()

    return None
