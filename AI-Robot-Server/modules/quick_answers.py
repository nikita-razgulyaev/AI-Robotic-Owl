"""Словарь быстрых ответов — мгновенная реакция на заранее заданные вопросы
и команды без обращения к LLM (ни локальной, ни облачной). Используется для
частых вопросов ("как тебя зовут") и команд роботу ("помаши крылом", "кивни") —
там, где ответ всегда один и тот же и ждать секунды на "раздумья" ИИ незачем.

Формат файла — character/quick_answers.json, список объектов:
  {
    "id": "уникальный_id",
    "triggers": ["точная фраза 1", "фраза 2", ...],
    "response": "текст ответа",
    "emotion": "calm",     # опционально, см. character/Soren_emotions.json
    "action": "wave"       # опционально, см. config.settings.ANIMATIONS
  }

Матчинг в два уровня:
  1) точное совпадение после нормализации (регистр/пунктуация/пробелы) —
     мгновенный поиск по dict, без перебора;
  2) нечёткое совпадение по SequenceMatcher (тот же подход, что и в
     modules/fuzzy_matcher.py) с высоким порогом — чтобы случайно не
     подменить настоящий вопрос к ИИ похожим по буквам быстрым ответом.
"""
import json
import logging
import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import CHARACTER_DIR

logger = logging.getLogger(__name__)

QUICK_ANSWERS_PATH = CHARACTER_DIR / "quick_answers.json"

# Порог нечёткого совпадения (0.0-1.0). Намеренно высокий: ложное срабатывание
# на реальном вопросе к ИИ куда хуже, чем один пропущенный быстрый ответ —
# в этом случае просто отработает обычный путь через LLM.
FUZZY_THRESHOLD = 0.85

_PUNCT_RE = re.compile(r"[^\w\sа-яёa-z0-9]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Нижний регистр, без пунктуации, схлопнутые пробелы — для сравнения фраз"""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass
class QuickAnswer:
    id: str
    triggers: List[str]
    response: str
    emotion: Optional[str] = None
    action: Optional[str] = None


class QuickAnswerBook:
    """Загружает и матчит словарь быстрых ответов. Потокобезопасен, поддерживает hot-reload"""

    def __init__(self, path: Path = QUICK_ANSWERS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._entries: List[QuickAnswer] = []
        # normalized_trigger -> QuickAnswer, для мгновенного точного совпадения без перебора
        self._exact_index: Dict[str, QuickAnswer] = {}
        self.reload()

    def reload(self) -> int:
        """Перечитывает character/quick_answers.json с диска. Возвращает число записей.
        Можно вызывать прямо во время работы сервера — правки в JSON подхватятся
        без рестарта (см. POST /quick_answers/reload)"""
        entries: List[QuickAnswer] = []
        exact_index: Dict[str, QuickAnswer] = {}

        if not self.path.exists():
            logger.warning(f"Файл быстрых ответов не найден: {self.path}")
        else:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    qa = QuickAnswer(
                        id=item["id"],
                        triggers=item["triggers"],
                        response=item["response"],
                        emotion=item.get("emotion"),
                        action=item.get("action"),
                    )
                    entries.append(qa)
                    for trigger in qa.triggers:
                        norm = _normalize(trigger)
                        if norm:
                            exact_index[norm] = qa
            except Exception as e:
                logger.error(f"Ошибка загрузки быстрых ответов: {e}")

        with self._lock:
            self._entries = entries
            self._exact_index = exact_index

        total_triggers = sum(len(e.triggers) for e in entries)
        logger.info(f"⚡ Словарь быстрых ответов: {len(entries)} записей, {total_triggers} триггеров")
        return len(entries)

    def find(self, text: str) -> Optional[QuickAnswer]:
        """Ищет быстрый ответ на текст пользователя. None, если ничего не подошло
        (тогда вызывающий код идёт обычным путём через LLM)"""
        norm = _normalize(text)
        if not norm:
            return None

        with self._lock:
            entries = self._entries
            exact_index = self._exact_index

        # 1) Точное совпадение — мгновенно
        exact = exact_index.get(norm)
        if exact:
            return exact

        # 2) Триггер целиком входит в текст пользователя как подстрока —
        # покрывает случаи с лишними словами вокруг ("помаши крылом пожалуйста",
        # "ну давай кивни"). Сравниваем по границам слов (с пробелами по краям),
        # иначе короткие триггеры вроде "пока" ложно матчатся внутри "покачай".
        padded = f" {norm} "
        for qa in entries:
            for trigger in qa.triggers:
                norm_trigger = _normalize(trigger)
                if len(norm_trigger) >= 4 and f" {norm_trigger} " in padded:
                    return qa

        # 3) Нечёткое совпадение по всей строке — для лёгких искажений STT
        # самого триггера ("памаши крылом" вместо "помаши крылом").
        # Словарь на десятки-сотни записей перебирается за доли миллисекунды —
        # это всё равно на порядки быстрее любого похода в LLM.
        best_qa, best_score = None, 0.0
        for qa in entries:
            for trigger in qa.triggers:
                score = SequenceMatcher(None, norm, _normalize(trigger)).ratio()
                if score > best_score:
                    best_score, best_qa = score, qa

        if best_qa and best_score >= FUZZY_THRESHOLD:
            logger.info(f"⚡ Быстрый ответ по нечёткому совпадению ({best_score:.0%}): '{text}' → '{best_qa.id}'")
            return best_qa

        return None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# Глобальный экземпляр — грузится один раз при импорте модуля (десятки/сотни
# записей, файл маленький — загрузка занимает миллисекунды)
quick_answers = QuickAnswerBook()
