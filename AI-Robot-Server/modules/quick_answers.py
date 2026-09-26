"""Словарь быстрых ответов — мгновенная реакция на заранее заданные вопросы
и команды без обращения к LLM (ни локальной, ни облачной). Используется для
частых вопросов ("как тебя зовут") и команд роботу ("помаши крылом", "кивни") —
там, где ответ всегда один и тот же и ждать секунды на "раздумья" ИИ незачем.

Формат файла — character/quick_answers.json, список объектов. Поддерживаются
ДВЕ формы записи (новая и старая — для обратной совместимости):

Новая форма (несколько вариантов ответа на один триггер — чтобы одна и та же
фраза пользователя не звучала как заезженная пластинка при повторах):
  {
    "id": "уникальный_id",
    "triggers": ["точная фраза 1", "фраза 2", ...],
    "responses": [
      {"text": "...", "audio": "cache/how_are_you_1.wav", "emotion": "calm",
       "action": "tilt_head", "sfx": null},
      {"text": "...", "audio": "cache/how_are_you_2.wav", "emotion": "curious"}
    ]
  }

Старая форма (один ответ — эквивалентна responses из одного варианта):
  {
    "id": "уникальный_id",
    "triggers": [...],
    "response": "текст ответа",
    "emotion": "calm",     # опционально
    "action": "wave"       # опционально
  }

Поле "audio" — путь (относительно character/) к заранее сгенерированному
WAV-файлу (см. scripts/build_audio_cache.py). Если не указано или файла нет
на диске — вызывающий код должен сам сделать fallback на live TTS от text.

Матчинг в два уровня (без изменений от предыдущей версии):
  1) точное совпадение после нормализации (регистр/пунктуация/пробелы) —
     мгновенный поиск по dict, без перебора;
  2) нечёткое совпадение по SequenceMatcher с высоким порогом — чтобы
     случайно не подменить настоящий вопрос к ИИ похожим по буквам ответом.
"""
import json
import logging
import random
import re
import threading
from dataclasses import dataclass, field
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

# Сколько последних вариантов ответа помнить на каждый id, чтобы не повторять подряд
RESPONSE_ANTI_REPEAT_WINDOW = 2

_PUNCT_RE = re.compile(r"[^\w\sа-яёa-z0-9]", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Нижний регистр, без пунктуации, схлопнутые пробелы — для сравнения фраз"""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


@dataclass
class ResponseVariant:
    text: str
    audio: Optional[str] = None   # относительный путь к WAV в character/, см. build_audio_cache.py
    emotion: Optional[str] = None
    action: Optional[str] = None
    sfx: Optional[str] = None     # короткий non-verbal звук, проигрывается независимо от action


@dataclass
class QuickAnswer:
    id: str
    triggers: List[str]
    responses: List[ResponseVariant] = field(default_factory=list)


class QuickAnswerBook:
    """Загружает и матчит словарь быстрых ответов. Потокобезопасен, поддерживает hot-reload"""

    def __init__(self, path: Path = QUICK_ANSWERS_PATH):
        self.path = path
        self._lock = threading.Lock()
        self._entries: List[QuickAnswer] = []
        # normalized_trigger -> QuickAnswer, для мгновенного точного совпадения без перебора
        self._exact_index: Dict[str, QuickAnswer] = {}
        # id -> последние выданные индексы responses, для антиповтора между вызовами
        self._recent_variant_idx: Dict[str, List[int]] = {}
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
                    responses = self._parse_responses(item)
                    if not responses:
                        logger.error(f"Быстрый ответ '{item.get('id')}': нет ни 'responses', ни 'response' — запись пропущена")
                        continue
                    qa = QuickAnswer(id=item["id"], triggers=item["triggers"], responses=responses)
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
            # не сбрасываем _recent_variant_idx — hot-reload не должен разрешать
            # мгновенный повтор варианта, который только что звучал

        total_triggers = sum(len(e.triggers) for e in entries)
        total_variants = sum(len(e.responses) for e in entries)
        logger.info(f"⚡ Словарь быстрых ответов: {len(entries)} записей, {total_triggers} триггеров, {total_variants} вариантов ответа")
        return len(entries)

    @staticmethod
    def _parse_responses(item: dict) -> List[ResponseVariant]:
        """Новая форма ('responses': [...]) или старая ('response'/'emotion'/'action'),
        приведённая к единому List[ResponseVariant]."""
        if "responses" in item and item["responses"]:
            return [
                ResponseVariant(
                    text=r["text"],
                    audio=r.get("audio"),
                    emotion=r.get("emotion"),
                    action=r.get("action"),
                    sfx=r.get("sfx"),
                )
                for r in item["responses"]
            ]
        if "response" in item:
            return [ResponseVariant(text=item["response"], emotion=item.get("emotion"), action=item.get("action"))]
        return []

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

    def pick_response(self, qa: QuickAnswer) -> ResponseVariant:
        """Взвешенный выбор варианта ответа без повтора последних
        RESPONSE_ANTI_REPEAT_WINDOW индексов для этого id. Если у записи
        всего один вариант — просто его и возвращает (антиповтор не имеет смысла)."""
        if len(qa.responses) <= 1:
            return qa.responses[0]

        with self._lock:
            recent = self._recent_variant_idx.get(qa.id, [])
            pool_idx = [i for i in range(len(qa.responses)) if i not in recent]
            if not pool_idx:
                pool_idx = list(range(len(qa.responses)))
            chosen = random.choice(pool_idx)
            recent = (recent + [chosen])[-RESPONSE_ANTI_REPEAT_WINDOW:]
            self._recent_variant_idx[qa.id] = recent

        return qa.responses[chosen]

    def get_by_id(self, entry_id: str) -> Optional[QuickAnswer]:
        """Прямой поиск по id записи (не по тексту триггера) — например,
        для привязки sfx/audio к событиям, не связанным с фразой пользователя
        (взлёт/посадка по геймпаду, см. modules.motion_primitives)."""
        with self._lock:
            for qa in self._entries:
                if qa.id == entry_id:
                    return qa
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


# Глобальный экземпляр — грузится один раз при импорте модуля (десятки/сотни
# записей, файл маленький — загрузка занимает миллисекунды)
quick_answers = QuickAnswerBook()
