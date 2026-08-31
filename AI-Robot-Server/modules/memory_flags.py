"""Runtime-переключатели уровней памяти LLM.

В отличие от FAST_MODE (config.yaml, читается один раз при старте — см.
config/settings.py), эти флаги можно менять прямо во время работы сервера
через веб-панель (/memory_config), и они переживают перезапуск — состояние
сохраняется в JSON-файл рядом с остальной памятью (MEMORY_DIR).

Уровни:
  stm     — краткосрочная память (последние N реплик в RAM)
  ltm     — долгосрочная память (Qdrant: диалоги, эмоциональные моменты, факты)
  profile — эмоциональный профиль (доминантные эмоции, привычки, приветствие)
  rag     — RAG канона мира (поиск по Soren_rag_chunks в LLMEngine)

При FAST_MODE=true в config.yaml дефолты для ltm/profile/rag — False, но их
всё равно можно включить из панели: тяжёлые ресурсы (Qdrant, энкодер, файл
канона) в этом случае поднимаются лениво при первом включении, а не остаются
выключенными до перезапуска.
"""
import json
import logging
import threading
from pathlib import Path
from typing import Dict

from config.settings import MEMORY_DIR, FAST_MODE

logger = logging.getLogger(__name__)

_FLAGS_PATH = MEMORY_DIR / "memory_flags.json"
_LOCK = threading.Lock()

_DEFAULTS: Dict[str, bool] = {
    "stm": True,
    "ltm": not FAST_MODE,
    "profile": not FAST_MODE,
    "rag": not FAST_MODE,
}

_flags: Dict[str, bool] = dict(_DEFAULTS)


def _load() -> None:
    global _flags
    if not _FLAGS_PATH.exists():
        _flags = dict(_DEFAULTS)
        return
    try:
        with open(_FLAGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(_DEFAULTS)
        for key in _DEFAULTS:
            if key in data:
                merged[key] = bool(data[key])
        _flags = merged
        logger.info(f"Флаги памяти загружены из {_FLAGS_PATH.name}: {_flags}")
    except Exception as e:
        logger.error(f"Ошибка загрузки флагов памяти, использую дефолты: {e}")
        _flags = dict(_DEFAULTS)


def _save() -> None:
    try:
        with open(_FLAGS_PATH, "w", encoding="utf-8") as f:
            json.dump(_flags, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения флагов памяти: {e}")


def get_flags() -> Dict[str, bool]:
    """Копия текущего состояния всех уровней памяти"""
    with _LOCK:
        return dict(_flags)


def set_flag(name: str, value: bool) -> Dict[str, bool]:
    """Включает/выключает один уровень памяти, сохраняет на диск, возвращает всё состояние"""
    if name not in _DEFAULTS:
        raise ValueError(f"Неизвестный уровень памяти: {name}")
    with _LOCK:
        _flags[name] = bool(value)
        _save()
        result = dict(_flags)
    logger.info(f"Уровень памяти '{name}' → {'включён' if value else 'выключен'}")
    return result


_load()
