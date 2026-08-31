"""Переключатели подсветки-ночника в подставке — состояние персистится в JSON
(переживает перезапуск сервера), аналогично modules/memory_flags.py.

Два независимых флага:
  auto   — авто-режим (закат/рассвет + подключение устройства). По умолчанию
           ВЫКЛЮЧЕН (см. config.yaml backlight.enabled) — ничего не горит
           само по себе, пока не включат явно из панели.
  manual — ручное состояние, имеет эффект только когда auto == False
           (см. compute_effective_state() в websocket_server.py).
"""
import json
import logging
import threading
from typing import Dict

from config.settings import (
    MEMORY_DIR, BACKLIGHT_AUTO_DEFAULT, BACKLIGHT_MANUAL_DEFAULT
)

logger = logging.getLogger(__name__)

_STATE_PATH = MEMORY_DIR / "backlight_state.json"
_LOCK = threading.Lock()

_DEFAULTS: Dict[str, bool] = {
    "auto": BACKLIGHT_AUTO_DEFAULT,
    "manual": BACKLIGHT_MANUAL_DEFAULT,
}

_state: Dict[str, bool] = dict(_DEFAULTS)


def _load() -> None:
    global _state
    if not _STATE_PATH.exists():
        _state = dict(_DEFAULTS)
        return
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(_DEFAULTS)
        for key in _DEFAULTS:
            if key in data:
                merged[key] = bool(data[key])
        _state = merged
        logger.info(f"Состояние подсветки загружено из {_STATE_PATH.name}: {_state}")
    except Exception as e:
        logger.error(f"Ошибка загрузки состояния подсветки, использую дефолты: {e}")
        _state = dict(_DEFAULTS)


def _save() -> None:
    try:
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния подсветки: {e}")


def get_state() -> Dict[str, bool]:
    with _LOCK:
        return dict(_state)


def set_auto(value: bool) -> Dict[str, bool]:
    with _LOCK:
        _state["auto"] = bool(value)
        _save()
        result = dict(_state)
    logger.info(f"Подсветка: авто-режим → {'включён' if value else 'выключен'}")
    return result


def set_manual(value: bool) -> Dict[str, bool]:
    with _LOCK:
        _state["manual"] = bool(value)
        _save()
        result = dict(_state)
    logger.info(f"Подсветка: ручное состояние → {'вкл' if value else 'выкл'}")
    return result


_load()
