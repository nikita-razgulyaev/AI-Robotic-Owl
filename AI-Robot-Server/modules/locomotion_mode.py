"""Режим локомоции: walk (ходьба) vs flight (полёт).

Отдельно от control_mode (manual/autonomous): тот отвечает на вопрос «КТО
управляет», этот — «КАК робот сейчас движется». От режима зависит базовая
поза крыльев и то, что делает левый стик геймпада:

  walk    крылья (сервы взмаха CH0/CH3) прижаты на 0°, стик вперёд/назад → шаги
  flight  крылья в нейтрали 90°, стик вперёд → цикл взмаха (modules/wing_flap.py)

Переключение:
  - геймпад: кнопки из modules/gamepad_input.py (L1 — ходьба, R1 — полёт)
  - панель/код: {type: "set_locomotion_mode", mode: "walk"|"flight"}
  - из кода: locomotion_mode.set("flight")
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, List

logger = logging.getLogger(__name__)

WALK = "walk"
FLIGHT = "flight"
VALID_MODES = (WALK, FLIGHT)


class LocomotionMode:
    def __init__(self, initial: str = WALK):
        self._mode = initial if initial in VALID_MODES else WALK
        self._lock = threading.Lock()
        self._listeners: List[Callable[[str], None]] = []

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def is_walk(self) -> bool:
        return self.mode == WALK

    def is_flight(self) -> bool:
        return self.mode == FLIGHT

    def set(self, mode: str, reason: str = "unknown") -> dict:
        if mode not in VALID_MODES:
            return {"status": "error", "message": f"Неизвестный режим: {mode}. Допустимо: {VALID_MODES}"}
        with self._lock:
            prev = self._mode
            self._mode = mode
        if prev != mode:
            logger.info(f"🚶🪽 Режим локомоции: {prev} → {mode} ({reason})")
            for cb in list(self._listeners):
                try:
                    cb(mode)
                except Exception as e:
                    logger.warning(f"locomotion_mode listener error: {e}")
        return {"status": "ok", "mode": mode, "previous": prev, "reason": reason}

    def on_change(self, callback: Callable[[str], None]):
        self._listeners.append(callback)

    def to_dict(self) -> dict:
        return {"locomotion_mode": self.mode, "locomotion_modes": list(VALID_MODES)}


locomotion_mode = LocomotionMode(initial=WALK)
