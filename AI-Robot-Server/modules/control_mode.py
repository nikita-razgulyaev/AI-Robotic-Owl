"""Режим управления локомоцией: manual (геймпад/панель) vs autonomous (сам робот).

Единая точка истины для всех слоёв (геймпад, панель, будущие голосовые команды,
будущий автономный планировщик). Переключение:
  - с панели: {type: "set_control_mode", mode: "manual"|"autonomous"}
  - голосом (через quick_answers + handle_command): тот же type
  - из кода: control_mode.set("autonomous")

В manual геймпад и кнопки локомоции на панели активны.
В autonomous геймпад игнорирует оси/кнопки локомоции (чтобы не конфликтовать
с собственными решениями робота); emergency_stop и return_to_default всегда
доступны независимо от режима.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

VALID_MODES = ("manual", "autonomous")


class ControlMode:
    def __init__(self, initial: str = "manual"):
        self._mode = initial if initial in VALID_MODES else "manual"
        self._lock = threading.Lock()
        self._listeners: List[Callable[[str], None]] = []

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def is_manual(self) -> bool:
        return self.mode == "manual"

    def is_autonomous(self) -> bool:
        return self.mode == "autonomous"

    def set(self, mode: str, reason: str = "unknown") -> dict:
        if mode not in VALID_MODES:
            return {"status": "error", "message": f"Неизвестный режим: {mode}. Допустимо: {VALID_MODES}"}
        with self._lock:
            prev = self._mode
            self._mode = mode
        if prev != mode:
            logger.info(f"🎮 Режим управления: {prev} → {mode} ({reason})")
            for cb in list(self._listeners):
                try:
                    cb(mode)
                except Exception as e:
                    logger.warning(f"control_mode listener error: {e}")
        return {"status": "ok", "mode": mode, "previous": prev, "reason": reason}

    def on_change(self, callback: Callable[[str], None]):
        self._listeners.append(callback)

    def to_dict(self) -> dict:
        return {"control_mode": self.mode, "modes": list(VALID_MODES)}


control_mode = ControlMode(initial="manual")
