"""Геймпад — пункт 11 архитектуры.

Физически подключается к ПК по Bluetooth, читается через pygame.joystick.
Раскладка под DualShock 4 (SDL2 / pygame 2.x).

Режимы локомоции (modules/locomotion_mode.py):
  L1 — режим ходьбы (крылья на 0°), R1 — режим полёта (крылья в нейтрали 90°).
Левый стик:
  ходьба: вперёд/назад = шаги, влево/вправо = поворот;
  полёт:  вперёд = цикл взмаха (60…120°), чем дальше стик — тем быстрее
          (частота нарастает плавно), влево/вправо = поворот хвостом.
Взлёт/посадка с геймпада пока отключены (✕/○ свободны); правый стик, □, △,
L2/R2 не назначены.

Команды идут через RobotBrain.handle_command() с source="gamepad".
Локомоция активна только в control_mode == "manual".

on_state(callback) — для визуализации на панели: текущие оси и кнопки.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from modules import motion_primitives
from modules.control_mode import control_mode
from modules.locomotion_mode import FLIGHT, WALK, locomotion_mode

logger = logging.getLogger(__name__)

POLL_HZ = 30

# TUNE: индексы осей/кнопок под DualShock 4 (python -m pygame.examples.joystick)
AXIS_LX = 0
AXIS_LY = 1
AXIS_RX = 2
AXIS_RY = 3
AXIS_L2 = 4
AXIS_R2 = 5

BUTTON_CROSS = 0
BUTTON_CIRCLE = 1
BUTTON_SQUARE = 2
BUTTON_TRIANGLE = 3
BUTTON_L1 = 9
BUTTON_R1 = 10

# TUNE: кнопки переключения режима локомоции
BUTTON_WALK_MODE = BUTTON_L1
BUTTON_FLIGHT_MODE = BUTTON_R1

BUTTON_LABELS = {
    BUTTON_CROSS: ("CROSS", "—"),
    BUTTON_CIRCLE: ("CIRCLE", "—"),
    BUTTON_SQUARE: ("SQUARE", "—"),
    BUTTON_TRIANGLE: ("TRIANGLE", "—"),
    BUTTON_L1: ("L1", "Ходьба"),
    BUTTON_R1: ("R1", "Полёт"),
}
TRIGGER_LABELS = {
    "l2": ("L2", "—"),
    "r2": ("R2", "—"),
}

DEADZONE = 0.15
MAX_STEP_DISTANCE_CM = 20.0
MAX_TURN_DEG = 30.0
MIN_COOLDOWN_S = 0.15
STATE_BROADCAST_HZ = 10
RECONNECT_INTERVAL_S = 2.0  # как часто искать геймпад, пока он не подключён


class GamepadInput:
    def __init__(self, handle_command, on_state: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self._handle_command = handle_command
        self._on_state = on_state
        self._task: Optional[asyncio.Task] = None
        self._joystick = None
        self._cooldown_until = 0.0
        self._connected = False
        self._last_state_ts = 0.0
        self._name = ""
        self._trigger_rest = {AXIS_L2: -1.0, AXIS_R2: -1.0}
        self._prev_pressed: Dict[int, bool] = {}
        self._flap_active = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        if self._task is not None:
            return
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
        except ImportError:
            logger.warning("[gamepad] pygame не установлен (pip install pygame)")
            return
        except Exception as e:
            logger.error(f"[gamepad] Ошибка инициализации pygame: {e}")
            return
        # Цикл стартует всегда: геймпад может быть выключен/не спарен в момент
        # запуска сервера — подключимся, как только он появится.
        self._task = asyncio.create_task(self._loop())

    def _try_connect(self, pygame) -> bool:
        """Пересканирует устройства и подключает первый найденный геймпад."""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                return False
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._name = self._joystick.get_name()
            self._calibrate_triggers()
            self._prev_pressed.clear()
            self._flap_active = False
            self._connected = True
            logger.info(f"[gamepad] Подключен: {self._name}")
            return True
        except Exception as e:
            logger.warning(f"[gamepad] Не удалось подключиться: {e}")
            self._joystick = None
            return False

    async def _disconnect(self, reason: str):
        if self._connected:
            logger.warning(f"[gamepad] Отключён ({reason}) — жду переподключения")
        self._connected = False
        self._joystick = None
        self._flap_active = False
        await self._emit_state({
            "type": "gamepad_state", "connected": False, "name": "",
            "axes": {}, "buttons": {}, "triggers": {},
        })

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._connected = False

    def _calibrate_triggers(self):
        j = self._joystick
        for axis in self._trigger_rest:
            raw = j.get_axis(axis) if j.get_numaxes() > axis else -1.0
            self._trigger_rest[axis] = raw if -1.01 <= raw <= 0.1 else -1.0

    def _trigger_value(self, axis: int) -> float:
        j = self._joystick
        if j.get_numaxes() <= axis:
            return 0.0
        rest = self._trigger_rest[axis]
        value = (j.get_axis(axis) - rest) / (1.0 - rest)
        return max(0.0, min(1.0, value))

    def _read_raw(self) -> Dict[str, Any]:
        j = self._joystick
        lx = j.get_axis(AXIS_LX) if j.get_numaxes() > AXIS_LX else 0.0
        ly = j.get_axis(AXIS_LY) if j.get_numaxes() > AXIS_LY else 0.0
        rx = j.get_axis(AXIS_RX) if j.get_numaxes() > AXIS_RX else 0.0
        ry = j.get_axis(AXIS_RY) if j.get_numaxes() > AXIS_RY else 0.0

        buttons = {}
        for idx, (label, purpose) in BUTTON_LABELS.items():
            pressed = bool(j.get_button(idx)) if j.get_numbuttons() > idx else False
            buttons[label] = {"pressed": pressed, "purpose": purpose, "index": idx}

        triggers = {}
        for key, axis in (("l2", AXIS_L2), ("r2", AXIS_R2)):
            label, purpose = TRIGGER_LABELS[key]
            triggers[key] = {"value": round(self._trigger_value(axis), 3), "label": label, "purpose": purpose}

        return {
            "type": "gamepad_state",
            "connected": True,
            "name": self._name,
            "control_mode": control_mode.mode,
            "locomotion_mode": locomotion_mode.mode,
            "axes": {
                "lx": {"value": round(lx, 3), "label": "LX", "purpose": "Поворот"},
                "ly": {"value": round(ly, 3), "label": "LY", "purpose": "Вперёд / назад (в полёте — взмах)"},
                "rx": {"value": round(rx, 3), "label": "RX", "purpose": "—"},
                "ry": {"value": round(ry, 3), "label": "RY", "purpose": "—"},
            },
            "buttons": buttons,
            "triggers": triggers,
        }

    async def _emit_state(self, state: Dict[str, Any]):
        if self._on_state is None:
            return
        try:
            result = self._on_state(state)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug(f"[gamepad] on_state error: {e}")

    def _turn(self, turn: float, now: float):
        angle = turn * MAX_TURN_DEG
        self._cooldown_until = now + max(MIN_COOLDOWN_S, motion_primitives.turn_duration_s(angle))
        asyncio.create_task(self._handle_command({
            "type": "turn", "angle_deg": angle, "source": "gamepad",
        }))

    async def _loop(self):
        import pygame

        next_scan = 0.0
        try:
            while True:
                await asyncio.sleep(1.0 / POLL_HZ)
                pygame.event.pump()
                now = time.monotonic()

                if not self._connected:
                    if now >= next_scan:
                        next_scan = now + RECONNECT_INTERVAL_S
                        self._try_connect(pygame)
                    continue

                try:
                    if not self._joystick.get_init() or pygame.joystick.get_count() == 0:
                        await self._disconnect("устройство пропало")
                        continue
                    state = self._read_raw()
                except pygame.error as e:
                    await self._disconnect(str(e))
                    continue

                if now - self._last_state_ts >= 1.0 / STATE_BROADCAST_HZ:
                    self._last_state_ts = now
                    await self._emit_state(state)

                if not control_mode.is_manual():
                    continue

                # Переключение режима локомоции — по фронту нажатия, а не по удержанию.
                for button, mode in ((BUTTON_WALK_MODE, WALK), (BUTTON_FLIGHT_MODE, FLIGHT)):
                    pressed = bool(self._joystick.get_button(button)) if self._joystick.get_numbuttons() > button else False
                    was_pressed = self._prev_pressed.get(button, False)
                    self._prev_pressed[button] = pressed
                    if pressed and not was_pressed and locomotion_mode.mode != mode:
                        asyncio.create_task(self._handle_command({
                            "type": "set_locomotion_mode", "mode": mode, "source": "gamepad",
                        }))

                forward = -state["axes"]["ly"]["value"]
                turn = state["axes"]["lx"]["value"]

                if locomotion_mode.is_flight():
                    # Полёт: стик вперёд → throttle 0…1 → плавно ускоряющийся цикл взмаха.
                    # Команда шлётся каждый тик (dead-man в WingFlapController); после
                    # отпускания один раз шлём 0, чтобы взмах доехал до нейтрали.
                    throttle = (forward - DEADZONE) / (1.0 - DEADZONE) if forward > DEADZONE else 0.0
                    if throttle > 0.0 or self._flap_active:
                        asyncio.create_task(self._handle_command({
                            "type": "flap", "throttle": throttle, "source": "gamepad",
                        }))
                        self._flap_active = throttle > 0.0
                    if throttle == 0.0 and abs(turn) > DEADZONE and now >= self._cooldown_until:
                        self._turn(turn, now)
                    continue

                self._flap_active = False
                if abs(forward) > DEADZONE and now >= self._cooldown_until:
                    distance = forward * MAX_STEP_DISTANCE_CM
                    self._cooldown_until = now + max(
                        MIN_COOLDOWN_S, motion_primitives.walk_duration_s(distance)
                    )
                    asyncio.create_task(self._handle_command({
                        "type": "walk", "distance_cm": distance, "source": "gamepad",
                    }))
                elif abs(turn) > DEADZONE and now >= self._cooldown_until:
                    self._turn(turn, now)
        except asyncio.CancelledError:
            logger.info("[gamepad] Остановлен")
        except Exception as e:
            logger.exception(f"[gamepad] Ошибка в цикле опроса: {e}")
            self._connected = False
            self._task = None