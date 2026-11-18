"""Непрерывный цикл взмаха крыльев (режим полёта).

Левый стик вперёд → throttle 0…1 → частота взмаха растёт ПЛАВНО (ограничение
ускорения), а не скачком. Угол считается из накапливаемой фазы
(angle = flap_angle(phase)), поэтому смена частоты на лету не даёт рывков.
Отпустили стик → частота падает до минимальной, текущий взмах доводится до
нейтрали 90° и цикл останавливается.

Безопасность:
  - dead-man: если set_throttle() не вызывали DEADMAN_S секунд (геймпад пропал,
    сервер завис) — взмах сам затухает;
  - арбитр: каждый тик продлевает бронь GAMEPAD; если держатель приоритетнее
    (SAFETY/PANEL) — цикл прекращается;
  - пока идёт взмах, ServoController.is_animating = True и задача зарегистрирована
    как активная, поэтому emergency_stop() / cancel_current() его останавливают.

Двигает только сервы взмаха (motion_primitives.WING_FLAP_CHANNELS); остальные
каналы остаются как есть.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Awaitable, Callable, List, Optional

from modules import motion_primitives
from modules.arbiter import Priority, arbiter

logger = logging.getLogger(__name__)

# === TUNE ===
TICK_HZ = 40               # частота обновления серв
MIN_FLAP_HZ = 0.8          # взмахов/с при минимальном отклонении стика
MAX_FLAP_HZ = 2.0          # взмахов/с при стике до упора (SG90: ~470°/с на пике при 2.5 Гц — не форсируйте)
ACCEL_HZ_PER_S = 0.6       # насколько быстро растёт частота (Гц за секунду)
DECEL_HZ_PER_S = 1.5       # насколько быстро падает частота
DEADMAN_S = 0.3            # нет обновлений throttle дольше — считаем стик отпущенным
THROTTLE_EPS = 0.01        # throttle ниже — «стик отпущен»

TWO_PI = 2 * math.pi

FrameCallback = Callable[[List[int]], Awaitable[None]]


class WingFlapController:
    def __init__(self, servos, get_on_frame: Callable[[], Optional[FrameCallback]]):
        """get_on_frame — функция, возвращающая актуальный on_frame (в RobotBrain он
        назначается позже, из websocket_server, поэтому берём лениво)."""
        self._servos = servos
        self._get_on_frame = get_on_frame
        self._task: Optional[asyncio.Task] = None
        self._throttle = 0.0
        self._throttle_ts = 0.0
        self.frequency_hz = 0.0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def set_throttle(self, value: float) -> bool:
        """Задаёт целевую интенсивность 0…1 (обычно — отклонение стика вперёд).
        Запускает цикл при необходимости. Возвращает False, если запуск
        отклонён (сервы заняты другой анимацией)."""
        value = max(0.0, min(1.0, float(value)))
        self._throttle = value
        self._throttle_ts = time.monotonic()
        if value > THROTTLE_EPS and not self.running:
            if self._servos.is_animating:
                return False
            self._servos.is_animating = True
            self._task = asyncio.create_task(self._run())
        return True

    async def stop(self):
        """Немедленно останавливает цикл (поза остаётся как есть, без возврата в 90°)."""
        task = self._task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _effective_throttle(self) -> float:
        if time.monotonic() - self._throttle_ts > DEADMAN_S:
            return 0.0
        return self._throttle

    @staticmethod
    def target_frequency(throttle: float) -> float:
        return MIN_FLAP_HZ + (MAX_FLAP_HZ - MIN_FLAP_HZ) * throttle

    async def _emit(self, wing_deg: int):
        angles = self._servos.get_current_angles()
        if all(angles[ch] == wing_deg for ch in motion_primitives.WING_FLAP_CHANNELS):
            return
        for ch in motion_primitives.WING_FLAP_CHANNELS:
            angles[ch] = wing_deg
        self._servos.set_all_servos(angles, notify=False, quiet=True)
        on_frame = self._get_on_frame()
        if on_frame:
            await on_frame(angles)

    async def _run(self):
        self._servos._active_task = asyncio.current_task()
        dt = 1.0 / TICK_HZ
        phase = 0.0
        freq = MIN_FLAP_HZ
        stop_phase: Optional[float] = None
        last = time.monotonic()
        logger.info("🪽 Взмах: старт")
        try:
            while True:
                await asyncio.sleep(dt)
                now = time.monotonic()
                step = now - last
                last = now

                if not await arbiter.request(Priority.GAMEPAD, 0.3, label="motion:flap"):
                    logger.info("🪽 Взмах прерван: арбитр занят более приоритетным источником")
                    return

                throttle = self._effective_throttle()
                if throttle > THROTTLE_EPS:
                    stop_phase = None
                    target = self.target_frequency(throttle)
                else:
                    target = MIN_FLAP_HZ
                    if stop_phase is None:
                        stop_phase = math.ceil(phase / TWO_PI) * TWO_PI

                rate = ACCEL_HZ_PER_S if target > freq else DECEL_HZ_PER_S
                max_delta = rate * step
                freq += max(-max_delta, min(max_delta, target - freq))
                self.frequency_hz = freq

                phase += TWO_PI * freq * step

                if stop_phase is not None and phase >= stop_phase:
                    await self._emit(motion_primitives.FLAP_NEUTRAL_DEG)
                    logger.info("🪽 Взмах: стоп (нейтраль 90°)")
                    return

                await self._emit(motion_primitives.flap_angle(phase))
        finally:
            self.frequency_hz = 0.0
            if self._servos._active_task is asyncio.current_task():
                self._servos._active_task = None
            self._servos.is_animating = False
            self._task = None
