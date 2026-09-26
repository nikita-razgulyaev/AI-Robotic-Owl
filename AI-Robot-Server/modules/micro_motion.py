"""Джиттер-слой ("дыхание") — пункт 3 архитектуры.

Не анимация и не решение арбитра — это аддитивный шум поверх уже
установленной позы, который просто не даёт роботу выглядеть "мёртвым" в
статике. Работает ТОЛЬКО когда сервы реально простаивают (не идёт анимация,
не активен геймпад/панель/environment-реакция) — то есть арбитр свободен
для INDEPENDENT. Никогда не перебивает и не резервирует арбитра сам — если
в любой момент арбитр занят чем-то, джиттер просто не шлёт кадр в этот тик.

Специально синусоида, а не случайное блуждание: гарантированно не уводит
позу от базовой линии со временем (в отличие от random walk, который рано
или поздно накопит заметный дрейф).
"""
import asyncio
import logging
import math
import time
from typing import Callable, Dict, Optional

from modules.arbiter import arbiter, Priority

logger = logging.getLogger(__name__)

TICK_HZ = 8  # частота обновления джиттера — заметно ниже частоты серво-цикла,
             # микродвижение не обязано быть таким же частым, как управляющие кадры


class MicroJitter:
    def __init__(
        self,
        get_current_angles: Callable[[], list],
        on_frame: Optional[Callable] = None,
        servo_profiles: Optional[Dict[int, tuple]] = None,
    ):
        """
        get_current_angles — обычно servos.get_current_angles (не мутируется,
            только читается как база для offset'а).
        on_frame — коллбэк отправки кадра (обычно RobotBrain.on_servo_frame).
            ВАЖНО: джиттер намеренно шлёт кадр напрямую через on_frame, а не
            через servos.set_all_servos — не хотим, чтобы шумовые offset'ы
            записывались в current_angles как "настоящая" целевая поза.
        servo_profiles — {индекс_серво: (амплитуда_град, период_сек)}. Каждый
            серво колеблется по своей синусоиде с собственной фазой (иначе
            вся поза будет дышать синхронно, что выглядит механически, а не
            живо). Дефолт ниже — ЗАГЛУШКА под голову/хвост, подберите индексы
            под свою раскладку каналов PCA9685.
        """
        self._get_current_angles = get_current_angles
        self._on_frame = on_frame
        self._task: Optional[asyncio.Task] = None
        self._base_angles: Optional[list] = None
        self._phases: Dict[int, float] = {}

        # TUNE: подставьте реальные индексы серв (голова/хвост/крылья на
        # позе покоя) и желаемую амплитуду в градусах. Амплитуда 1-2° —
        # заметно на глаз, но не выглядит как тик/дефект серво.
        self._profiles = servo_profiles or {
            # индекс: (амплитуда_град, период_сек)
            2: (1.5, 4.0),   # TUNE: голова, тангаж — пример
            16: (1.0, 6.0),  # TUNE: голова, крен — пример
        }
        for idx in self._profiles:
            self._phases[idx] = idx * 0.7  # разная стартовая фаза на канал, чтобы не дышали синхронно

    def start(self):
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("[jitter] Джиттер-слой запущен")

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self):
        was_idle = False
        try:
            while True:
                await asyncio.sleep(1.0 / TICK_HZ)

                if not arbiter.can_preempt(Priority.INDEPENDENT):
                    was_idle = False  # арбитр занят — сбрасываем базу, чтобы
                                       # при следующем простое не "доехать" рывком
                    continue

                if not was_idle:
                    # Только что освободились — фиксируем базовую позу один раз,
                    # а не на каждом тике (иначе синусоида будет гоняться за
                    # шумом самого get_current_angles())
                    self._base_angles = list(self._get_current_angles())
                    was_idle = True

                if self._base_angles is None or self._on_frame is None:
                    continue

                t = time.monotonic()
                frame = list(self._base_angles)
                for idx, (amplitude, period) in self._profiles.items():
                    if idx >= len(frame):
                        continue
                    phase = self._phases[idx]
                    offset = amplitude * math.sin(2 * math.pi * t / period + phase)
                    frame[idx] = max(0, min(180, int(round(frame[idx] + offset))))

                await self._on_frame(frame)
        except asyncio.CancelledError:
            logger.info("[jitter] Джиттер-слой остановлен")
            raise
