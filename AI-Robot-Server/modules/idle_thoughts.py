"""Independent stream — фоновые idle-паттерны, не завязанные на окружение.

Раз в случайный интервал предлагает арбитру сыграть одну из idle-анимаций
(взгляд по сторонам, лёгкое покачивание и т.п.) — БЕЗ обращения к LLM,
чистый выбор из готовой библиотеки animation_book. Если арбитр сейчас занят
более приоритетным источником (диалог, геймпад, реакция на лицо/звук) —
предложение просто отклоняется, idle-поток ждёт следующего тика.

Список idle-анимаций задаётся в config.settings.IDLE_ANIMATIONS — сюда
специально не подставляется весь animation_book целиком: не любая
анимация в character/animations/ обязательно уместна как фоновая (среди
них могут быть демонстрационные/тестовые, специфичные под команду и т.п.),
это осознанный выбор автора набора.
"""
import asyncio
import logging
import random
from typing import Callable, List, Optional

from modules.arbiter import arbiter, Priority
from modules.animation_loader import animation_book

logger = logging.getLogger(__name__)

# Сколько последних анимаций помнить, чтобы не повторять подряд
ANTI_REPEAT_WINDOW = 3


class IdleThoughts:
    """Фоновая задача independent stream. Запускается через start(),
    останавливается через stop() (например, при выключении сервера)."""

    def __init__(
        self,
        play_animation: Callable,
        on_frame: Optional[Callable] = None,
        min_interval_s: float = 8.0,
        max_interval_s: float = 25.0,
        animation_names: Optional[List[str]] = None,
    ):
        """
        play_animation — обычно servos.play_animation (async-метод ServoController);
            передаётся явно, а не импортируется как синглтон, чтобы модуль
            было легко тестировать без реального ServoController.
        on_frame — коллбэк отправки кадра на ESP32/панель (обычно
            robot_brain.on_servo_frame), пробрасывается в play_animation.
        animation_names — список имён анимаций-кандидатов для фона. Если не
            передан — берётся config.settings.IDLE_ANIMATIONS; если и его нет —
            idle-поток стартует, но ничего не предлагает (см. предупреждение в логе).
        """
        self._play_animation = play_animation
        self._on_frame = on_frame
        self._min_interval = min_interval_s
        self._max_interval = max_interval_s
        self._task: Optional[asyncio.Task] = None
        self._recent: List[str] = []

        if animation_names is None:
            try:
                from config.settings import IDLE_ANIMATIONS
                animation_names = list(IDLE_ANIMATIONS)
            except ImportError:
                animation_names = []
        self._candidates = animation_names

        if not self._candidates:
            logger.warning(
                "[idle] Список IDLE_ANIMATIONS пуст — фоновый поток запущен, "
                "но предлагать нечего. Добавьте имена анимаций в config.settings.IDLE_ANIMATIONS."
            )

    def start(self):
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("[idle] Independent stream запущен")

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _pick_next(self) -> Optional[str]:
        """Взвешенный выбор без повтора последних ANTI_REPEAT_WINDOW имён.
        Если кандидатов меньше, чем окно антиповтора — исключаем всё равно
        по максимуму возможного (иначе список схлопнется в пустой)."""
        if not self._candidates:
            return None
        pool = [name for name in self._candidates if name not in self._recent]
        if not pool:
            pool = list(self._candidates)
        choice = random.choice(pool)
        self._recent.append(choice)
        self._recent = self._recent[-ANTI_REPEAT_WINDOW:]
        return choice

    async def _loop(self):
        try:
            while True:
                await asyncio.sleep(random.uniform(self._min_interval, self._max_interval))

                if not arbiter.can_preempt(Priority.INDEPENDENT):
                    continue  # арбитр занят кем-то приоритетнее — тихо ждём следующий тик

                name = self._pick_next()
                if name is None:
                    continue

                frames = animation_book.get_frames(name)
                if not frames:
                    logger.warning(f"[idle] Анимация '{name}' из IDLE_ANIMATIONS не найдена в animation_book")
                    continue
                duration_s = frames[-1]["time"] / 1000.0

                accepted = await arbiter.request(Priority.INDEPENDENT, duration_s, label=f"idle:{name}")
                if not accepted:
                    continue  # арбитр забрал кто-то приоритетнее между can_preempt() и request()

                logger.debug(f"[idle] Фоновая анимация: {name}")
                asyncio.create_task(self._play_animation(name, on_frame=self._on_frame))
        except asyncio.CancelledError:
            logger.info("[idle] Independent stream остановлен")
            raise
