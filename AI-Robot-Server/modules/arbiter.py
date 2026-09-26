"""Приоритетный арбитр источников команд на сервоприводы.

Идея (см. обсуждение архитектуры): "живость" робота собирается из нескольких
независимых источников, которые хотят одновременно распоряжаться сервами —
фоновые idle-паттерны (independent stream), реакции на окружение (лицо/звук),
геймпад, веб-панель. Не нужно два "мышления" — нужен один цикл выбора,
кто прямо сейчас имеет право отправить кадр на сервы.

Арбитр НЕ исполняет анимации/движения сам — это по-прежнему
ServoController.play_animation() / motion_primitives (см. следующий модуль).
Арбитр только решает "можно" или "нет", резервируя себя на время действия
конкретного источника.

Правило: источник с более высоким приоритетом (меньшее число в Priority)
может перебить/заблокировать источник с более низким приоритетом. Источник
того же или более низкого приоритета, что и текущий активный держатель,
пока тот не истёк — отклоняется.
"""
import asyncio
import logging
import time
from collections import deque
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Меньше число — выше приоритет."""
    SAFETY = 0        # аппаратные рефлексы/аварийная остановка (обычно с прошивки,
                       # но зарезервировано и здесь на случай серверной safety-логики)
    PANEL = 1          # явная команда с веб-панели — прямое намерение человека
    GAMEPAD = 2        # геймпад — тоже прямое намерение человека, но ниже панели
    ENVIRONMENT = 3    # реакция на окружение (лицо в кадре, звук, распознанная фраза)
    INDEPENDENT = 4    # фоновые idle-паттерны — низший приоритет, перебивается всем


class Arbiter:
    """Потокобезопасный (в рамках одного event loop) арбитр приоритетов.

    Использование:
        accepted = await arbiter.request(
            Priority.ENVIRONMENT, duration_s=1.2, label="face_seen",
        )
        if accepted:
            asyncio.create_task(servos.play_animation("look_up", on_frame=...))

    request() только резервирует "окно" арбитра — саму анимацию/примитив
    запускает вызывающий код отдельно (через play_animation или
    motion_primitives), чтобы арбитр не знал деталей того, ЧТО исполняется.
    """

    def __init__(self):
        self._active_priority: Priority = Priority.INDEPENDENT
        self._active_until: float = 0.0
        self._active_label: str = ""
        self._lock = asyncio.Lock()
        # Пункт 1 (метрики): (timestamp, priority, label) на каждую УСПЕШНУЮ
        # выдачу — чтобы видеть, кто как часто реально перехватывает
        # управление. Ограничено по размеру, а не только по времени — если
        # что-то начнёт спамить запросами, список не будет расти бесконечно.
        self._grant_log: deque = deque(maxlen=2000)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    async def request(self, source: Priority, duration_s: float, label: str = "") -> bool:
        """Пытается забронировать арбитра для source на duration_s секунд.

        Возвращает True, если источник получил право действовать (либо
        текущий держатель того же/ниже приоритета уже истёк, либо source
        приоритетнее текущего держателя — в этом случае source его перебивает).
        Возвращает False, если сейчас активен источник СТРОГО выше приоритетом
        и ещё не истёк — тогда вызывающий код должен промолчать/отступить.
        """
        async with self._lock:
            now = self._now()
            still_held = now < self._active_until
            if still_held and source > self._active_priority:
                logger.debug(
                    f"[arbiter] {source.name} отклонён — держит {self._active_priority.name} "
                    f"('{self._active_label}'), ещё {self._active_until - now:.1f}с"
                )
                return False

            self._active_priority = source
            self._active_until = now + max(0.0, duration_s)
            self._active_label = label
            self._grant_log.append((now, source, label))
            logger.debug(f"[arbiter] {source.name} захватил на {duration_s:.1f}с ('{label}')")
            return True

    def can_preempt(self, source: Priority) -> bool:
        """Проверка БЕЗ захвата — удобно, чтобы не тратить ресурсы (например,
        не выбирать случайную idle-анимацию), если всё равно откажут."""
        now = self._now()
        if now >= self._active_until:
            return True
        return source <= self._active_priority

    def release(self, source: Optional[Priority] = None):
        """Досрочно освобождает арбитра (например, геймпад отпустили стик).
        Если source указан и не совпадает с текущим держателем — no-op,
        чтобы случайно не сбросить чужую бронь гонкой вызовов."""
        if source is not None and source != self._active_priority:
            return
        self._active_until = 0.0
        self._active_priority = Priority.INDEPENDENT
        self._active_label = ""

    @property
    def active_label(self) -> str:
        return self._active_label if self._now() < self._active_until else ""

    def stats(self, window_s: float = 300.0) -> dict:
        """Пункт 1: сколько раз каждый приоритет реально забирал управление
        за последние window_s секунд. Пример использования — отдать через
        HTTP-эндпоинт /arbiter/stats на панели, чтобы видеть вживую, не
        забивает ли, например, GAMEPAD environment-реакции слишком часто.

        Возвращает {"SAFETY": 0, "PANEL": 3, "GAMEPAD": 41, "ENVIRONMENT": 12,
        "INDEPENDENT": 8, "window_s": 300.0}"""
        cutoff = self._now() - window_s
        counts = {p.name: 0 for p in Priority}
        for ts, source, _label in self._grant_log:
            if ts >= cutoff:
                counts[source.name] += 1
        counts["window_s"] = window_s
        return counts


# Глобальный экземпляр — один арбитр на сервер, как и остальные синглтоны
# в этом проекте (quick_answers, animation_book)
arbiter = Arbiter()
