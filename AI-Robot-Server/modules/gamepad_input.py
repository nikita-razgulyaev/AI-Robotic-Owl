"""Геймпад — пункт 11 архитектуры.

Физически подключается к ПК по Bluetooth (не к роботу — см. README, раздел
"Сеть и связь"), читается через pygame.joystick. Левый стик — ходьба/полёт
(вперёд/назад = дистанция за тик, влево/вправо = поворот), кнопки —
взлёт/посадка. Команды идут ровно через тот же RobotBrain.handle_command(),
что и панель — просто с source="gamepad", чтобы арбитр резервировался на
Priority.GAMEPAD (см. modules/arbiter.py), а не PANEL.

pygame.joystick требует периодического pygame.event.pump() — здесь это
сделано через asyncio-таск с быстрым poll'ом, а не отдельный поток: частота
опроса стика (десятки Гц) не настолько высокочастотна, чтобы блокировать
event loop заметно на каждый pump().

ВАЖНО: это скелет для одного конкретного геймпада ни разу не тестировался
на реальном железе — mapping осей/кнопок (JOY_AXIS_*, JOY_BUTTON_*) почти
наверняка придётся подстроить под конкретную модель геймпада (проверяется
tools вроде `python -m pygame.examples.joystick`).
"""
import asyncio
import logging
from typing import Optional

from modules import motion_primitives

logger = logging.getLogger(__name__)

POLL_HZ = 30

# TUNE: индексы осей/кнопок — подберите под свой геймпад
AXIS_FORWARD = 1      # левый стик, вертикаль (обычно -1 = вперёд, +1 = назад)
AXIS_TURN = 0          # левый стик, горизонталь
BUTTON_TAKEOFF = 0     # обычно "A"/"X" в зависимости от геймпада
BUTTON_LAND = 1        # обычно "B"/"O"

DEADZONE = 0.15         # игнорировать дрожание стика в нейтрали
MAX_STEP_DISTANCE_CM = 20.0   # на что домножается полное отклонение стика за один тик
MAX_TURN_DEG = 30.0
# Пункт 6: кулдаун НЕ константа — считается по факту через
# motion_primitives.*_duration_s() из реальных таймингов примитива, чтобы
# при правке SINGLE_STEP/SINGLE_TURN_TICK ничего не рассинхронизировалось.
MIN_COOLDOWN_S = 0.15    # нижняя граница на случай очень короткого/нулевого примитива —
                          # чтобы не заспамить handle_command при некорректных TUNE-значениях


class GamepadInput:
    def __init__(self, handle_command):
        """handle_command — обычно robot_brain.handle_command (async), тот же
        путь, что использует веб-панель — держим единственную точку входа
        команд на сервы, как и задумано в архитектуре (пункт 10)."""
        self._handle_command = handle_command
        self._task: Optional[asyncio.Task] = None
        self._joystick = None
        self._cooldown_until = 0.0  # monotonic-время, до которого новые команды локомоции не шлются

    def start(self):
        if self._task is not None:
            return
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                logger.warning("[gamepad] Геймпад не найден — модуль не активируется")
                return
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            logger.info(f"[gamepad] Подключен: {self._joystick.get_name()}")
        except ImportError:
            logger.warning("[gamepad] Библиотека pygame не установлена (pip install pygame) — модуль не активируется")
            return
        except Exception as e:
            logger.error(f"[gamepad] Ошибка инициализации: {e}")
            return

        self._task = asyncio.create_task(self._loop())

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self):
        import pygame
        import time

        try:
            while True:
                await asyncio.sleep(1.0 / POLL_HZ)
                pygame.event.pump()

                now = time.monotonic()
                forward = -self._joystick.get_axis(AXIS_FORWARD)  # инверсия: вверх на стике = вперёд
                turn = self._joystick.get_axis(AXIS_TURN)

                if abs(forward) > DEADZONE and now >= self._cooldown_until:
                    distance = forward * MAX_STEP_DISTANCE_CM
                    self._cooldown_until = now + max(MIN_COOLDOWN_S, motion_primitives.walk_duration_s(distance))
                    asyncio.create_task(self._handle_command({
                        "type": "walk", "distance_cm": distance, "source": "gamepad",
                    }))
                elif abs(turn) > DEADZONE and now >= self._cooldown_until:
                    angle = turn * MAX_TURN_DEG
                    self._cooldown_until = now + max(MIN_COOLDOWN_S, motion_primitives.turn_duration_s(angle))
                    asyncio.create_task(self._handle_command({
                        "type": "turn", "angle_deg": angle, "source": "gamepad",
                    }))
                if self._joystick.get_button(BUTTON_TAKEOFF) and now >= self._cooldown_until:
                    self._cooldown_until = now + motion_primitives.takeoff_duration_s()
                    asyncio.create_task(self._handle_command({"type": "takeoff", "source": "gamepad"}))
                elif self._joystick.get_button(BUTTON_LAND) and now >= self._cooldown_until:
                    self._cooldown_until = now + motion_primitives.landing_duration_s()
                    asyncio.create_task(self._handle_command({"type": "land", "source": "gamepad"}))

        except asyncio.CancelledError:
            logger.info("[gamepad] Остановлен")
            raise
        except Exception as e:
            logger.error(f"[gamepad] Ошибка в цикле опроса: {e}")
