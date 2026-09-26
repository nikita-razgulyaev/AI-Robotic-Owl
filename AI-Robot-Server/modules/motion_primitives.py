"""Motion primitives — пункт 9 архитектуры.

Кинематика ОДНОГО примитива (шаг/взмах/поворот/касание посадки) — то, что вы
будете отлаживать вручную по месту (значения ниже — ЗАГЛУШКИ с нейтральной
позой 90°, чтобы модуль был рабочим "из коробки", смысла в конкретных углах
пока нет). Работа этого модуля — не кинематика, а формула: сколько раз
повторить примитив, чтобы пройти нужное расстояние/повернуть на нужный угол,
и нарезка результата в тот же формат кадров, что понимает
ServoController.play_frames() (см. правку servo_controller.py).

Как использовать после ручной настройки кинематики:
    frames = build_walk(distance_cm=40)
    await servos.play_motion(frames, on_frame=robot_brain.on_servo_frame)

Формат STEP_TABLE — намеренно одни и те же поля для шага/взмаха/поворота
(single_frames, physical_unit_cm_or_deg) — единообразие, чтобы дальше можно
было добавлять новые примитивы (взлёт/приземление — см. ниже) не меняя код
build_*(), только данные.
"""
from typing import Dict, List

# === TUNE: физические константы примитивов ===
# Сколько см/градусов проходит робот за ОДИН повтор примитива. Подбирается
# вручную по факту (например: "5 см = 1 шаг", как в исходной постановке
# задачи) — сейчас заглушка, замерьте на реальном роботе и поправьте.
STEP_LENGTH_CM = 5.0
FLAP_ADVANCE_CM = 15.0   # на сколько см вперёд продвигает один взмах в полёте (грубая оценка)
TURN_STEP_DEG = 15.0     # на сколько градусов поворачивает хвост за один "тик" поворота

# === TUNE: один цикл каждого примитива, миллисекунды + углы 18 серв ===
# Все — нейтральная поза-заглушка (90° везде). Замените средние кадры на
# реальные позы конечностей/крыльев/хвоста под свою механику; НАЧАЛЬНЫЙ И
# КОНЕЧНЫЙ кадр каждого примитива должны совпадать с нейтральной позой —
# иначе повторы примитива будут "прыгать" на стыках.
_NEUTRAL = [90] * 18

SINGLE_STEP: List[Dict] = [
    {"time": 0,   "servos": list(_NEUTRAL)},
    {"time": 250, "servos": list(_NEUTRAL)},  # TUNE: фаза переноса лапы
    {"time": 500, "servos": list(_NEUTRAL)},  # обратно в нейтраль — конец цикла
]

SINGLE_FLAP: List[Dict] = [
    {"time": 0,   "servos": list(_NEUTRAL)},
    {"time": 200, "servos": list(_NEUTRAL)},  # TUNE: крылья вверх
    {"time": 400, "servos": list(_NEUTRAL)},  # TUNE: крылья вниз (рабочий ход)
    {"time": 600, "servos": list(_NEUTRAL)},  # обратно в нейтраль
]

SINGLE_TURN_TICK: List[Dict] = [
    {"time": 0,   "servos": list(_NEUTRAL)},
    {"time": 200, "servos": list(_NEUTRAL)},  # TUNE: хвост отклонён на TURN_STEP_DEG
    {"time": 400, "servos": list(_NEUTRAL)},  # хвост обратно в нейтраль
]

TAKEOFF: List[Dict] = [
    {"time": 0,   "servos": list(_NEUTRAL)},
    {"time": 150, "servos": list(_NEUTRAL)},  # TUNE: присед лапами перед толчком
    {"time": 300, "servos": list(_NEUTRAL)},  # TUNE: толчок лапами + начало быстрых взмахов
    {"time": 450, "servos": list(_NEUTRAL)},  # TUNE: лапы поджаты, взмахи в полном темпе
]

LANDING: List[Dict] = [
    {"time": 0,   "servos": list(_NEUTRAL)},  # TUNE: наклон крыльев/хвоста для торможения
    {"time": 250, "servos": list(_NEUTRAL)},  # TUNE: лапы выставлены вперёд под посадку
    {"time": 500, "servos": list(_NEUTRAL)},  # TUNE: касание, вес на лапах
    {"time": 700, "servos": list(_NEUTRAL)},  # устойчивая нейтральная поза стояния
]


def _repeat_frames(single_cycle: List[Dict], count: int) -> List[Dict]:
    """Склеивает N повторов одного цикла примитива в единый список кадров
    со сквозным нарастающим time. count <= 0 -> пустой список (без повторов
    просто не отправляем ничего, вызывающий код может это игнорировать)."""
    if count <= 0:
        return []

    frames: List[Dict] = []
    cycle_duration = single_cycle[-1]["time"]
    offset = 0
    for rep in range(count):
        for i, kf in enumerate(single_cycle):
            # На стыке повторов не дублируем совпадающий кадр (последний кадр
            # цикла N == первый кадр цикла N+1, это одна и та же нейтральная
            # поза) — иначе на стыке будет секундная пауза "на месте".
            if rep > 0 and i == 0:
                continue
            frames.append({"time": offset + kf["time"], "servos": kf["servos"]})
        offset += cycle_duration
    return frames


def build_walk(distance_cm: float) -> List[Dict]:
    """Ходьба на заданное расстояние. Отрицательная дистанция — назад,
    если ваша кинематика SINGLE_STEP симметрична (иначе заведите отдельный
    SINGLE_STEP_BACK и переключайте здесь по знаку)."""
    steps = round(abs(distance_cm) / STEP_LENGTH_CM)
    return _repeat_frames(SINGLE_STEP, steps)


def build_flight_forward(distance_cm: float) -> List[Dict]:
    """Взмахи для горизонтального перемещения в полёте на заданное расстояние."""
    flaps = round(abs(distance_cm) / FLAP_ADVANCE_CM)
    return _repeat_frames(SINGLE_FLAP, flaps)


def build_turn(angle_deg: float) -> List[Dict]:
    """Поворот хвостом на заданный угол (по модулю — знак угла в этой
    заглушке не используется, т.к. кинематика SINGLE_TURN_TICK ещё не
    задана; при ручной настройке разведите лево/право по знаку angle_deg,
    как для build_walk с направлением)."""
    ticks = round(abs(angle_deg) / TURN_STEP_DEG)
    return _repeat_frames(SINGLE_TURN_TICK, ticks)


def build_takeoff() -> List[Dict]:
    return list(TAKEOFF)


def build_landing() -> List[Dict]:
    return list(LANDING)


def walk_duration_s(distance_cm: float) -> float:
    """Длительность build_walk(distance_cm) в секундах — без сборки самих
    кадров, только по формуле (пункт 6: чтобы кулдаун геймпада не был
    захардкоженной константой, оторванной от реальных таймингов SINGLE_STEP)."""
    steps = round(abs(distance_cm) / STEP_LENGTH_CM)
    return 0.0 if steps <= 0 else steps * (SINGLE_STEP[-1]["time"] / 1000.0)


def turn_duration_s(angle_deg: float) -> float:
    ticks = round(abs(angle_deg) / TURN_STEP_DEG)
    return 0.0 if ticks <= 0 else ticks * (SINGLE_TURN_TICK[-1]["time"] / 1000.0)


def takeoff_duration_s() -> float:
    return TAKEOFF[-1]["time"] / 1000.0


def landing_duration_s() -> float:
    return LANDING[-1]["time"] / 1000.0
