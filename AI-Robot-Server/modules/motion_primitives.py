"""Motion primitives — кинематика локомоции.

Как задавать углы
=================
1. Карта каналов (индекс = номер в массиве из 22 углов):

   0  L_WING_FLAP     1  L_WING_FOLD     2  L_WING_TILT
   3  R_WING_FLAP     4  R_WING_FOLD     5  R_WING_TILT
   6  L_LEG_TURN      7  L_LEG_HIP       8  L_LEG_SHIN      9  L_LEG_TARSUS
   10 R_LEG_TURN      11 R_LEG_HIP       12 R_LEG_SHIN      13 R_LEG_TARSUS
   14 TAIL_YAW        15 TAIL_SPREAD     16 TAIL_LIFT
   17 HEAD_YAW        18 HEAD_PITCH      19 BEAK
   20 PELVIS_YAW      21 PELVIS_PITCH   (имена 17–21 уточните под свою механику)

2. Цикличные движения (шаг, взмах) — список SINGLE_* ниже.
   Каждый элемент: {"time": мс_от_начала_цикла, "servos": [22 угла]}.
   Первый и последний кадр цикла должны совпадать с нейтралью (или друг с другом),
   иначе на стыке повторов будет рывок.

   Удобный способ править отдельные каналы без переписывания всех 22:
       pose = patch(_NEUTRAL, {2: 60, 5: 120})   # CH02=60°, CH05=120°

3. Наклон хвоста / головы / таза — те же patch() в SINGLE_TURN_TICK / TAKEOFF / LANDING
   или в отдельных анимациях character/animations/*.json.

4. Физические константы STEP_LENGTH_CM / FLAP_ADVANCE_CM / TURN_STEP_DEG —
   сколько см/градусов даёт ОДИН повтор примитива. Замерьте на роботе и поправьте.

5. Крылья (сервы взмаха CH0/CH3): режим ходьбы — 0°, режим полёта — нейтраль 90°,
   взмах 60…120° вокруг 90° (см. WING_* / FLAP_* ниже и modules/locomotion_mode.py).

Как использовать:
    frames = build_walk(distance_cm=40)
    await servos.play_motion(frames, on_frame=...)
"""
import math
from typing import Dict, List, Mapping, Optional, Sequence, Union

# === TUNE: физические константы примитивов ===
STEP_LENGTH_CM = 5.0
FLAP_ADVANCE_CM = 15.0
TURN_STEP_DEG = 15.0

CHANNEL_NAMES = {
    0: "L_WING_FLAP", 1: "L_WING_FOLD", 2: "L_WING_TILT",
    3: "R_WING_FLAP", 4: "R_WING_FOLD", 5: "R_WING_TILT",
    6: "L_LEG_TURN", 7: "L_LEG_HIP", 8: "L_LEG_SHIN", 9: "L_LEG_TARSUS",
    10: "R_LEG_TURN", 11: "R_LEG_HIP", 12: "R_LEG_SHIN", 13: "R_LEG_TARSUS",
    14: "TAIL_YAW", 15: "TAIL_SPREAD", 16: "TAIL_LIFT",
    17: "HEAD_YAW", 18: "HEAD_PITCH", 19: "BEAK",
    20: "PELVIS_YAW", 21: "PELVIS_PITCH",
}
NAME_TO_CH = {v: k for k, v in CHANNEL_NAMES.items()}

_NEUTRAL: List[int] = [90] * 22

# === TUNE: крылья ===
WING_FLAP_CHANNELS = (0, 3)   # L_WING_FLAP, R_WING_FLAP — обе сервы получают один и тот же угол
WING_WALK_DEG = 0             # режим ходьбы: крылья прижаты
FLAP_NEUTRAL_DEG = 90         # режим полёта: нейтраль (середина взмаха)
FLAP_MIN_DEG = 60             # нижняя точка взмаха
FLAP_MAX_DEG = 120            # верхняя точка взмаха
FLAP_CYCLE_MS = 600           # длительность одного взмаха в примитиве SINGLE_FLAP (fly с панели)
FLAP_CYCLE_STEPS = 12         # кадров на цикл в SINGLE_FLAP
WING_TRANSITION_MS = 600      # плавный переход крыльев между 0° (ходьба) и 90° (полёт)


def patch(
    base_or_overrides: Optional[Union[Sequence[int], Mapping[Union[int, str], int]]] = None,
    overrides: Optional[Mapping[Union[int, str], int]] = None,
) -> List[int]:
    """Копия нейтрали (или base) с точечными правками каналов.

    Вызовы:
        patch({2: 60, 5: 120})
        patch({"TAIL_YAW": 110})
        patch(current_angles, {7: 80})
        patch()  # чистая нейтраль
    """
    base: Optional[Sequence[int]] = None
    ov = overrides
    if isinstance(base_or_overrides, Mapping):
        ov = base_or_overrides
    elif base_or_overrides is not None:
        base = base_or_overrides

    angles = list(base if base is not None else _NEUTRAL)
    if not ov:
        return angles
    for key, value in ov.items():
        if isinstance(key, int):
            idx = key
        else:
            s = str(key).strip().upper()
            if s in NAME_TO_CH:
                idx = NAME_TO_CH[s]
            elif s.startswith("CH"):
                idx = int(s[2:])
            else:
                raise KeyError(f"Неизвестный канал: {key!r}")
        if not 0 <= idx < 22:
            raise IndexError(f"Канал вне 0..21: {idx}")
        angles[idx] = int(value)
    return angles


# === TUNE: один цикл шага (заглушка по данным пользователя) ===
# Пользователь задал цикл «правая нога → левая нога» через CH02 / CH05.
# В карте README CH02 = L_WING_TILT, CH05 = R_WING_TILT — если на железе
# эти каналы другие, просто поменяйте номера в patch() ниже.
# Времена (мс) — заглушки; подгоните под реальную скорость шага.
#
# Правая:  CH02:60 → CH05:120 → CH02:120
# Левая:   CH02:60 → CH02:90  → CH02:60
SINGLE_STEP: List[Dict] = [
    {"time": 0,   "servos": patch()},
    # --- правая нога ---
    {"time": 150, "servos": patch({2: 60})},
    {"time": 300, "servos": patch({2: 60, 5: 120})},
    {"time": 450, "servos": patch({2: 120, 5: 120})},
    # --- левая нога ---
    {"time": 600, "servos": patch({2: 60})},
    {"time": 750, "servos": patch({2: 90})},
    {"time": 900, "servos": patch({2: 60})},
    {"time": 1000, "servos": patch()},
]

def flap_angle(phase: float) -> int:
    """Угол крыла для фазы взмаха (радианы): 0 → 90°, π/2 → 120°, 3π/2 → 60°.

    Синусоида: на границах цикла скорость серво плавно переходит в ноль у
    пиков, а на нейтрали (фаза 0 / 2π) угол ровно 90° — поэтому циклы
    стыкуются без рывка, а частоту можно менять на лету (см. modules/wing_flap.py).
    """
    s = math.sin(phase)
    amplitude = (FLAP_MAX_DEG - FLAP_NEUTRAL_DEG) if s >= 0 else (FLAP_NEUTRAL_DEG - FLAP_MIN_DEG)
    return int(round(FLAP_NEUTRAL_DEG + amplitude * s))


def build_flap_cycle(period_ms: int = FLAP_CYCLE_MS, steps: int = FLAP_CYCLE_STEPS) -> List[Dict]:
    """Один цикл взмаха 90 → 120 → 90 → 60 → 90 (первый и последний кадр = нейтраль)."""
    frames: List[Dict] = []
    for i in range(steps + 1):
        angle = flap_angle(2 * math.pi * i / steps)
        frames.append({
            "time": round(period_ms * i / steps),
            "servos": patch({ch: angle for ch in WING_FLAP_CHANNELS}),
        })
    return frames


# === TUNE: один взмах в полёте (60…120°, нейтраль 90°) ===
SINGLE_FLAP: List[Dict] = build_flap_cycle()

# === TUNE: один тик поворота (хвост) ===
# TURN_STEP_DEG ≈ насколько градусов корпуса даёт этот тик.
SINGLE_TURN_TICK: List[Dict] = [
    {"time": 0,   "servos": patch()},
    {"time": 200, "servos": patch({"TAIL_YAW": 110})},
    {"time": 400, "servos": patch()},
]

# === TUNE: взлёт / посадка (одноразовые, не циклы) ===
TAKEOFF: List[Dict] = [
    {"time": 0,   "servos": patch()},
    {"time": 150, "servos": patch({7: 70, 11: 70})},
    {"time": 300, "servos": patch({7: 110, 11: 110, 0: 50, 3: 50})},
    {"time": 450, "servos": patch({7: 90, 11: 90, 0: 130, 3: 130})},
    {"time": 600, "servos": patch({0: 50, 3: 50})},
    {"time": 750, "servos": patch()},
]

LANDING: List[Dict] = [
    {"time": 0,   "servos": patch({"TAIL_LIFT": 70, 0: 100, 3: 100})},
    {"time": 250, "servos": patch({7: 70, 11: 70, 8: 70, 12: 70})},
    {"time": 500, "servos": patch({7: 90, 11: 90})},
    {"time": 700, "servos": patch()},
]


def _repeat_frames(single_cycle: List[Dict], count: int) -> List[Dict]:
    """Склеивает N повторов цикла со сквозным time. count <= 0 → []."""
    if count <= 0:
        return []
    frames: List[Dict] = []
    cycle_duration = single_cycle[-1]["time"]
    offset = 0
    for rep in range(count):
        for i, kf in enumerate(single_cycle):
            if rep > 0 and i == 0:
                continue
            frames.append({"time": offset + kf["time"], "servos": list(kf["servos"])})
        offset += cycle_duration
    return frames


def wing_baseline_deg(mode: str) -> int:
    """Угол серв взмаха в покое для режима локомоции: walk → 0°, flight → 90°."""
    return WING_WALK_DEG if mode == "walk" else FLAP_NEUTRAL_DEG


def apply_wing_baseline(frames: List[Dict], wing_deg: int) -> List[Dict]:
    """Фиксирует сервы взмаха на wing_deg во ВСЕХ кадрах (in-place, возвращает frames).

    Нужно, потому что шаги/повороты собраны из patch() с нейтралью 90° на всех
    каналах — без этого каждый кадр ходьбы вытаскивал бы крылья из 0° в 90°.
    """
    for kf in frames:
        for ch in WING_FLAP_CHANNELS:
            kf["servos"][ch] = int(wing_deg)
    return frames


def wings_at(angles: Sequence[int], wing_deg: int, tolerance: int = 1) -> bool:
    """True, если обе сервы взмаха уже в wing_deg (с допуском)."""
    return all(abs(angles[ch] - wing_deg) <= tolerance for ch in WING_FLAP_CHANNELS)


def build_wing_transition(
    current_angles: Sequence[int],
    wing_deg: int,
    duration_ms: int = WING_TRANSITION_MS,
    steps: int = 12,
) -> List[Dict]:
    """Плавный (smoothstep) переход ТОЛЬКО сервов взмаха к wing_deg; остальные
    каналы остаются как есть. Первый кадр = текущая поза, поэтому лид-ин
    ServoController не добавляет рывка."""
    start = list(current_angles)
    frames: List[Dict] = []
    for i in range(steps + 1):
        t = i / steps
        eased = t * t * (3 - 2 * t)
        pose = list(start)
        for ch in WING_FLAP_CHANNELS:
            pose[ch] = int(round(start[ch] + (wing_deg - start[ch]) * eased))
        frames.append({"time": round(duration_ms * i / steps), "servos": pose})
    return frames


def build_walk(distance_cm: float = 0.0, steps: Optional[int] = None) -> List[Dict]:
    if steps is None:
        steps = round(abs(distance_cm) / STEP_LENGTH_CM)
    return _repeat_frames(SINGLE_STEP, max(0, int(steps)))


def build_flight_forward(distance_cm: float) -> List[Dict]:
    flaps = round(abs(distance_cm) / FLAP_ADVANCE_CM)
    return _repeat_frames(SINGLE_FLAP, flaps)


def build_turn(angle_deg: float) -> List[Dict]:
    ticks = round(abs(angle_deg) / TURN_STEP_DEG)
    frames = _repeat_frames(SINGLE_TURN_TICK, ticks)
    if angle_deg < 0 and frames:
        for kf in frames:
            yaw = kf["servos"][14]
            kf["servos"][14] = 180 - yaw
    return frames


def build_takeoff() -> List[Dict]:
    return [{"time": kf["time"], "servos": list(kf["servos"])} for kf in TAKEOFF]


def build_landing() -> List[Dict]:
    return [{"time": kf["time"], "servos": list(kf["servos"])} for kf in LANDING]


def walk_duration_s(distance_cm: float) -> float:
    steps = round(abs(distance_cm) / STEP_LENGTH_CM)
    return 0.0 if steps <= 0 else steps * (SINGLE_STEP[-1]["time"] / 1000.0)


def turn_duration_s(angle_deg: float) -> float:
    ticks = round(abs(angle_deg) / TURN_STEP_DEG)
    return 0.0 if ticks <= 0 else ticks * (SINGLE_TURN_TICK[-1]["time"] / 1000.0)


def takeoff_duration_s() -> float:
    return TAKEOFF[-1]["time"] / 1000.0


def landing_duration_s() -> float:
    return LANDING[-1]["time"] / 1000.0


def flap_duration_s(distance_cm: float) -> float:
    flaps = round(abs(distance_cm) / FLAP_ADVANCE_CM)
    return 0.0 if flaps <= 0 else flaps * (SINGLE_FLAP[-1]["time"] / 1000.0)
