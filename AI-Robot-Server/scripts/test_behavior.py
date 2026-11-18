"""Тест логики поведения по датчикам опоры (без железа и без запущенного сервера).

Запуск из корня проекта:  python scripts/test_behavior.py

Проверяет: таблицу переходов кнопок -> события, загрузку/валидацию rules.yaml,
подтверждение по времени (дребезг не создаёт событий), отличие взлёта от падения,
запрет взлёта без опоры и выбор правила (первое подходящее, кулдаун).
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import behavior as bh  # noqa: E402
from modules.behavior import SupportState as S  # noqa: E402

failures = []


def check(name, cond, extra=""):
    print(("✅ " if cond else "❌ ") + name + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        failures.append(name)


# --- 1. состояние из кнопок ---
check("лапа -> standing", bh.state_from_buttons(True, False) is S.STANDING)
check("крыло -> held", bh.state_from_buttons(False, True) is S.HELD)
check("обе -> held_touch", bh.state_from_buttons(True, True) is S.HELD_TOUCH)
check("никакая -> none", bh.state_from_buttons(False, False) is S.NONE)

# --- 2. события из переходов ---
cases = [
    (S.STANDING, S.NONE, "support_lost"),
    (S.HELD, S.NONE, "released"),
    (S.HELD_TOUCH, S.NONE, "released"),
    (S.NONE, S.STANDING, "touchdown"),
    (S.NONE, S.HELD_TOUCH, "touchdown"),
    (S.NONE, S.HELD, "grabbed"),
    (S.STANDING, S.HELD, "grabbed"),
    (S.STANDING, S.HELD_TOUCH, "grabbed"),
    (S.HELD, S.STANDING, "support_regained"),
    (S.HELD_TOUCH, S.STANDING, "support_regained"),
    (S.HELD, S.HELD_TOUCH, None),
    (S.HELD_TOUCH, S.HELD, None),
    (S.UNKNOWN, S.STANDING, None),
    (S.STANDING, S.STANDING, None),
]
for prev, new, want in cases:
    check(f"{prev.value} -> {new.value} = {want}", bh.derive_event(prev, new) == want)

# --- 3. загрузка правил ---
rules = bh.load_rules(bh.BEHAVIOR_RULES_FILE)
check("rules.yaml из проекта загружается", len(rules) >= 5, f"загружено {len(rules)}")

bad_yaml = """
rules:
  - id: ok
    when: {event: grabbed}
    do: {anim: x}
  - id: bad_event
    when: {event: exploded}
    do: {anim: x}
  - id: bad_state
    when: {event: grabbed, prev_state: [flying]}
    do: {anim: x}
  - id: bad_priority
    when: {event: grabbed}
    do: {priority: URGENT}
  - id: ok
    when: {event: released}
    do: {anim: y}
"""
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
    f.write(bad_yaml)
loaded = bh.load_rules(f.name)
check("некорректные правила пропускаются, валидное остаётся", [r["id"] for r in loaded] == ["ok"], str([r["id"] for r in loaded]))
check("несуществующий файл -> пустой список", bh.load_rules("/no/such/file.yaml") == [])


# --- 4. движок с подменой серв ---
class FakeServos:
    def __init__(self):
        self.played = []
        self.cancelled = 0

    async def cancel_current(self):
        self.cancelled += 1

    async def play_animation(self, name, on_frame=None):
        self.played.append(name)


async def scenario():
    from modules.animation_loader import animation_book, AnimationInfo
    from modules.arbiter import arbiter

    # фиктивные анимации, чтобы правила имели что проигрывать
    frames = [{"time": 0, "servos": [90] * 22}, {"time": 100, "servos": [90] * 22}]
    for name in ("fall_react", "grabbed_react", "recover_stand", "settle_after_put"):
        animation_book._animations[name] = AnimationInfo(name=name, frames=frames, source="custom")

    eng = bh.BehaviorEngine()
    eng._rules = bh.load_rules(bh.BEHAVIOR_RULES_FILE)
    servos = FakeServos()
    phrases, emotions = [], []
    eng.attach(servos, on_frame=None, get_emotion=lambda: "calm",
               set_emotion=emotions.append, play_phrase=lambda p: _async_append(phrases, p))

    async def settle():
        await asyncio.sleep((bh.BEHAVIOR_CONFIRM_NONE_MS + 60) / 1000)

    # стартовое состояние: стоит
    eng.on_sensor(True, False, initial=True)
    check("начальное состояние standing", eng.state is S.STANDING)

    # дребезг: опора пропала и вернулась быстрее подтверждения — события нет
    eng.on_sensor(False, False)
    await asyncio.sleep(0.02)
    eng.on_sensor(True, False)
    await settle()
    check("дребезг не создаёт события", eng.state is S.STANDING and eng.last_event is None and not servos.played)

    # падение: опора пропала надолго, взлёт не запускали
    arbiter.release()
    eng.on_sensor(False, False)
    await settle()
    check("падение: событие support_lost", eng.last_event == "support_lost")
    check("падение: играется fall_react", servos.played[-1:] == ["fall_react"], str(servos.played))
    check("падение: прервано текущее движение", servos.cancelled >= 1)
    check("падение: эмоция alert", emotions[-1:] == ["alert"], str(emotions))
    await asyncio.sleep(0.05)
    check("падение: фраза behavior_fall", phrases[-1:] == ["behavior_fall"], str(phrases))

    # приземление после падения
    arbiter.release()
    eng.on_sensor(True, False)
    await settle()
    check("приземление: touchdown + recover_stand", eng.last_event == "touchdown" and servos.played[-1] == "recover_stand")

    # взлёт запрещён не со стойки, разрешён со стойки
    ok, _ = eng.can_takeoff()
    check("взлёт разрешён со стойки", ok)

    # свой взлёт: потеря опоры не считается падением
    played_before = len(servos.played)
    arbiter.release()
    eng.begin_self_flight()
    eng.on_sensor(False, False)
    await settle()
    check("свой взлёт: падение не срабатывает", eng.last_event == "support_lost" and len(servos.played) == played_before)

    # посадка после своего полёта: флаг сбрасывается касанием, правила fall/recover не срабатывают
    arbiter.release()
    eng.on_sensor(True, False)
    await settle()
    check("посадка после полёта: recover_stand не играется", len(servos.played) == played_before)
    check("после касания флаг своего полёта сброшен", not eng.self_flight_active)

    # поймали руками, потом отпустили
    arbiter.release()
    eng.on_sensor(False, True)  # standing -> held
    await settle()
    check("взяли с опоры: grabbed_react", eng.last_event == "grabbed" and servos.played[-1] == "grabbed_react")
    ok, reason = eng.can_takeoff()
    check("взлёт запрещён, пока держат", (not ok) and "held" in reason, reason)

    arbiter.release()
    await asyncio.sleep(0.01)
    eng.on_sensor(False, False)  # held -> none
    await settle()
    check("отпустили в воздухе: released", eng.last_event == "released" and servos.played[-1] == "fall_react")

    # плата отключилась
    eng.reset()
    check("reset -> unknown, взлёт не блокируется", eng.state is S.UNKNOWN and eng.can_takeoff()[0])


async def _async_append(lst, item):
    lst.append(item)


asyncio.run(scenario())

print()
if failures:
    print(f"❌ Провалено проверок: {len(failures)}")
    sys.exit(1)
print("✅ Все проверки пройдены")
