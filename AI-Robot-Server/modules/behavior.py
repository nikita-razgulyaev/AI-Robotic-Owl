"""Поведение по датчикам опоры: кнопка в лапе + кнопка в крыле -> состояние
опоры -> событие -> правило из character/behavior/rules.yaml -> реакция.

Что здесь есть:
  1. Состояние опоры из двух кнопок (лапа = "стоит", крыло = "держат").
  2. События, выводимые из смены состояния (support_lost, grabbed, released,
     touchdown, support_regained) с подтверждением по времени (защита от дребезга).
  3. Таблица правил (YAML, правится руками, hot-reload): "событие + состояние ->
     анимация + эмоция + фраза". Первое подходящее правило побеждает.
  4. Флаг "сам взлетел" — чтобы отличать взлёт от падения: потеря опоры сразу
     после команды взлёта это не падение.

Чего здесь НЕТ (намеренно): рефлексов на самой плате. ESP32 только читает кнопки
и шлёт {"type":"sensor","paw":bool,"wing":bool}; все решения принимает сервер.
Без IMU/барометра нельзя определить ориентацию/высоту — это события на будущее.

Реакция идёт через арбитр (modules/arbiter.py): "падение" бронирует Priority.SAFETY
и может перебить любую текущую анимацию; остальное — по приоритету из правила.
Если анимации из правила ещё нет в animation_book — выполняются остальные части
правила (эмоция/фраза), в лог пишется предупреждение один раз.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import yaml

from config.settings import (
    BEHAVIOR_ENABLED,
    BEHAVIOR_RULES_FILE,
    BEHAVIOR_CONFIRM_MS,
    BEHAVIOR_CONFIRM_NONE_MS,
    BEHAVIOR_SELF_FLIGHT_TIMEOUT_SEC,
)
from modules.arbiter import arbiter, Priority
from modules.animation_loader import animation_book

logger = logging.getLogger(__name__)


class SupportState(str, Enum):
    UNKNOWN = "unknown"
    STANDING = "standing"
    HELD = "held"
    HELD_TOUCH = "held_touch"
    NONE = "none"


EVENTS = ("support_lost", "support_regained", "grabbed", "released", "touchdown")

_STATE_NAMES = {s.value for s in SupportState}


def state_from_buttons(paw: bool, wing: bool) -> SupportState:
    if paw and not wing:
        return SupportState.STANDING
    if wing and not paw:
        return SupportState.HELD
    if paw and wing:
        return SupportState.HELD_TOUCH
    return SupportState.NONE


def derive_event(prev: SupportState, new: SupportState) -> Optional[str]:
    """Чистая функция: какое событие означает переход prev -> new (или None)."""
    if prev is SupportState.UNKNOWN or prev is new:
        return None

    if new is SupportState.NONE:
        if prev is SupportState.STANDING:
            return "support_lost"
        return "released"

    if prev is SupportState.NONE:
        if new is SupportState.HELD:
            return "grabbed"
        return "touchdown"

    if prev is SupportState.STANDING:
        return "grabbed"

    if new is SupportState.STANDING:
        return "support_regained"

    return None


# ---------- Правила ----------

def _as_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def load_rules(path) -> List[dict]:
    """Читает и проверяет YAML с правилами. Некорректное правило пропускается
    (с ошибкой в логе), а не роняет сервер."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"[behavior] Файл правил не найден: {path} — реакции отключены")
        return []
    except Exception as e:
        logger.error(f"[behavior] Не удалось прочитать {path}: {e}")
        return []

    items = raw.get("rules") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        logger.error(f"[behavior] {path}: ожидается ключ 'rules' со списком правил")
        return []

    rules: List[dict] = []
    seen_ids = set()
    for idx, item in enumerate(items):
        ctx = f"правило #{idx}"
        if not isinstance(item, dict):
            logger.error(f"[behavior] {ctx}: не объект — пропущено")
            continue
        rule_id = str(item.get("id") or f"rule_{idx}")
        ctx = f"правило «{rule_id}»"
        if rule_id in seen_ids:
            logger.error(f"[behavior] {ctx}: повторяющийся id — пропущено")
            continue

        when = item.get("when")
        do = item.get("do")
        if not isinstance(when, dict) or not isinstance(do, dict):
            logger.error(f"[behavior] {ctx}: нужны разделы 'when' и 'do' — пропущено")
            continue
        if when.get("event") not in EVENTS:
            logger.error(f"[behavior] {ctx}: неизвестное событие {when.get('event')!r} (допустимо: {', '.join(EVENTS)}) — пропущено")
            continue

        prev_state = _as_list(when.get("prev_state"))
        state = _as_list(when.get("state"))
        bad = [s for s in (prev_state or []) + (state or []) if s not in _STATE_NAMES]
        if bad:
            logger.error(f"[behavior] {ctx}: неизвестные состояния {bad} (допустимо: {sorted(_STATE_NAMES)}) — пропущено")
            continue

        priority_name = str(do.get("priority", "ENVIRONMENT")).upper()
        if priority_name not in Priority.__members__:
            logger.error(f"[behavior] {ctx}: неизвестный priority {priority_name!r} — пропущено")
            continue

        rules.append({
            "id": rule_id,
            "when": {
                "event": when["event"],
                "prev_state": prev_state,
                "state": state,
                "self_takeoff": when.get("self_takeoff"),   # None = не важно
                "emotion": _as_list(when.get("emotion")),
            },
            "do": {
                "anim": do.get("anim"),
                "emotion": do.get("emotion"),
                "phrase": do.get("phrase"),
                "priority": priority_name,
                "cooldown_sec": float(do.get("cooldown_sec", 0)),
                "hold_sec": float(do.get("hold_sec", 1.0)),
            },
        })
        seen_ids.add(rule_id)

    return rules


def rule_matches(rule: dict, event: str, prev: SupportState, new: SupportState,
                 self_takeoff: bool, emotion: Optional[str]) -> bool:
    when = rule["when"]
    if when["event"] != event:
        return False
    if when["prev_state"] and prev.value not in when["prev_state"]:
        return False
    if when["state"] and new.value not in when["state"]:
        return False
    if when["self_takeoff"] is not None and bool(when["self_takeoff"]) != self_takeoff:
        return False
    if when["emotion"] and emotion not in when["emotion"]:
        return False
    return True


# ---------- Движок ----------

class BehaviorEngine:
    def __init__(self):
        self.enabled = BEHAVIOR_ENABLED
        self.state: SupportState = SupportState.UNKNOWN
        self.last_event: Optional[str] = None

        self._pending_state: Optional[SupportState] = None
        self._pending_task: Optional[asyncio.Task] = None
        self._self_flight_until = 0.0
        self._rules: List[dict] = []
        self._cooldown_until: Dict[str, float] = {}
        self._warned_anims = set()

        self._servos = None
        self._on_frame: Optional[Callable[[List[int]], Awaitable[None]]] = None
        self._get_emotion: Optional[Callable[[], str]] = None
        self._set_emotion: Optional[Callable[[str], None]] = None
        self._play_phrase: Optional[Callable[[str], Awaitable[None]]] = None

        self.reload_rules()

    # --- настройка ---
    def attach(self, servos, on_frame, get_emotion=None, set_emotion=None, play_phrase=None):
        self._servos = servos
        self._on_frame = on_frame
        self._get_emotion = get_emotion
        self._set_emotion = set_emotion
        self._play_phrase = play_phrase

    def reload_rules(self) -> int:
        """Перечитывает rules.yaml. Можно вызывать во время работы сервера."""
        self._rules = load_rules(BEHAVIOR_RULES_FILE)
        self._warned_anims.clear()
        logger.info(f"[behavior] Правил загружено: {len(self._rules)} ({BEHAVIOR_RULES_FILE})")
        return len(self._rules)

    def reset(self):
        """Плата отключилась — состояние опоры неизвестно."""
        self._cancel_pending()
        self.state = SupportState.UNKNOWN
        self.end_self_flight()

    # --- флаг "сам взлетел" ---
    def begin_self_flight(self):
        self._self_flight_until = time.monotonic() + BEHAVIOR_SELF_FLIGHT_TIMEOUT_SEC

    def end_self_flight(self):
        self._self_flight_until = 0.0

    @property
    def self_flight_active(self) -> bool:
        return time.monotonic() < self._self_flight_until

    def can_takeoff(self) -> Tuple[bool, str]:
        """Проверка условия «взлёт разрешён»: стоит на опоре. Если данных
        от кнопок нет (UNKNOWN) или поведение выключено — не блокируем."""
        if not self.enabled or self.state is SupportState.UNKNOWN:
            return True, ""
        if self.state is SupportState.STANDING:
            return True, ""
        return False, f"Взлёт запрещён: состояние опоры «{self.state.value}» (нужно «standing» — лапа нажата, крыло нет)"

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "support_state": self.state.value,
            "self_flight": self.self_flight_active,
            "last_event": self.last_event,
            "rules": len(self._rules),
        }

    # --- вход от датчиков ---
    def on_sensor(self, paw: bool, wing: bool, initial: bool = False):
        """Вызывается при каждом сообщении платы с состоянием кнопок.
        initial=True — первичный/восстановленный снимок (после подключения):
        принимается сразу, без событий и подтверждения."""
        if not self.enabled:
            return
        raw = state_from_buttons(paw, wing)

        if initial or self.state is SupportState.UNKNOWN:
            self._cancel_pending()
            if raw is not self.state:
                logger.info(f"[behavior] Начальное состояние опоры: {raw.value}")
            self.state = raw
            return

        if raw is self.state:
            self._cancel_pending()
            return
        if raw is self._pending_state:
            return

        self._cancel_pending()
        self._pending_state = raw
        delay_ms = BEHAVIOR_CONFIRM_NONE_MS if raw is SupportState.NONE else BEHAVIOR_CONFIRM_MS
        self._pending_task = asyncio.create_task(self._confirm(raw, delay_ms / 1000.0))

    def _cancel_pending(self):
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = None
        self._pending_state = None

    async def _confirm(self, raw: SupportState, delay_s: float):
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        if self._pending_state is not raw:
            return
        self._pending_task = None
        self._pending_state = None
        await self._commit(raw)

    async def _commit(self, new: SupportState):
        prev, self.state = self.state, new
        event = derive_event(prev, new)
        logger.info(f"[behavior] Опора: {prev.value} -> {new.value}" + (f" (событие {event})" if event else ""))
        if event:
            await self._handle_event(event, prev, new)

    # --- правила ---
    async def _handle_event(self, event: str, prev: SupportState, new: SupportState):
        self.last_event = event
        self_takeoff = self.self_flight_active
        emotion = self._get_emotion() if self._get_emotion else None

        try:
            for rule in self._rules:
                if not rule_matches(rule, event, prev, new, self_takeoff, emotion):
                    continue
                now = time.monotonic()
                if now < self._cooldown_until.get(rule["id"], 0.0):
                    logger.debug(f"[behavior] Правило «{rule['id']}» на кулдауне — пропуск")
                else:
                    await self._apply(rule)
                break
        except Exception as e:
            logger.error(f"[behavior] Ошибка при обработке события {event}: {e}")
        finally:
            if event in ("touchdown", "grabbed", "support_regained"):
                self.end_self_flight()

    async def _apply(self, rule: dict):
        do = rule["do"]
        anim = do["anim"]
        frames = None
        if anim:
            frames = animation_book.get_frames(anim)
            if not frames:
                if anim not in self._warned_anims:
                    logger.warning(f"[behavior] Правило «{rule['id']}»: анимация «{anim}» не найдена в animation_book — движение пропущено")
                    self._warned_anims.add(anim)

        duration_s = (frames[-1]["time"] / 1000.0 + 0.3) if frames else do["hold_sec"]
        priority = Priority[do["priority"]]
        accepted = await arbiter.request(priority, duration_s, label=f"behavior:{rule['id']}")
        if not accepted:
            logger.info(f"[behavior] Правило «{rule['id']}» отклонено арбитром (занято приоритетнее {priority.name})")
            return

        if do["cooldown_sec"] > 0:
            self._cooldown_until[rule["id"]] = time.monotonic() + do["cooldown_sec"]

        logger.info(f"[behavior] Правило «{rule['id']}»: anim={anim} emotion={do['emotion']} phrase={do['phrase']} ({priority.name})")

        if do["emotion"] and self._set_emotion:
            self._set_emotion(do["emotion"])

        if do["phrase"] and self._play_phrase:
            asyncio.create_task(self._play_phrase(do["phrase"]))

        if frames and self._servos is not None:
            await self._servos.cancel_current()
            asyncio.create_task(self._servos.play_animation(anim, on_frame=self._on_frame))


behavior_engine = BehaviorEngine()
