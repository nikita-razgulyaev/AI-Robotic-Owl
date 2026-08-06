"""Система памяти Сорена — краткосрочная (RAM) и долгосрочная (Qdrant + эмбеддинги)

Структура хранения:
/soren_project/
├── memory/
│   └── user_001_profile.json          ← JSON эмоционального профиля
├── qdrant_storage/                    ← векторная БД
│   ├── collections/
│   │   ├── soren_emotional_moments/   ← Ключевые моменты
│   │   ├── soren_dialogue_turns/      ← Все реплики
│   │   ├── soren_facts/               ← Факты о пользователе
│   │   └── soren_rag_chunks/          ← Канон мира
│   └── snapshots/                     ← бэкапы
"""
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from config.settings import (
    MEMORY_DIR, SHORT_TERM_MEMORY_MAX, LONG_TERM_MEMORY_ENABLED,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTIONS, QDRANT_PATH,
    RAG_ENCODER_MODEL, FAST_MODE
)
from modules.qdrant_singleton import get_qdrant_client, encode_text
from modules import memory_flags

logger = logging.getLogger(__name__)


# === ДАТАКЛАССЫ ===

@dataclass
class DialogueTurn:
    """Одна реплика диалога"""
    timestamp: str
    role: str
    text: str
    emotion: str = "calm"
    is_deep: bool = False


@dataclass
class EmotionalMoment:
    """Ключевой эмоциональный момент"""
    timestamp: str
    user_emotion: str
    soren_emotion: str
    trigger: str
    impact: float
    text: str
    category: str = ""


@dataclass
class UserFact:
    """Факт о пользователе"""
    timestamp: str
    fact_type: str
    content: str
    is_active: bool = True
    confidence: float = 0.8


@dataclass
class EmotionalProfile:
    """Эмоциональный профиль пользователя — 6 блоков"""
    user_id: str = "default"
    created_at: str = ""
    last_seen: str = ""

    dominant_emotions: Dict[str, float] = field(default_factory=lambda: {
        "joy": 0.0, "sadness": 0.0, "anger": 0.0,
        "fear": 0.0, "trust": 0.0, "surprise": 0.0
    })

    emotional_arc: List[Dict] = field(default_factory=list)
    key_moments: List[Dict] = field(default_factory=list)

    triggers: Dict[str, List[str]] = field(default_factory=lambda: {
        "joy": [], "fear": [], "anger": [], "sadness": []
    })

    relationship_temp: Dict[str, float] = field(default_factory=lambda: {
        "warmth": 0.5, "trust": 0.3, "intimacy": 0.2, "tension": 0.1
    })

    habits: Dict[str, Any] = field(default_factory=lambda: {
        "open_time": None,
        "topics_avoided": [],
        "favorite_topics": [],
        "avg_session_duration_min": 0,
        "total_sessions": 0
    })

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


# === КРАТКОСРОЧНАЯ ПАМЯТЬ (RAM) ===

class ShortTermMemory:
    """Краткосрочная память — последние N реплик в RAM"""

    def __init__(self, max_turns: int = SHORT_TERM_MEMORY_MAX):
        self.max_turns = max_turns
        self.turns: List[DialogueTurn] = []
        self.current_emotion = "calm"
        self.session_start = datetime.now()

    def add_turn(self, role: str, text: str, emotion: str = "calm"):
        turn = DialogueTurn(
            timestamp=datetime.now().isoformat(),
            role=role,
            text=text,
            emotion=emotion,
            is_deep=self._is_deep_moment(text, emotion)
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        self.current_emotion = emotion
        logger.debug(f"STM: добавлена реплика {role}, эмоция {emotion}")

    def _is_deep_moment(self, text: str, emotion: str) -> bool:
        deep_indicators = [
            "клудд", "потеря", "смерть", "предательство", "любовь",
            "страх", "надежда", "прощение", "брат", "сестра", "родители",
            "память", "скорбь", "радость", "доверие", "воссоединение",
            "прощание", "признание", "тайна", "правда", "ложь"
        ]
        text_lower = text.lower()
        has_deep = any(ind in text_lower for ind in deep_indicators)
        is_emotional = emotion in ["sad", "loving", "angry"]
        return has_deep or is_emotional

    def get_context(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns[-5:]:
            prefix = "Пользователь" if turn.role == "user" else "Сорен"
            lines.append(f"{prefix}: {turn.text}")
        return "\n".join(lines)

    def get_summary(self) -> Dict:
        duration = (datetime.now() - self.session_start).total_seconds() / 60
        return {
            "turns_count": len(self.turns),
            "current_emotion": self.current_emotion,
            "last_user_message": self.turns[-1].text if self.turns and self.turns[-1].role == "user" else "",
            "deep_moments": sum(1 for t in self.turns if t.is_deep),
            "session_duration_min": round(duration, 1)
        }

    def clear(self):
        self.turns = []
        self.current_emotion = "calm"
        self.session_start = datetime.now()


# === ДОЛГОСРОЧНАЯ ПАМЯТЬ (Qdrant + эмбеддинги) ===

class LongTermMemory:
    """Долгосрочная память через Qdrant с векторными эмбеддингами"""

    def __init__(self):
        # self.enabled — "поднят ли клиент Qdrant" (возможность), а не "включён ли
        # уровень памяти прямо сейчас" (это memory_flags.get_flags()["ltm"],
        # проверяется отдельно в каждом save_*/search_*). Поднимаем клиент сразу,
        # только если LTM включена по дефолту (config.yaml/FAST_MODE) — иначе
        # лениво поднимем через ensure_ready() при первом включении из панели.
        self.enabled = False
        self.client = None
        if memory_flags.get_flags().get("ltm", LONG_TERM_MEMORY_ENABLED):
            self._init_qdrant()

    def ensure_ready(self):
        """Ленивая инициализация клиента Qdrant — вызывается при включении LTM
        из веб-панели, если клиент ещё не был поднят при старте (например, был
        выключен по умолчанию через FAST_MODE)"""
        if self.client is None:
            self._init_qdrant()

    def _init_qdrant(self):
        try:
            from qdrant_client.models import Distance, VectorParams

            # Используем общий синглтон
            self.client = get_qdrant_client()
            self.enabled = True
            logger.info(f"✅ Qdrant LTM подключён (shared client)")
            self._ensure_collections()
        except ImportError as e:
            logger.warning(f"Зависимости не установлены: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {e}")
            self.enabled = False

    def _ensure_collections(self):
        if not self.client:
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            for name in QDRANT_COLLECTIONS.values():
                try:
                    self.client.get_collection(name)
                except:
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                    logger.info(f"Создана коллекция Qdrant: {name}")
        except Exception as e:
            logger.error(f"Ошибка создания коллекций: {e}")

    def _make_id(self, text: str, timestamp: str) -> str:
        return hashlib.md5(f"{timestamp}_{text[:50]}".encode()).hexdigest()

    def save_dialogue_turn(self, turn: DialogueTurn, user_id: str = "default"):
        """Сохраняет реплику с эмбеддингом"""
        if not memory_flags.get_flags().get("ltm", True) or not self.enabled or not self.client:
            return
        try:
            from qdrant_client.models import PointStruct
            point_id = self._make_id(turn.text, turn.timestamp)
            vector = encode_text(turn.text)
            self.client.upsert(
                collection_name=QDRANT_COLLECTIONS["dialogue_turns"],
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        **asdict(turn)
                    }
                )]
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения диалога: {e}")

    def save_emotional_moment(self, moment: EmotionalMoment, user_id: str = "default"):
        """Сохраняет эмоциональный момент с эмбеддингом"""
        if not memory_flags.get_flags().get("ltm", True) or not self.enabled or not self.client:
            return
        try:
            from qdrant_client.models import PointStruct
            point_id = self._make_id(moment.trigger, moment.timestamp)
            vector = encode_text(moment.text)
            self.client.upsert(
                collection_name=QDRANT_COLLECTIONS["emotional_moments"],
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        **asdict(moment)
                    }
                )]
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения момента: {e}")

    def save_fact(self, fact: UserFact, user_id: str = "default"):
        """Сохраняет факт с эмбеддингом"""
        if not memory_flags.get_flags().get("ltm", True) or not self.enabled or not self.client:
            return
        try:
            from qdrant_client.models import PointStruct
            point_id = self._make_id(fact.content, fact.timestamp)
            vector = encode_text(fact.content)
            self.client.upsert(
                collection_name=QDRANT_COLLECTIONS["facts"],
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        **asdict(fact)
                    }
                )]
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения факта: {e}")

    def search_relevant(self, query: str, collection: str, user_id: str = "default", limit: int = 3) -> List[Dict]:
        """Векторный поиск релевантных воспоминаний"""
        if not memory_flags.get_flags().get("ltm", True) or not self.enabled or not self.client:
            return []
        try:
            query_vector = encode_text(query)
            results = self.client.search(
                collection_name=QDRANT_COLLECTIONS.get(collection, collection),
                query_vector=query_vector,
                limit=limit * 2
            )
            filtered = [
                {
                    "score": round(r.score, 3),
                    **r.payload
                }
                for r in results
                if r.payload.get("user_id") == user_id
            ]
            return filtered[:limit]
        except Exception as e:
            logger.error(f"Ошибка поиска в LTM: {e}")
            return []

    def search_emotional_moments(self, query: str = "", user_emotion: str = None, min_impact: float = 0.0, limit: int = 5) -> List[Dict]:
        """Поиск ключевых эмоциональных моментов"""
        results = self.search_relevant(query or "эмоциональный момент", "emotional_moments", limit=limit)
        if user_emotion:
            results = [r for r in results if r.get("user_emotion") == user_emotion]
        if min_impact > 0:
            results = [r for r in results if r.get("impact", 0) >= min_impact]
        return results

    def search_dialogue_by_context(self, query: str, limit: int = 5) -> List[Dict]:
        """Поиск диалогов по смыслу (векторный)"""
        return self.search_relevant(query, "dialogue_turns", limit=limit)

    def search_facts(self, query: str = "", fact_type: str = None, is_active: bool = True, limit: int = 5) -> List[Dict]:
        """Поиск фактов о пользователе"""
        results = self.search_relevant(query or "факт о пользователе", "facts", limit=limit)
        if fact_type:
            results = [r for r in results if r.get("fact_type") == fact_type]
        if is_active:
            results = [r for r in results if r.get("is_active", True)]
        return results


# === МЕНЕДЖЕР ПАМЯТИ ===

class MemoryManager:
    """Управляет всей памятью Сорена"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.profile = self._load_profile()

    def _load_profile(self) -> EmotionalProfile:
        # Если уровень "профиль" выключен (по умолчанию из-за fast_mode или
        # вручную из панели) — не читаем ничего с диска, пустой профиль в RAM.
        if not memory_flags.get_flags().get("profile", not FAST_MODE):
            return EmotionalProfile(user_id=self.user_id)
        profile_path = MEMORY_DIR / f"user_{self.user_id}_profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return EmotionalProfile(**data)
            except Exception as e:
                logger.error(f"Ошибка загрузки профиля: {e}")
        return EmotionalProfile(user_id=self.user_id)

    def save_profile(self):
        # Уровень "профиль" выключен — не пишем ничего на диск.
        if not memory_flags.get_flags().get("profile", not FAST_MODE):
            return
        profile_path = MEMORY_DIR / f"user_{self.user_id}_profile.json"
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.profile), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения профиля: {e}")

    def record_interaction(self, user_text: str, soren_text: str, emotion: str):
        """Записывает взаимодействие в системы памяти — каждый уровень (stm/ltm/profile)
        независимо проверяется через memory_flags, вместо единого FAST_MODE как раньше"""
        flags = memory_flags.get_flags()

        # Краткосрочная — RAM only, нужна для связности диалога в пределах текущей
        # сессии, никакой записи на диск/в БД. Единственный уровень, включённый
        # по умолчанию даже в fast_mode — но теперь тоже переключаем через панель.
        if flags.get("stm", True):
            self.short_term.add_turn("user", user_text)
            self.short_term.add_turn("assistant", soren_text, emotion)

        ltm_on = flags.get("ltm", True)
        profile_on = flags.get("profile", True)

        if not ltm_on and not profile_on:
            # Ни долгосрочной памяти, ни профиля — максимальная скорость
            # (эквивалент старого FAST_MODE, но настраивается отдельно для каждого).
            return

        now = datetime.now().isoformat()

        if profile_on:
            self.profile.last_seen = now
            self._update_emotions(emotion)
            self._update_habits(user_text, soren_text)

        if ltm_on:
            # Долгосрочная — диалог
            user_turn = DialogueTurn(timestamp=now, role="user", text=user_text)
            soren_turn = DialogueTurn(timestamp=now, role="assistant", text=soren_text, emotion=emotion)
            self.long_term.save_dialogue_turn(user_turn, self.user_id)
            self.long_term.save_dialogue_turn(soren_turn, self.user_id)

            # Долгосрочная — эмоциональный момент
            if soren_turn.is_deep:
                moment = EmotionalMoment(
                    timestamp=now,
                    user_emotion="unknown",
                    soren_emotion=emotion,
                    trigger=user_text[:100],
                    impact=0.7 if emotion in ["sad", "loving", "angry"] else 0.5,
                    text=soren_text[:200],
                    category=self._categorize_moment(emotion, user_text)
                )
                self.long_term.save_emotional_moment(moment, self.user_id)
                if profile_on:
                    self.profile.key_moments.append(asdict(moment))

            # Долгосрочная — факты
            self._extract_facts(user_text, now)

        if profile_on:
            self.save_profile()

    def _update_emotions(self, emotion: str):
        alpha = 0.3
        emotion_map = {
            "calm": "joy", "loving": "joy", "determined": "trust",
            "sad": "sadness", "angry": "anger", "surprised": "surprise",
            "tired": "sadness"
        }
        target = emotion_map.get(emotion, "joy")
        for key in self.profile.dominant_emotions:
            if key == target:
                self.profile.dominant_emotions[key] = min(1.0,
                    self.profile.dominant_emotions[key] * (1 - alpha) + alpha)
            else:
                self.profile.dominant_emotions[key] *= (1 - alpha * 0.5)

    def _update_habits(self, user_text: str, soren_text: str):
        text_lower = user_text.lower()

        avoid_indicators = ["не хочу", "не говори", "забудь", "не вспоминай"]
        for ind in avoid_indicators:
            if ind in text_lower:
                topic = user_text.split(ind)[-1][:30].strip()
                if topic and topic not in self.profile.habits["topics_avoided"]:
                    self.profile.habits["topics_avoided"].append(topic)

        if any(w in text_lower for w in ["расскажи", "что", "как", "почему"]):
            topic = user_text[:40].strip()
            if topic and topic not in self.profile.habits["favorite_topics"]:
                self.profile.habits["favorite_topics"].append(topic)
                if len(self.profile.habits["favorite_topics"]) > 20:
                    self.profile.habits["favorite_topics"].pop(0)

    def _extract_facts(self, user_text: str, timestamp: str):
        """Извлекает факты о пользователе из текста"""
        text_lower = user_text.lower()

        # Имя
        if "меня зовут" in text_lower or "моё имя" in text_lower:
            import re
            match = re.search(r"(?:меня зовут|моё имя)[\s\w]*([А-Я][а-я]+)", user_text)
            if match:
                fact = UserFact(
                    timestamp=timestamp,
                    fact_type="name",
                    content=f"Имя пользователя: {match.group(1)}",
                    confidence=0.9
                )
                self.long_term.save_fact(fact, self.user_id)

        # Страхи
        fear_words = ["боюсь", "страшно", "боязнь", "страх"]
        for word in fear_words:
            if word in text_lower:
                fact = UserFact(
                    timestamp=timestamp,
                    fact_type="fear",
                    content=user_text[:100],
                    confidence=0.7
                )
                self.long_term.save_fact(fact, self.user_id)
                break

        # Радость
        joy_words = ["рад", "счастлив", "люблю", "нравится", "обожаю"]
        for word in joy_words:
            if word in text_lower:
                fact = UserFact(
                    timestamp=timestamp,
                    fact_type="joy",
                    content=user_text[:100],
                    confidence=0.7
                )
                self.long_term.save_fact(fact, self.user_id)
                break

    def _categorize_moment(self, emotion: str, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["прости", "прощ", "извини", "понял"]):
            return "breakthrough"
        if emotion == "loving" or any(w in text_lower for w in ["друг", "доверие", "благодар"]):
            return "bonding"
        if emotion == "angry" or any(w in text_lower for w in ["нет", "не буду", "против"]):
            return "conflict"
        if emotion == "sad" or any(w in text_lower for w in ["потеря", "смерть", "боль"]):
            return "crisis"
        return "breakthrough"

    def get_day_greeting(self) -> str:
        now = datetime.now()
        last = self.profile.last_seen

        if not last:
            return "Приветствую, друг."

        last_dt = datetime.fromisoformat(last)
        hours_passed = (now - last_dt).total_seconds() / 3600

        if hours_passed > 24:
            return f"Новый день, новый ветер. Прошло {int(hours_passed)} часов с нашей последней беседы."
        elif hours_passed > 6:
            return f"Прошло {int(hours_passed)} часов. Что изменилось?"
        else:
            return "Снова здесь. Хорошо."

    def get_context_for_llm(self) -> str:
        """Формирует контекст для LLM из памяти (fallback-путь для прямых вызовов
        LLMEngine без предварительно собранного memory_context — см. LLMEngine.generate).
        Основной путь для робота — RobotBrain._build_memory_context, использующий те же флаги."""
        flags = memory_flags.get_flags()

        stm = self.short_term.get_context() if flags.get("stm", True) else ""

        if not flags.get("profile", not FAST_MODE):
            # Только последние реплики текущего диалога — без профиля,
            # доминантных эмоций и привычек (это лишние токены в промпте).
            return f"Последние реплики:\n{stm}" if stm else ""

        parts = []
        if stm:
            parts.append(f"Последние реплики:\n{stm}")

        # Эмоциональный профиль
        dom = max(self.profile.dominant_emotions, key=self.profile.dominant_emotions.get)
        parts.append(f"Доминантная эмоция пользователя: {dom}")

        # Триггеры
        if self.profile.triggers["joy"]:
            parts.append(f"Что радует пользователя: {', '.join(self.profile.triggers['joy'][:3])}")
        if self.profile.triggers["fear"]:
            parts.append(f"Что тревожит пользователя: {', '.join(self.profile.triggers['fear'][:3])}")

        # Привычки
        if self.profile.habits["favorite_topics"]:
            parts.append(f"Любимые темы: {', '.join(self.profile.habits['favorite_topics'][:3])}")

        return "\n\n".join(parts) if parts else ""

    def get_relevant_memories(self, query: str) -> str:
        """Получает релевантные воспоминания из долгосрочной памяти"""
        if not memory_flags.get_flags().get("ltm", True) or not self.long_term.enabled:
            return ""

        memories = []

        # Ищем эмоциональные моменты
        moments = self.long_term.search_emotional_moments(query, limit=2)
        if moments:
            memories.append("Ключевые моменты:")
            for m in moments:
                memories.append(f"  - [{m.get('category', '')}] {m.get('trigger', '')}")

        # Ищем факты
        facts = self.long_term.search_facts(query, limit=2)
        if facts:
            memories.append("Факты о пользователе:")
            for f in facts:
                memories.append(f"  - {f.get('content', '')}")

        # Ищем похожие диалоги
        dialogues = self.long_term.search_dialogue_by_context(query, limit=2)
        if dialogues:
            memories.append("Похожие разговоры:")
            for d in dialogues:
                role = "Пользователь" if d.get("role") == "user" else "Сорен"
                memories.append(f"  - {role}: {d.get('text', '')[:60]}...")

        return "\n".join(memories) if memories else ""

    def clear(self):
        """Очищает краткосрочную память"""
        self.short_term.clear()
        logger.info("Краткосрочная память очищена")