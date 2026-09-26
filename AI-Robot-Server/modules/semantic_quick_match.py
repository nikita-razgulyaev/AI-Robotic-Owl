"""Семантический слой быстрых ответов — третья ступень матчинга, между
fuzzy-совпадением (modules.quick_answers, по буквам) и вызовом LLM.

Зачем: fuzzy по SequenceMatcher ловит опечатки и близкие по написанию фразы,
но не ловит смысловые перефразировки — "Сколько тебе лет?" и "Какой у тебя
возраст?" по буквам совпадают слабо, а по смыслу это один и тот же вопрос.

Подход: при старте эмбеддим все triggers из quick_answers.json в отдельную
коллекцию Qdrant (используем тот же клиент/энкодер, что и modules.memory —
см. modules/qdrant_singleton.py, чтобы не поднимать второй encoder в памяти).
На каждый вопрос пользователя, если fuzzy не нашёл ответа — ищем ближайший
вектор; если похожесть выше порога — отдаём этот quick_answer вместо LLM.

Дороже fuzzy (один embedding-запрос, это миллисекунды на CPU для короткой
фразы), но на порядки дешевле полного обращения к LLM — и всё ещё укладывается
в рамки "мгновенно" с точки зрения пользователя.
"""
import logging
import threading
from typing import Optional

from qdrant_client.models import Distance, VectorParams, PointStruct

from modules.qdrant_singleton import get_qdrant_client, encode_text
from modules.quick_answers import quick_answers, QuickAnswer

logger = logging.getLogger(__name__)

COLLECTION_NAME = "soren_quick_answer_triggers"
VECTOR_SIZE = 384  # размерность RAG_ENCODER_MODEL (paraphrase-multilingual-MiniLM-L12-v2)

# Порог косинусной близости для семантического совпадения. Выше, чем можно
# было бы ожидать для "просто похожих" фраз — ложное срабатывание здесь
# означает подмену настоящего ответа LLM заранее заготовленным не по теме,
# это хуже, чем один лишний поход в LLM.
SEMANTIC_THRESHOLD = 0.80


class SemanticQuickMatch:
    """Строит и хранит embedding-индекс триггеров quick_answers в Qdrant.
    reload() пересобирает индекс с нуля — вызывается вместе с
    quick_answers.reload() (см. build_index ниже), т.к. набор триггеров мог
    измениться."""

    def __init__(self):
        self._lock = threading.Lock()
        self._id_to_qa: dict[int, QuickAnswer] = {}
        self._ready = False
        self.build_index()

    def build_index(self):
        """Пересобирает коллекцию Qdrant с нуля из текущего quick_answers.
        Вызывать при старте сервера и после quick_answers.reload()."""
        client = get_qdrant_client()

        try:
            if client.collection_exists(COLLECTION_NAME):
                client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
        except Exception as e:
            logger.error(f"[semantic] Не удалось создать коллекцию Qdrant: {e}")
            self._ready = False
            return

        points = []
        id_to_qa = {}
        point_id = 0
        # entries доступны только через find()/count() в публичном API
        # quick_answers, поэтому проходим по триггерам через служебный доступ
        # к _entries — сознательно завязываемся на внутренний список, т.к.
        # это тот же модуль проекта, а не сторонняя библиотека
        for qa in quick_answers._entries:
            for trigger in qa.triggers:
                vector = encode_text(trigger)
                points.append(PointStruct(id=point_id, vector=vector, payload={"qa_id": qa.id}))
                id_to_qa[point_id] = qa
                point_id += 1

        if points:
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
            except Exception as e:
                logger.error(f"[semantic] Не удалось загрузить векторы в Qdrant: {e}")
                self._ready = False
                return

        with self._lock:
            self._id_to_qa = id_to_qa
            self._ready = True

        logger.info(f"🔎 Семантический индекс quick_answers: {len(points)} триггеров")

    def find(self, text: str) -> Optional[QuickAnswer]:
        """Семантический поиск ближайшего триггера. None, если индекс не
        готов, пуст, или ближайшее совпадение ниже SEMANTIC_THRESHOLD."""
        with self._lock:
            if not self._ready or not self._id_to_qa:
                return None
            id_to_qa = self._id_to_qa

        client = get_qdrant_client()
        try:
            vector = encode_text(text)
            hits = client.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=1)
        except Exception as e:
            logger.error(f"[semantic] Ошибка поиска: {e}")
            return None

        if not hits or hits[0].score < SEMANTIC_THRESHOLD:
            return None

        qa = id_to_qa.get(hits[0].id)
        if qa:
            logger.info(f"🔎 Семантическое совпадение ({hits[0].score:.2f}): {text!r} -> '{qa.id}'")
        return qa


# Глобальный экземпляр — индекс строится один раз при старте (как и
# quick_answers). Если нужно подхватить правки quick_answers.json без
# перезапуска сервера — вызвать quick_answers.reload(), затем
# semantic_quick_match.build_index() (например, из одного HTTP-эндпоинта
# /quick_answers/reload).
semantic_quick_match = SemanticQuickMatch()
