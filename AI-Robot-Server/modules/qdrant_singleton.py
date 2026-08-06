"""Общий синглтон Qdrant клиента для всего приложения

Проблема: Qdrant в локальном режиме (path=) не поддерживает 
несколько одновременных подключений к одной папке.

Решение: один клиент на всё приложение.
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config.settings import QDRANT_PATH, RAG_ENCODER_MODEL

# === Singleton instances ===
_qdrant_client = None
_encoder = None


def get_qdrant_client() -> QdrantClient:
    """Возвращает единственный экземпляр QdrantClient"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def get_encoder() -> SentenceTransformer:
    """Возвращает единственный экземпляр энкодера"""
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(RAG_ENCODER_MODEL)
    return _encoder


def encode_text(text: str) -> list:
    """Преобразует текст в вектор"""
    enc = get_encoder()
    return enc.encode(text).tolist()
