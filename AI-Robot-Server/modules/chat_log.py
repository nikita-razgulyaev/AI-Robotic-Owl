"""Общий журнал диалога для панели.

Один источник правды для чата: что бы ни происходило — голос с робота, голос с микрофона
ПК, текст из панели — каждая реплика (вопрос пользователя и ответ Сорена) попадает сюда,
сохраняется в короткой истории и рассылается всем подключённым панелям событием
{"type": "chat_message", "id", "role", "text", "emotion", "source", "ts"}.

Зачем так:
  - панель показывает диалог независимо от того, откуда пришёл вопрос (раньше чат строился
    только из ответов на запросы самой панели, поэтому разговор с роботом был не виден);
  - вопрос появляется сразу после распознавания речи, а не вместе с готовым ответом:
    событие отправляется между STT и LLM;
  - только что открытая панель получает последние реплики (GET /chat/history).

Дедупликация на стороне панели идёт по полю id: если реплика уже показана (например,
панель сама добавила своё сообщение), повторно она не рисуется.

emit() можно вызывать из любого потока (в том числе из рабочего потока STT/LLM/TTS);
emit_async() — из event loop, когда нужно гарантированно отправить ДО следующего
блокирующего вызова.
"""
import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from typing import Awaitable, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

SOURCES = ("robot", "pc", "panel")


class ChatLog:
    def __init__(self, maxlen: int = 200):
        self._history: Deque[dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._broadcast: Optional[Callable[[dict], Awaitable[None]]] = None

    def bind(self, loop: asyncio.AbstractEventLoop, broadcast: Callable[[dict], Awaitable[None]]):
        """Подключает рассылку по панелям (вызывается при старте сервера)."""
        self._loop = loop
        self._broadcast = broadcast

    @staticmethod
    def make_event(role: str, text: str, emotion: Optional[str] = None, source: str = "robot",
                   chat_id: Optional[str] = None) -> dict:
        return {
            "id": chat_id or uuid.uuid4().hex,
            "role": "user" if role == "user" else "bot",
            "text": text,
            "emotion": emotion,
            "source": source if source in SOURCES else "robot",
            "ts": time.time(),
        }

    def _store(self, event: dict):
        with self._lock:
            if any(e["id"] == event["id"] for e in self._history):
                return False
            self._history.append(event)
            return True

    def history(self, limit: int = 100) -> List[dict]:
        with self._lock:
            return list(self._history)[-limit:]

    def clear(self):
        with self._lock:
            self._history.clear()

    def emit(self, role: str, text: str, emotion: Optional[str] = None, source: str = "robot",
             chat_id: Optional[str] = None) -> str:
        """Потокобезопасно: сохраняет реплику и планирует рассылку панелям. Возвращает id реплики."""
        text = (text or "").strip()
        event = self.make_event(role, text, emotion, source, chat_id)
        if not text:
            return event["id"]
        if self._store(event) and self._loop is not None and self._broadcast is not None:
            msg = {"type": "chat_message", **event}
            try:
                self._loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._safe_broadcast(msg)))
            except RuntimeError:
                pass
        return event["id"]

    async def emit_async(self, role: str, text: str, emotion: Optional[str] = None, source: str = "robot",
                         chat_id: Optional[str] = None) -> str:
        """Из event loop: ждёт, пока реплика реально ушла панелям (до следующего блокирующего вызова)."""
        text = (text or "").strip()
        event = self.make_event(role, text, emotion, source, chat_id)
        if not text:
            return event["id"]
        if self._store(event) and self._broadcast is not None:
            await self._safe_broadcast({"type": "chat_message", **event})
        return event["id"]

    async def _safe_broadcast(self, msg: dict):
        try:
            await self._broadcast(msg)
        except Exception as e:
            logger.warning(f"[chat] Не удалось разослать реплику панелям: {e}")


chat_log = ChatLog()
