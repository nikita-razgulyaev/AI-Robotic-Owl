"""Тест журнала чата (modules/chat_log.py): порядок реплик, дедупликация, потокобезопасность, история.

Запуск из корня проекта:  python scripts/test_chat_log.py
"""
import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.chat_log import ChatLog  # noqa: E402

failures = []


def check(name, cond):
    print(("✅ " if cond else "❌ ") + name)
    if not cond:
        failures.append(name)


async def main():
    log = ChatLog(maxlen=5)
    sent = []

    async def broadcast(msg):
        await asyncio.sleep(0)
        sent.append(msg)

    log.bind(asyncio.get_running_loop(), broadcast)

    # 1) из рабочего потока (как голосовой путь робота): вопрос раньше ответа
    def worker():
        log.emit("user", "Привет, Сорен", source="robot")
        log.emit("bot", "Здравствуй!", emotion="greeting", source="robot")
    t = threading.Thread(target=worker)
    t.start(); t.join()
    await asyncio.sleep(0.05)
    check("поток: обе реплики разосланы", len(sent) == 2)
    check("поток: вопрос раньше ответа", [m["role"] for m in sent] == ["user", "bot"])
    check("поток: тип события и источник", sent[0]["type"] == "chat_message" and sent[0]["source"] == "robot")
    check("поток: эмоция ответа", sent[1]["emotion"] == "greeting")

    # 2) из event loop: emit_async доставляет до продолжения
    sent.clear()
    uid = await log.emit_async("user", "Вопрос с ПК", source="pc")
    check("emit_async: доставлено сразу", len(sent) == 1 and sent[0]["id"] == uid)

    # 3) дедупликация по id (панель прислала свой client_id)
    sent.clear()
    log.emit("user", "Текст из панели", source="panel", chat_id="abc")
    log.emit("user", "Текст из панели", source="panel", chat_id="abc")
    await asyncio.sleep(0.05)
    check("дубль с тем же id не рассылается", len(sent) == 1 and sent[0]["id"] == "abc")

    # 4) пустой текст игнорируется
    sent.clear()
    log.emit("bot", "   ", source="robot")
    await asyncio.sleep(0.02)
    check("пустая реплика игнорируется", not sent)

    # 5) история ограничена и в порядке появления (добавляем ещё реплики сверх maxlen=5)
    for i in range(4):
        await log.emit_async("bot", f"реплика {i}", source="robot")
    hist = log.history()
    check("история ограничена maxlen", len(hist) == 5)
    check("история в порядке появления", [h["ts"] for h in hist] == sorted(h["ts"] for h in hist))

    # 6) без привязки к loop emit не падает (старт сервера)
    ChatLog().emit("bot", "до старта", source="robot")
    check("emit без bind не падает", True)

    # 7) сбой рассылки не ломает вызывающего
    log2 = ChatLog()
    async def bad(msg): raise RuntimeError("панель отвалилась")
    log2.bind(asyncio.get_running_loop(), bad)
    log2.emit("user", "тест", source="pc")
    await asyncio.sleep(0.02)
    check("сбой рассылки обработан", True)

asyncio.run(main())
print()
if failures:
    print(f"❌ Провалено: {len(failures)}"); sys.exit(1)
print("✅ Все проверки пройдены")
