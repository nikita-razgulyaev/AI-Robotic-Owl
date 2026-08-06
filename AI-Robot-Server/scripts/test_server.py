"""Тестовый скрипт для проверки сервера без ESP32"""
import asyncio
import websockets
import json
import time


async def test_text_command():
    """Тест текстовых команд с Сореном"""
    uri = "ws://localhost:8765/ws"

    print("🧪 Тестирование Robot AI Server - Сорен")
    print("=" * 50)

    try:
        async with websockets.connect(uri) as ws:
            print("✅ Подключено к серверу")

            # 1. Пинг
            print("1. Пинг...")
            await ws.send(json.dumps({"type": "ping", "timestamp": time.time()}))
            response = await ws.recv()
            print(f"   Ответ: {response}")

            # 2. Статус
            print("2. Получение статуса...")
            await ws.send(json.dumps({"type": "get_status"}))
            response = await ws.recv()
            data = json.loads(response)
            print(f"   Статус: {data.get('status')}")
            print(f"   Эмоция: {data.get('current_emotion')}")
            print(f"   Серво: {data.get('servo_angles', [])[:5]}...")

            # 3. Текстовый диалог с Сореном
            print("3. Диалог с Сореном...")
            test_messages = [
                "Привет, Сорен!",
                "Расскажи о Великом Древе.",
                "Что ты думаешь о Клудде?",
                "Как стать сильным?",
            ]

            for msg in test_messages:
                print(f"
   👤 Я: {msg}")
                await ws.send(json.dumps({"type": "text", "text": msg}))
                response = await ws.recv()
                data = json.loads(response)
                print(f"   🦉 Сорен: {data.get('response')}")
                print(f"   Эмоция: {data.get('emotion')}")
                print(f"   Действие: {data.get('action')}")
                await asyncio.sleep(1)

            # 4. Очистка истории
            print("
4. Очистка истории...")
            await ws.send(json.dumps({"type": "clear_history"}))
            response = await ws.recv()
            print(f"   {json.loads(response).get('message')}")

            print("
✅ Все тесты пройдены!")

    except ConnectionRefusedError:
        print("❌ Не удалось подключиться. Запусти сервер: python websocket_server.py")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(test_text_command())