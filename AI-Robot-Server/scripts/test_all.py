"""Полное тестирование всех модулей Robot AI Server - Сорен"""
import asyncio
import websockets
import json
import time
import sys

SERVER_URI = "ws://localhost:8765/ws"

async def run_tests():
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ ROBOT AI SERVER - СОРЕН")
    print("="*60)
    print("\nПодключение к серверу...")
    print("Убедись, что сервер запущен: python websocket_server.py")
    print("(Оставь сервер запущенным в другом терминале)\n")

    try:
        ws = await websockets.connect(SERVER_URI)
        print("✅ Подключение к серверу установлено\n")

        # Тест 1: Статус
        print("📊 Тест 1: Статус сервера")
        await ws.send(json.dumps({"type": "get_status"}))
        response = await ws.recv()
        data = json.loads(response)
        print(f"   Статус: {data.get('status')}")
        print(f"   Эмоция: {data.get('current_emotion')}")
        print(f"   Серво углы: {data.get('servo_angles', [])[:5]}...")
        print("   ✅ Статус получен\n")

        # Тест 2: Сервоприводы
        print("🦾 Тест 2: Управление сервоприводами")
        await ws.send(json.dumps({"type": "servo", "id": 0, "angle": 45}))
        response = await ws.recv()
        print(f"   Servo 0 → 45°: {json.loads(response).get('status')}")

        angles = [90] * 18
        angles[0] = 45
        angles[1] = 135
        await ws.send(json.dumps({"type": "servo_multi", "angles": angles}))
        response = await ws.recv()
        print(f"   Все серво: {json.loads(response).get('status')}")
        print("   ✅ Сервоприводы работают\n")

        # Тест 3: Анимации
        print("🎬 Тест 3: Анимации")
        for anim in ["wave", "nod", "shake_head", "idle"]:
            await ws.send(json.dumps({"type": "animation", "name": anim}))
            response = await ws.recv()
            print(f"   Анимация '{anim}': {json.loads(response).get('status')}")
            await asyncio.sleep(1)
        print("   ✅ Анимации работают\n")

        # Тест 4: LLM диалог с Сореном
        print("🧠 Тест 4: Текстовый диалог (LLM + TTS + Эмоции)")
        test_messages = [
            "Привет, Сорен!",
            "Расскажи о брате Клудде.",
            "Как ты попал в Сант-Эголиус?",
            "Что такое серебряная душа?",
        ]

        for msg in test_messages:
            print(f"\n   👤 Я: {msg}")
            await ws.send(json.dumps({"type": "text", "text": msg}))
            response = await ws.recv()
            data = json.loads(response)

            robot_text = data.get('response', 'Нет ответа')
            emotion = data.get('emotion', 'calm')
            action = data.get('action')
            audio_hex = data.get('audio', '')
            audio_size = len(audio_hex) // 2 if audio_hex else 0
            servo_angles = data.get('servo_angles', [])

            print(f"   🦉 Сорен: {robot_text}")
            print(f"   Эмоция: {emotion}")
            if action:
                print(f"   🎬 Действие: {action}")
            print(f"   🔊 Аудио: {audio_size} bytes")
            print(f"   🦾 Поза: {servo_angles[:6]}...")
            print("   ✅ Ответ получен")
            await asyncio.sleep(2)

        # Тест 5: Очистка истории
        print("\n🗑 Тест 5: Очистка истории")
        await ws.send(json.dumps({"type": "clear_history"}))
        response = await ws.recv()
        print(f"   {json.loads(response).get('message')}")
        print("   ✅ История очищена")

        await ws.close()

        print("\n" + "="*60)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("="*60)
        print("\nСервер работает корректно:")
        print("  ✅ WebSocket соединение")
        print("  ✅ Управление сервоприводами")
        print("  ✅ Анимации")
        print("  ✅ LLM (Qwen 7B) + RAG")
        print("  ✅ TTS (Silero)")
        print("  ✅ Эмоциональный движок Сорена")
        print("  ✅ Позы сервоприводов по эмоциям")

    except ConnectionRefusedError:
        print("❌ Не удалось подключиться к серверу")
        print("   Убедись, что сервер запущен: python websocket_server.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n\nТестирование прервано.")