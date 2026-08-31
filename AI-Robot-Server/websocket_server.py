"""WebSocket сервер - связь с ESP32 и веб-интерфейсом + голосовое общение"""
import os
import asyncio
import json
import logging
import base64
import io
import wave
import tempfile
from pathlib import Path
from typing import Set, Optional, Dict, Tuple, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from modules.robot_brain import RobotBrain
from modules.wake_word import strip_wake_word
from modules.auth import create_session_token, verify_session_token, is_valid_device_ping
from modules import backlight_state
from modules.backlight import is_night
from config.settings import (
    SERVER_HOST, SERVER_PORT,
    VIDEO_PANEL_MIN_INTERVAL_SEC, VIDEO_PANEL_JPEG_QUALITY,
    SERVO_UPDATE_MIN_INTERVAL_SEC,
    PANEL_PASSWORD, DEVICE_KEY,
    SESSION_COOKIE_NAME, SESSION_SHORT_MAX_AGE_SEC, SESSION_REMEMBER_MAX_AGE_SEC,
    BACKLIGHT_LATITUDE, BACKLIGHT_LONGITUDE, BACKLIGHT_TIMEZONE, BACKLIGHT_CHECK_INTERVAL_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot AI Server - Soren", version="3.0")

# Статика для новой панели (изображение совы + фон) — файлы лежат рядом с этим
# скриптом в static/panel/. Если папки нет (например, при первом деплое) —
# создаём пустую, чтобы mount() не падал; картинки нужно положить туда вручную
# (owl-cutout.png, bg-nature.jpg — присланы вместе с этим файлом).
PANEL_ASSETS_DIR = Path(__file__).resolve().parent / "static" / "panel"
PANEL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/panel-assets", StaticFiles(directory=str(PANEL_ASSETS_DIR)), name="panel-assets")

robot_brain: RobotBrain = None
active_connections: Set[WebSocket] = set()

# Разделяем подключения /ws на "устройство" (ESP32, шлёт бинарные AUDI/VIDE кадры)
# и "панель" (браузер, инструментальная панель). Новое соединение по умолчанию
# считается панелью и переклассифицируется в устройство при первом бинарном пакете —
# так ESP32 не грузится ненужными ей JSON-кадрами видео.
panel_connections: Set[WebSocket] = set()
device_connections: Set[WebSocket] = set()

_last_panel_frame_ts = None

# Для каждого устройства помним, что и когда ему последний раз реально отправляли,
# чтобы слать: а) не чаще SERVO_UPDATE_MIN_INTERVAL_SEC, б) только ИЗМЕНИВШИЕСЯ углы
# (delta), а не все 18 на каждый кадр — интерполяция на ESP32 и так плавно доедет
# между редкими целями (см. interpolateServos в прошивке).
_last_servo_send: Dict[WebSocket, Tuple[float, Optional[List[int]]]] = {}

# Хочет ли конкретная панель получать видео (чекбокс "Показывать видео") —
# по умолчанию True для новых подключений, пока панель явно не скажет иначе.
panel_wants_video: Dict[WebSocket, bool] = {}

# ===== РАЗДЕЛЬНЫЕ РЕЖИМЫ АУДИО =====
audio_input_mode = "robot"   # "robot" = ESP32 микрофон, "local" = микрофон ноутбука
audio_output_mode = "robot"  # "robot" = ESP32 динамик, "local" = наушники ноутбука

# ===== LIFECYCLE =====

@app.on_event("startup")
async def startup():
    global robot_brain
    logger.info("🦉 Запуск сервера Сорена...")
    robot_brain = RobotBrain()
    robot_brain.on_servo_frame = broadcast_servo_angles_to_devices
    # Fallback: подключаем напрямую к ServoController, чтобы ручное управление
    # из панели (servo / servo_multi) тоже уходило на ESP32
    robot_brain.servos.on_servo_frame = broadcast_servo_angles_to_devices
    asyncio.create_task(_backlight_loop())
    logger.info(f"✅ Сервер готов: ws://{SERVER_HOST}:{SERVER_PORT}")

@app.on_event("shutdown")
async def shutdown():
    if robot_brain:
        robot_brain.shutdown()
    logger.info("Сервер остановлен")

# ===== HTTP ENDPOINTS =====

@app.get("/")
async def root():
    return {
        "status": "Robot AI Server - Soren is running",
        "version": "3.0",
        "audio_input_mode": audio_input_mode,
        "audio_output_mode": audio_output_mode,
        "ai_modes": robot_brain.get_modes() if robot_brain else {}
    }

@app.get("/status")
async def status():
    if robot_brain is None:
        return {"status": "initializing"}
    return {
        "status": "ready",
        "audio_input_mode": audio_input_mode,
        "audio_output_mode": audio_output_mode,
        "ai_modes": robot_brain.get_modes(),
        "memory_flags": robot_brain.get_memory_flags(),
        "model_config": robot_brain.get_model_config(),
        "quick_answers": robot_brain.get_quick_answers_status(),
        "backlight": {**backlight_state.get_state(), "effective": compute_effective_backlight_state()},
        "connections": len(active_connections),
        "panel_connections": len(panel_connections),
        "device_connections": len(device_connections),
        "servo_angles": robot_brain.servos.get_current_angles(),
        "vision_context": robot_brain.vision_context,
        "current_emotion": robot_brain.current_emotion,
        "dialog_active": robot_brain.is_dialog_active()
    }

@app.post("/audio_mode")
async def set_audio_mode(mode: str = Form(...), type: str = Form("output")):
    global audio_input_mode, audio_output_mode

    if mode not in ["robot", "local"]:
        return JSONResponse(
            {"status": "error", "message": "Invalid mode. Use 'robot' or 'local'"},
            status_code=400
        )

    if type == "input":
        audio_input_mode = mode
        logger.info(f"🎤 Режим ВВОДА аудио изменён: {mode}")
    elif type == "output":
        audio_output_mode = mode
        logger.info(f"🔊 Режим ВЫВОДА аудио изменён: {mode}")
    else:
        return JSONResponse(
            {"status": "error", "message": "Invalid type. Use 'input' or 'output'"},
            status_code=400
        )

    return JSONResponse({
        "status": "ok",
        "audio_input_mode": audio_input_mode,
        "audio_output_mode": audio_output_mode
    })

@app.post("/ai_mode")
async def set_ai_mode(module: str = Form(...), mode: str = Form(...)):
    """
    Переключение режима AI модулей.
    module: "stt", "tts", "llm"
    mode: "local" или "cloud"
    """
    if module not in ["stt", "tts", "llm"]:
        return JSONResponse(
            {"status": "error", "message": "Invalid module. Use 'stt', 'tts' or 'llm'"},
            status_code=400
        )

    if mode not in ["local", "cloud"]:
        return JSONResponse(
            {"status": "error", "message": "Invalid mode. Use 'local' or 'cloud'"},
            status_code=400
        )

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    result = await robot_brain.handle_command({"type": "set_mode", "module": module, "mode": mode})
    logger.info(f"🧠 Режим {module.upper()} изменён: {mode}")

    return JSONResponse(result)

@app.post("/llm_model")
async def set_llm_model(model_id: str = Form(...)):
    """Переключение конкретной модели LLM (id из config.yaml llm.local_models/cloud_models).
    Список, в котором найден id, сам определяет режим (local/cloud) — движок
    перезагружается сразу, без рестарта сервера."""
    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    result = await robot_brain.handle_command({"type": "set_llm_model", "model_id": model_id})
    logger.info(f"🧠 Модель LLM → {model_id}")

    return JSONResponse(result)

@app.post("/stt_model")
async def set_stt_model(model_id: str = Form(...)):
    """Переключение размера Whisper (id из config.yaml stt.whisper_models) — сразу, без рестарта"""
    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    result = await robot_brain.handle_command({"type": "set_stt_model", "model_id": model_id})
    logger.info(f"🎤 Модель STT → {model_id}")

    return JSONResponse(result)

@app.post("/tts_speaker")
async def set_tts_speaker(speaker: str = Form(...)):
    """Переключение голоса TTS для текущего режима (local: Silero-спикер, cloud: edge-tts voice)"""
    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    result = await robot_brain.handle_command({"type": "set_tts_speaker", "speaker": speaker})
    logger.info(f"🔊 Голос TTS → {speaker}")

    return JSONResponse(result)

@app.post("/memory_config")
async def set_memory_config(level: str = Form(...), enabled: str = Form(...)):
    """
    Включение/выключение уровня памяти LLM из веб-панели.
    level: "stm", "ltm", "profile" или "rag"
    enabled: "1"/"0" (или "true"/"false")
    """
    if level not in ("stm", "ltm", "profile", "rag"):
        return JSONResponse(
            {"status": "error", "message": "Invalid level. Use 'stm', 'ltm', 'profile' or 'rag'"},
            status_code=400
        )

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    enabled_bool = enabled.lower() in ("1", "true", "yes", "on")
    result = await robot_brain.handle_command({
        "type": "set_memory_config", "level": level, "enabled": enabled_bool
    })
    logger.info(f"🧠 Уровень памяти '{level}' → {'включён' if enabled_bool else 'выключен'}")

    return JSONResponse(result)

@app.post("/backlight")
async def set_backlight(mode: str = Form(...), enabled: str = Form(...)):
    """
    Управление подсветкой-ночником в подставке.
    mode: "auto" (закат-рассвет + подключение устройства) или "manual" (ручной тумблер)
    enabled: "1"/"0"
    """
    if mode not in ("auto", "manual"):
        return JSONResponse(
            {"status": "error", "message": "Invalid mode. Use 'auto' or 'manual'"},
            status_code=400
        )

    enabled_bool = enabled.lower() in ("1", "true", "yes", "on")
    if mode == "auto":
        state = backlight_state.set_auto(enabled_bool)
    else:
        state = backlight_state.set_manual(enabled_bool)

    await refresh_backlight(force=True)
    logger.info(f"💡 Подсветка: {mode} → {'вкл' if enabled_bool else 'выкл'}")

    return JSONResponse({
        "status": "ok",
        "backlight": state,
        "effective": compute_effective_backlight_state()
    })

@app.post("/quick_answers/reload")
async def reload_quick_answers():
    """Перечитывает character/quick_answers.json с диска — без рестарта сервера.
    Используется после ручного редактирования словаря быстрых ответов."""
    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    result = await robot_brain.handle_command({"type": "reload_quick_answers"})
    logger.info(f"⚡ Словарь быстрых ответов перезагружен: {result.get('quick_answers')}")

    return JSONResponse(result)

@app.post("/speak")
async def speak_text(text: str = Form(...)):
    logger.info(f"/speak вызван: '{text}'")

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    if robot_brain.llm is None:
        return JSONResponse({"status": "error", "message": "LLM не инициализирован"})

    try:
        try:
            from modules.fuzzy_matcher import correct_speech_text
            raw_text = text
            corrected_text = correct_speech_text(raw_text)
            if corrected_text != raw_text:
                logger.info(f"🎯 Fuzzy (/speak): '{raw_text}' → '{corrected_text}'")
            user_text = corrected_text
        except ImportError:
            user_text = text

        # Тот же будильник "Сорен", что и в handle_command/_process_speech —
        # без него /speak оставался бы обходным путём мимо всей логики
        # активации (см. историю с quick_answers выше по коду).
        command_text = strip_wake_word(user_text)
        if command_text is None:
            if robot_brain.is_wake_session_active():
                command_text = user_text
            else:
                return JSONResponse({
                    "status": "ignored",
                    "message": "Команда проигнорирована: нет будильника 'Сорен'",
                    "text": user_text,
                })

        robot_brain.mark_dialog_active()

        llm_result = robot_brain.generate_reply(command_text)
        response_text = llm_result.get("text", "")
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")

        if not response_text:
            return JSONResponse({"status": "error", "message": "LLM вернул пустой ответ"})

        servo_angles = robot_brain.emotion_engine.get_servo_angles(emotion)
        eye_led = robot_brain.emotion_engine.get_eye_led(emotion)

        # Текст от LLM уже готов — даже если синтез речи не удастся, ответ всё равно
        # должен дойти до чата (просто без озвучки), а не теряться целиком.
        tts_audio = robot_brain.tts.synthesize(response_text)
        if not tts_audio:
            logger.warning("TTS не вернул аудио — отправляем текстовый ответ без озвучки")

        if action:
            asyncio.create_task(robot_brain.servos.play_animation(action, on_frame=robot_brain.on_servo_frame))
        else:
            robot_brain.servos.set_all_servos(servo_angles)

        audio_b64 = ""
        if tts_audio:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(tts_audio)
            wav_bytes = wav_buffer.getvalue()
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        return JSONResponse({
            "status": "ok",
            "text": command_text,
            "raw_text": text,
            "response": response_text,
            "action": action,
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led,
            "audio_base64": audio_b64,
            "tts_failed": not bool(tts_audio),
            "audio_input_mode": audio_input_mode,
            "audio_output_mode": audio_output_mode,
            "ai_modes": robot_brain.get_modes()
        })

    except Exception as e:
        logger.error(f"Ошибка в /speak: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)})


@app.post("/voice")
async def voice_input(
    audio: UploadFile = File(...),
    audio_output_mode_param: str = Form("")
):
    current_output_mode = audio_output_mode_param if audio_output_mode_param in ["robot", "local"] else audio_output_mode

    logger.info(f"🎤 Голосовой ввод: {audio.filename}, {audio.size} bytes, output_mode={current_output_mode}")

    if robot_brain is None:
        return JSONResponse({"status": "error", "message": "Сервер ещё загружается"})

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info("Распознавание речи...")
        stt_result = robot_brain.stt.transcribe_from_file(tmp_path)
        os.unlink(tmp_path)

        if not stt_result.get("success"):
            return JSONResponse({"status": "error", "message": "Речь не распознана", "stt_error": stt_result.get("error")})

        raw_text = stt_result["text"]
        user_text = stt_result.get("corrected_text", raw_text) or raw_text

        if user_text != raw_text:
            logger.info(f"🎯 Fuzzy (/voice): '{raw_text}' → '{user_text}'")

        logger.info(f"👤 Распознано: '{user_text}'")

        if not user_text.strip():
            return JSONResponse({"status": "error", "message": "Пустой текст"})

        # Тот же будильник "Сорен", что и в handle_command/_process_speech —
        # /voice не должен быть обходным путём мимо активации по имени.
        command_text = strip_wake_word(user_text)
        if command_text is None:
            if robot_brain.is_wake_session_active():
                command_text = user_text
            else:
                return JSONResponse({
                    "status": "ignored",
                    "message": "Команда проигнорирована: нет будильника 'Сорен'",
                    "user_text": user_text,
                    "raw_text": raw_text,
                })

        robot_brain.mark_dialog_active()

        llm_result = robot_brain.generate_reply(command_text)
        response_text = llm_result.get("text", "")
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")

        servo_angles = robot_brain.emotion_engine.get_servo_angles(emotion)
        eye_led = robot_brain.emotion_engine.get_eye_led(emotion)

        tts_audio = robot_brain.tts.synthesize(response_text)

        if action:
            asyncio.create_task(robot_brain.servos.play_animation(action, on_frame=robot_brain.on_servo_frame))
        else:
            robot_brain.servos.set_all_servos(servo_angles)

        audio_b64 = ""
        if current_output_mode == "local" and tts_audio:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(tts_audio)
            wav_bytes = wav_buffer.getvalue()
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        return JSONResponse({
            "status": "ok",
            "user_text": command_text,
            "raw_text": raw_text,
            "response": response_text,
            "action": action,
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led,
            "audio_base64": audio_b64,
            "audio_input_mode": audio_input_mode,
            "audio_output_mode": current_output_mode,
            "ai_modes": robot_brain.get_modes()
        })

    except Exception as e:
        logger.error(f"Ошибка в /voice: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)})


# ===== WEBSOCKET =====

def _cookie_authorized(websocket: WebSocket) -> bool:
    """Панель/браузер: валидна ли cookie-сессия. Если PANEL_PASSWORD не задан —
    авторизация выключена (обратная совместимость)."""
    if not PANEL_PASSWORD:
        return True
    token = websocket.cookies.get(SESSION_COOKIE_NAME)
    return verify_session_token(token)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_info = f"{websocket.client.host}:{websocket.client.port}"

    is_trusted = _cookie_authorized(websocket)
    pending_first_message = None

    if not is_trusted and PANEL_PASSWORD:
        # Ни валидной cookie, ни ясности что это ESP32 — даём ОДИН шанс
        # представиться устройством (ping + device_key) в течение 5 секунд,
        # иначе соединение закрывается ДО обработки каких-либо команд.
        try:
            pending_first_message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            logger.warning(f"Неавторизованное подключение отклонено (таймаут): {client_info}")
            await websocket.close(code=4401)
            return

        text = pending_first_message.get("text")
        if not (text and is_valid_device_ping(text, DEVICE_KEY)):
            logger.warning(f"Неавторизованное подключение отклонено: {client_info}")
            await websocket.close(code=4401)
            return

        device_connections.add(websocket)

    active_connections.add(websocket)
    if websocket not in device_connections:
        panel_connections.add(websocket)  # по умолчанию считаем панелью, пока не докажет обратное

    logger.info(f"Клиент подключен: {client_info}")

    try:
        if pending_first_message is not None:
            # Уже прочитанное первое сообщение (ping от устройства) — обработать как обычно
            if "text" in pending_first_message:
                await handle_text_message(websocket, pending_first_message["text"])
            elif "bytes" in pending_first_message:
                await handle_binary_message(websocket, pending_first_message["bytes"])

        while True:
            message = await websocket.receive()
            if "text" in message:
                await handle_text_message(websocket, message["text"])
            elif "bytes" in message:
                await handle_binary_message(websocket, message["bytes"])
    except WebSocketDisconnect:
        logger.info(f"Клиент отключен: {client_info}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
    finally:
        active_connections.discard(websocket)
        panel_connections.discard(websocket)
        device_connections.discard(websocket)
        _last_servo_send.pop(websocket, None)
        panel_wants_video.pop(websocket, None)


async def send_audio_to_device(websocket: WebSocket, audio_bytes: bytes, chunk_size: int = 4096):
    """Шлёт TTS-аудио устройству кусками с троттлингом.

    4096 байт PCM 16-bit mono 48kHz = ~43 мс аудио.
    Спим ~30 мс между чанками — ESP32 успевает записать в I2S и вызвать
    webSocket.loop(), не переполняя TCP receive buffer.
    """
    target = websocket if websocket in device_connections else None

    if target is None:
        if not device_connections:
            logger.warning("📡 [TX audio] исходное устройство отключилось, доставить некому")
            return
        target = next(iter(device_connections))
        logger.info(
            f"📡 [TX audio] исходное соединение устарело, отправляю на "
            f"{target.client.host}:{target.client.port} вместо него"
        )

    total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
    logger.info(f"📡 [TX audio] отправка {len(audio_bytes)} байт ({total_chunks} чанков) → "
                f"{target.client.host}:{target.client.port}")
    start_ts = asyncio.get_event_loop().time()

    # ~43 мс на чанк; спим 30 мс, чтобы ESP32 успевал опустошать буфер
    chunk_audio_ms = (chunk_size / (48000 * 2)) * 1000
    sleep_ms = max(10, chunk_audio_ms * 0.7)

    try:
        for idx, offset in enumerate(range(0, len(audio_bytes), chunk_size)):
            chunk = audio_bytes[offset:offset + chunk_size]
            await asyncio.wait_for(target.send_bytes(b"AUDI" + chunk), timeout=5.0)

            # Троттлинг + yield event loop (обработка входящих ping/pong)
            if idx < total_chunks - 1:
                await asyncio.sleep(sleep_ms / 1000)

        elapsed = asyncio.get_event_loop().time() - start_ts
        logger.info(f"📡 [TX audio] отправлено за {elapsed:.2f}с")
    except asyncio.TimeoutError:
        logger.warning(
            f"📡 [TX audio] TIMEOUT — отправка зависла (>5с на чанке) для "
            f"{target.client.host}:{target.client.port}, разрываю"
        )
        device_connections.discard(target)
        active_connections.discard(target)
    except Exception as e:
        logger.warning(f"📡 [TX audio] failed to {target.client.host}:{target.client.port}: {e}")
        device_connections.discard(target)
        active_connections.discard(target)


async def broadcast_servo_angles_to_devices(angles: List[int]):
    """Шлёт полный кадр углов (18 int) всем подключённым устройствам как
    servo_update — используется как on_frame-коллбэк для
    ServoController.play_animation() (см. RobotBrain.on_servo_frame), чтобы
    кадры анимации/жеста реально доезжали до ESP32 по сети, а не оставались
    только в Python-состоянии ServoController (hardware_available там всегда
    False — сервер сам ничего не умеет двигать напрямую)."""
    if not device_connections:
        logger.debug("📡 [TX servo] broadcast skipped — no device connections")
        return
    servo_cmd = {"type": "servo_update", "angles": {str(i): a for i, a in enumerate(angles)}}
    logger.info(f"📡 [TX servo] broadcast → {len(device_connections)} device(s), angles: {angles}")
    dead = []
    for conn in list(device_connections):
        try:
            await conn.send_json(servo_cmd)
            logger.debug(f"📡 [TX servo] sent to {conn.client.host}:{conn.client.port}")
        except Exception as e:
            logger.warning(f"📡 [TX servo] failed to {conn.client.host}:{conn.client.port}: {e}")
            dead.append(conn)
    for conn in dead:
        device_connections.discard(conn)
        active_connections.discard(conn)


def compute_effective_backlight_state() -> bool:
    """Эффективное состояние подсветки-ночника прямо сейчас:
    - авто-режим включён → горит, если есть хоть одно подключённое устройство
      И сейчас между закатом и рассветом (см. modules/backlight.py);
    - авто-режим выключен → просто ручное состояние с панели."""
    state = backlight_state.get_state()
    if state["auto"]:
        connected = len(device_connections) > 0
        return connected and is_night(BACKLIGHT_LATITUDE, BACKLIGHT_LONGITUDE, BACKLIGHT_TIMEZONE)
    return state["manual"]


async def broadcast_backlight_to_devices(enabled: bool):
    """Шлёт команду подсветки всем подключённым устройствам"""
    cmd = {"type": "backlight", "enabled": enabled}
    dead = []
    for conn in list(device_connections):
        try:
            await conn.send_json(cmd)
        except Exception:
            dead.append(conn)
    for conn in dead:
        device_connections.discard(conn)
        active_connections.discard(conn)


_last_backlight_state: Optional[bool] = None


async def refresh_backlight(force: bool = False):
    """Пересчитывает эффективное состояние подсветки и, если оно изменилось
    (или force=True — сразу после ручного переключения в панели), рассылает
    устройствам. Регулярный вызов — из фонового цикла _backlight_loop(),
    который также подхватывает переход заката/рассвета и смену подключения
    устройства в течение BACKLIGHT_CHECK_INTERVAL_SEC."""
    global _last_backlight_state
    effective = compute_effective_backlight_state()
    if force or effective != _last_backlight_state:
        await broadcast_backlight_to_devices(effective)
        _last_backlight_state = effective
        logger.info(f"💡 Подсветка-ночник: {'ВКЛ' if effective else 'выкл'}")


async def _backlight_loop():
    """Фоновый цикл — раз в BACKLIGHT_CHECK_INTERVAL_SEC пересчитывает и,
    при изменении, рассылает состояние подсветки. Именно так подхватываются
    переход через закат/рассвет и (не)подключение устройства — без этого
    цикла авто-режим сработал бы только один раз, при старте сервера."""
    while True:
        try:
            await refresh_backlight()
        except Exception as e:
            logger.error(f"Ошибка в цикле подсветки: {e}")
        await asyncio.sleep(BACKLIGHT_CHECK_INTERVAL_SEC)


async def broadcast_to_panels(message: dict, recipients: Optional[Set[WebSocket]] = None):
    """Рассылает JSON-сообщение панелям мониторинга. Если recipients не указан —
    всем подключённым панелям; иначе — только переданному подмножеству
    (используется для видео, чтобы не слать тем, кто его отключил)."""
    targets = recipients if recipients is not None else panel_connections
    dead = []
    for conn in list(targets):
        try:
            await conn.send_json(message)
        except Exception:
            dead.append(conn)
    for conn in dead:
        panel_connections.discard(conn)
        active_connections.discard(conn)
        device_connections.discard(conn)
        panel_wants_video.pop(conn, None)


async def handle_text_message(websocket: WebSocket, text: str):
    try:
        data = json.loads(text)
        msg_type = data.get("type")

        if msg_type in ["servo", "servo_multi", "animation", "text", "get_status", "clear_history", "set_mode"]:
            result = await robot_brain.handle_command(data)
            await websocket.send_json(result)
            # Гарантированная отправка на ESP32 после ручного управления сервами из панели
            if msg_type in ("servo", "servo_multi") and device_connections:
                angles = robot_brain.servos.get_current_angles()
                logger.info(f"📡 [TX servo] manual {msg_type} trigger → broadcasting to {len(device_connections)} device(s)")
                await broadcast_servo_angles_to_devices(angles)
        elif msg_type == "ping":
            # ESP32 шлёт "ping" первым сообщением сразу после коннекта (см. .ino,
            # WStype_CONNECTED) — панель мониторинга такое никогда не шлёт. Это уже
            # существующий надёжный маркер устройства, используем его для явной
            # идентификации, не дожидаясь первого бинарного AUDI/VIDE-пакета.
            device_connections.add(websocket)
            panel_connections.discard(websocket)
            await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
        elif msg_type == "hello":
            # Явная самоидентификация клиента (например, панель шлёт {type:'hello', client:'panel'}
            # при подключении) — на случай, если в будущем логика "по умолчанию — панель"
            # изменится, лучше не полагаться только на неявный дефолт.
            client = data.get("client")
            if client == "panel":
                panel_connections.add(websocket)
                device_connections.discard(websocket)
            elif client == "esp32":
                device_connections.add(websocket)
                panel_connections.discard(websocket)
            await websocket.send_json({"type": "hello_ack", "client": client})
        elif msg_type == "video_pref":
            # Панель сообщает, хочет ли она получать видео (чекбокс "Показывать видео").
            # Не влияет на детекцию/трекинг — только на то, кому реально рассылается кадр.
            panel_wants_video[websocket] = bool(data.get("enabled", True))
        elif msg_type == "audio_mode":
            await websocket.send_json({
                "type": "audio_mode",
                "input_mode": audio_input_mode,
                "output_mode": audio_output_mode
            })
        elif msg_type == "ai_mode":
            await websocket.send_json({
                "type": "ai_mode",
                "modes": robot_brain.get_modes()
            })
        else:
            await websocket.send_json({"status": "error", "message": f"Неизвестный тип: {msg_type}"})
    except json.JSONDecodeError:
        await websocket.send_json({"status": "error", "message": "Неверный JSON"})
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        await websocket.send_json({"status": "error", "message": str(e)})


async def _maybe_broadcast_panel_frame():
    """Рендерит и рассылает кадр в панель мониторинга — не чаще VIDEO_PANEL_MIN_INTERVAL_SEC,
    и только тем панелям, у кого включено "Показывать видео". Если видео не хочет
    ВООБЩЕ никто — даже не рендерим кадр (экономия CV-потока, см. пункт про троттлинг).
    Вызывается и после VIDE (кадр детекции), и после VIDP (кадр повышенного качества,
    если прошивка обновлена) — реальную частоту ограничивает троттлинг внутри."""
    global _last_panel_frame_ts

    recipients = {conn for conn in panel_connections if panel_wants_video.get(conn, True)}
    if not recipients:
        return

    now = asyncio.get_event_loop().time()
    if _last_panel_frame_ts is not None and (now - _last_panel_frame_ts) < VIDEO_PANEL_MIN_INTERVAL_SEC:
        return

    annotated = await robot_brain.render_panel_jpeg(VIDEO_PANEL_JPEG_QUALITY)
    if not annotated:
        return

    _last_panel_frame_ts = now
    status = robot_brain.get_panel_annotation_status()
    frame_b64 = base64.b64encode(annotated).decode('ascii')
    await broadcast_to_panels({
        "type": "video_frame",
        "image": frame_b64,
        "face_detected": status["face_detected"],
        "face_bbox": status["face_bbox"],
        "faces_count": status["faces_count"],
        "dialog_active": status["dialog_active"]
    }, recipients=recipients)


async def handle_binary_message(websocket: WebSocket, data: bytes):
    try:
        # Только ESP32 шлёт бинарные AUDI/VIDE/VIDP пакеты — переклассифицируем соединение
        # (запасной вариант; основная идентификация теперь по "ping", см. handle_text_message)
        if websocket not in device_connections:
            device_connections.add(websocket)
            panel_connections.discard(websocket)

        if len(data) < 5:
            return
        data_type = data[:4].decode('ascii')
        payload = data[4:]

        if data_type == "AUDI":
            result = await robot_brain.process_audio_chunk(payload)
            if result:
                response = {
                    "type": "response",
                    "user_text": result["text"],
                    "robot_text": result["response"],
                    "action": result.get("action"),
                    "emotion": result.get("emotion", "calm"),
                    "servo_angles": result.get("servo_angles", [90]*18),
                    "eye_led": result.get("eye_led", "soft_white_low"),
                    "ai_modes": robot_brain.get_modes()
                }
                await websocket.send_json(response)

                if result["audio"] and audio_output_mode == "robot":
                    await send_audio_to_device(websocket, result["audio"])

        elif data_type == "VIDE":
            vision_result = await robot_brain.process_video_frame(payload)
            await _send_servo_update(websocket, vision_result)
            await _maybe_broadcast_panel_frame()

        elif data_type == "VIDP":
            # Кадр повышенного качества ТОЛЬКО для панели мониторинга (реже, крупнее) —
            # детекцию/трекинг/серво НЕ запускает. Если прошивка не обновлена и этот
            # тег никогда не приходит — ничего не меняется, работает как раньше на VIDE.
            ok = await robot_brain.update_panel_frame(payload)
            if ok:
                await _maybe_broadcast_panel_frame()

    except Exception as e:
        logger.error(f"Ошибка бинарных данных: {e}")


async def _send_servo_update(websocket: WebSocket, vision_result: dict):
    """Шлёт servo_update на ESP32: не чаще SERVO_UPDATE_MIN_INTERVAL_SEC и только
    ИЗМЕНИВШИЕСЯ углы (delta) — интерполяция на прошивке и так плавно доедет между
    редкими целями, полный массив на каждый кадр не нужен."""
    # FIX: не пытаться отправить в соединение, которое уже мёртво
    if websocket not in device_connections:
        return
    servo_angles = vision_result["servo_angles"]
    now = asyncio.get_event_loop().time()
    last_send_ts, last_sent_angles = _last_servo_send.get(websocket, (None, None))

    if last_send_ts is not None and (now - last_send_ts) < SERVO_UPDATE_MIN_INTERVAL_SEC:
        logger.debug(f"📡 [TX servo] throttled to {websocket.client.host}:{websocket.client.port}")
        return

    if last_sent_angles is None:
        # Первая отправка этому устройству — шлём всё, дальше только дельты
        delta = {str(i): a for i, a in enumerate(servo_angles)}
    else:
        delta = {str(i): a for i, a in enumerate(servo_angles) if a != last_sent_angles[i]}

    if not delta:
        # Ничего не изменилось — не спамим сеть, но время последней проверки обновляем
        _last_servo_send[websocket] = (now, last_sent_angles)
        logger.debug(f"📡 [TX servo] no delta to {websocket.client.host}:{websocket.client.port}")
        return

    servo_cmd = {
        "type": "servo_update",
        "angles": delta,
        "face_detected": vision_result["face_detected"],
        "face_offset": vision_result["face_offset"],
        "dialog_active": vision_result.get("dialog_active", False)
    }
    logger.info(f"📡 [TX servo] delta to {websocket.client.host}:{websocket.client.port}: {delta}")
    await websocket.send_json(servo_cmd)
    _last_servo_send[websocket] = (now, list(servo_angles))


# ===== ПАНЕЛЬ УПРАВЛЕНИЯ =====

@app.get("/panel", response_class=HTMLResponse)
async def control_panel(request: Request):
    if PANEL_PASSWORD:
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if not verify_session_token(token):
            return RedirectResponse(url="/panel/login")
        logout_link = '<a class="logout-link" href="/panel/logout">Выйти</a>'
    else:
        logout_link = ""
    return PANEL_HTML.replace("{{LOGOUT_LINK}}", logout_link)


@app.get("/panel/login", response_class=HTMLResponse)
async def panel_login_form(error: str = None):
    error_html = '<div class="error">Неверный пароль</div>' if error else ""
    return LOGIN_HTML.replace("{{ERROR}}", error_html)


@app.post("/panel/login")
async def panel_login_submit(password: str = Form(...), remember: str = Form(None)):
    if not PANEL_PASSWORD or password != PANEL_PASSWORD:
        return RedirectResponse(url="/panel/login?error=1", status_code=303)

    remember_me = bool(remember)
    token = create_session_token(
        remember=remember_me,
        short_max_age_sec=SESSION_SHORT_MAX_AGE_SEC,
        remember_max_age_sec=SESSION_REMEMBER_MAX_AGE_SEC,
    )
    max_age = SESSION_REMEMBER_MAX_AGE_SEC if remember_me else SESSION_SHORT_MAX_AGE_SEC

    resp = RedirectResponse(url="/panel", status_code=303)
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.get("/panel/logout")
async def panel_logout():
    resp = RedirectResponse(url="/panel/login")
    resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp


LOGIN_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soren — Вход</title>
<style>
  :root{
    --bg:#0b0d0b; --bg2:#0f130f; --panel:rgba(26,31,26,0.6); --border:rgba(255,255,255,0.09);
    --text:#eeece6; --text-faint:#8a8f87; --sage:#8bb385; --amber:#e0b94a; --error:#e07a68;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
    background:
      radial-gradient(900px 500px at 15% -10%, rgba(224,185,74,0.10), transparent 60%),
      radial-gradient(900px 600px at 100% 110%, rgba(139,179,133,0.10), transparent 60%),
      var(--bg);
    color:var(--text);
    font-family:'Segoe UI', system-ui, sans-serif;
  }
  .card{
    width:100%; max-width:360px; background:var(--panel); border:1px solid var(--border);
    border-radius:20px; padding:38px 32px; backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
    box-shadow:0 20px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    animation:riseIn .5s cubic-bezier(.2,.8,.2,1);
  }
  @keyframes riseIn{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }
  h1{ font-size:22px; margin:0 0 4px; text-align:center; letter-spacing:.3px; }
  .subtitle{ font-size:12.5px; color:var(--text-faint); text-align:center; margin:0 0 28px; }
  label{ font-size:12.5px; color:var(--text-faint); display:block; margin-bottom:7px; }
  input[type="password"]{
    width:100%; padding:13px 14px; border-radius:12px; border:1px solid var(--border);
    background:rgba(0,0,0,0.25); color:var(--text); font-size:14.5px; margin-bottom:18px;
    transition:border-color .15s, box-shadow .15s;
  }
  input[type="password"]:focus{ outline:none; border-color:var(--sage); box-shadow:0 0 0 3px rgba(139,179,133,0.15); }
  .remember{ display:flex; align-items:center; gap:9px; margin-bottom:22px; font-size:13px; color:var(--text-faint); cursor:pointer; }
  .remember input{ width:15px; height:15px; accent-color:var(--sage); cursor:pointer; }
  button{
    width:100%; padding:13px; border-radius:12px; border:none; cursor:pointer;
    background:linear-gradient(180deg, #97c090, var(--sage)); color:#10160f; font-size:14.5px; font-weight:700;
    letter-spacing:.2px; transition:transform .15s, box-shadow .15s, filter .15s;
    box-shadow:0 8px 20px rgba(139,179,133,0.25);
  }
  button:hover{ filter:brightness(1.05); transform:translateY(-1px); box-shadow:0 10px 24px rgba(139,179,133,0.32); }
  button:active{ transform:translateY(0); }
  .error{
    background:rgba(224,122,104,0.12); border:1px solid rgba(224,122,104,0.35); color:var(--error);
    font-size:12.5px; padding:10px 12px; border-radius:10px; margin-bottom:18px; text-align:center;
  }
</style>
</head>
<body>
  <div class="card">
    <h1>🦉 Soren</h1>
    <p class="subtitle">Панель мониторинга — вход</p>
    {{ERROR}}
    <form method="post" action="/panel/login">
      <label for="password">Пароль</label>
      <input type="password" id="password" name="password" autocomplete="current-password" autofocus required>
      <label class="remember">
        <input type="checkbox" name="remember" value="1">
        Запомнить это устройство (90 дней)
      </label>
      <button type="submit">Войти</button>
    </form>
  </div>
</body>
</html>
"""


PANEL_HTML = """

<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Сорен · Панель управления</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,380;9..144,520;9..144,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    /* ---- палитра (5 цветов, из окраса сипухи) ---- */
    --beige:        #F3EEE4;   /* светлый тёплый бежевый — панели, виньетка */
    --ash:          #A8A29A;   /* пепельно-серый — вторичный текст, неактив */
    --cream:        #FBF9F4;   /* кремово-белый — светлый текст на тёмном */
    --amber:        #B8875A;   /* приглушённый янтарь — акцент */
    --amber-2:      #C99968;   /* второй тон акцента, для градиента */
    --graphite:     #38332E;   /* графитовый — основной текст */
    --beige-rgb:    243,238,228;
    --cream-rgb:    251,249,244;
    --graphite-rgb: 56,51,46;
    --amber-rgb:    184,135,90;

    --text: var(--graphite);      /* основной текст — меняется в тёмной теме */
    --vignette-rgb: var(--beige-rgb); /* цвет виньетки поверх фото — меняется в тёмной теме */
    --line-rgb: var(--graphite-rgb);  /* бордеры/разделители/hover — меняется в тёмной теме */
    --surface-rgb: var(--cream-rgb);  /* подложки карточек/чата/полей — меняется в тёмной теме */

    --glass-fill:     rgba(var(--cream-rgb),0.66);
    --glass-fill-top: rgba(var(--cream-rgb),0.46);
    --glass-border:   rgba(var(--cream-rgb),0.9);
    --glass-shadow:   rgba(var(--graphite-rgb),0.24);

    --radius: 22px;
    --gap: 18px;
    --header-h: 64px;

    --ease-spring: cubic-bezier(.22,1.12,.36,1);
  }

  /* ==================== ТЁМНАЯ ТЕМА ====================
     Не инверсия светлой — отдельный тёплый графит (в тон тёмных перьев
     сипухи), а не нейтральный "айтишный" чёрный. */
  [data-theme="dark"]{
    --text: var(--cream);              /* весь текст — кремово-белый */
    --vignette-rgb: 30,27,24;          /* виньетка — тёплый тёмный графит, не бежевый */
    --line-rgb: var(--cream-rgb);      /* бордеры/разделители/hover — светлые на тёмном */
    --surface-rgb: 78,71,63;           /* подложки карточек/чата/полей — теплее и светлее фона панели */

    --amber:   #D9995F;                /* акцент чуть ярче — гаснет на тёмном иначе */
    --amber-2: #E8B37E;

    --glass-fill:     rgba(45,41,37,0.50);
    --glass-fill-top: rgba(70,64,57,0.30);
    --glass-border:   rgba(var(--cream-rgb),0.10);
    --glass-shadow:   rgba(0,0,0,0.45);
  }
  [data-theme="dark"] body{ background:#161412; }

  *{ box-sizing:border-box; }
  html,body{ height:100%; }
  body{
    margin:0;
    font-family:'Inter', sans-serif;
    color:var(--text);
    overflow:hidden;
    -webkit-font-smoothing:antialiased;
    transition:color .4s ease, background-color .4s ease;
  }

  /* ================= ФОН ================= */
  .bg-photo{
    position:fixed; inset:0;
    background-image:url('/panel-assets/bg.png');
    background-size:cover;
    background-position:center;
    z-index:-3;
  }
  /* лёгкий шум по всей сцене — убирает "пластиковый" плоский вид */
  .bg-noise{
    position:fixed; inset:0;
    z-index:-1;
    opacity:.05;
    mix-blend-mode:overlay;
    pointer-events:none;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }
  /* бежевая виньетка поверх фото: заблюрена и укрыта бежевым — КРОМЕ зоны совы,
     там прозрачно, чтобы фон был виден чётко под самой совой */
  .bg-vignette{
    position:fixed; inset:0;
    z-index:-2;
    backdrop-filter:blur(18px) saturate(118%);
    -webkit-backdrop-filter:blur(18px) saturate(118%);
    background:
      radial-gradient(ellipse 36% 68% at 17% 55%, rgba(var(--vignette-rgb),0) 0%, rgba(var(--vignette-rgb),0.10) 42%, rgba(var(--vignette-rgb),0.66) 84%),
      linear-gradient(180deg, rgba(var(--vignette-rgb),0.46), rgba(var(--vignette-rgb),0.52));
    transition:background .5s ease;
    opacity:0.8;
  }

  /* ================= LAYOUT ================= */
  .app{
    position:relative;
    height:100vh;
    display:flex;
    flex-direction:column;
    padding:14px 22px 22px;
    gap:14px;
  }

  /* ---- header ---- */
  header{
    height:var(--header-h);
    flex:0 0 auto;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 22px;
  }
  .logo{
    font-family:'Fraunces', serif;
    font-weight:600;
    font-size:22px;
    letter-spacing:.2px;
    display:flex;
    align-items:center;
    gap:10px;
  }
  .logo .mark{
    width:26px; height:26px;
    display:flex; align-items:center; justify-content:center;
    color:var(--amber);
  }
  .header-right{ display:flex; align-items:center; gap:22px; }
  .status{ display:flex; align-items:center; gap:8px; font-size:13.5px; color:var(--text); }
  .status .dot{
    width:8px; height:8px; border-radius:50%;
    background:#5C8A5C;
    box-shadow:0 0 0 3px rgba(92,138,92,.18);
    transition:background .25s, box-shadow .25s;
  }
  .status.offline .dot{
    background:#B2504A;
    box-shadow:0 0 0 3px rgba(178,80,74,.18);
  }
  .status .label{ font-variant-numeric:tabular-nums; }

  /* ==================== ПЕРЕКЛЮЧАТЕЛЬ ТЕМЫ: пейзаж день/ночь ==================== */
  .day-night-toggle{
    position:relative;
    width:62px; height:30px; flex:0 0 auto;
    border-radius:999px;
    overflow:hidden;
    cursor:pointer;
    padding:0;
    border:1px solid rgba(var(--line-rgb),0.2);
    box-shadow:0 1px 0 rgba(255,255,255,.5) inset, 0 3px 10px -4px rgba(var(--graphite-rgb),.45);
    transition:transform .18s var(--ease-spring), box-shadow .18s;
  }
  .day-night-toggle:hover{ transform:translateY(-1px); box-shadow:0 1px 0 rgba(255,255,255,.5) inset, 0 5px 14px -4px rgba(var(--graphite-rgb),.5); }
  .day-night-toggle:active .dn-knob{ width:26px; } /* лёгкое "сжатие" при клике, как у стеклянных кнопок */

  .dn-layer{ position:absolute; inset:0; transition:opacity .6s ease; }
  .dn-sky-day{ background:linear-gradient(180deg,#bce4fb 0%,#eef8ff 78%); opacity:1; }
  .dn-sky-night{ background:linear-gradient(180deg,#332a56 0%,#c97a44 100%); opacity:0; }
  html[data-theme="dark"] .dn-sky-day{ opacity:0; }
  html[data-theme="dark"] .dn-sky-night{ opacity:1; }

  .dn-stars{ position:absolute; inset:0; opacity:0; transition:opacity .6s ease; }
  html[data-theme="dark"] .dn-stars{ opacity:1; }
  .dn-stars circle{ fill:#fff; }

  .dn-mountains{ position:absolute; left:0; right:0; bottom:-1px; width:100%; height:15px; }
  .dn-mountains .dn-m-back{ fill:#5b8a63; transition:fill .6s ease; }
  .dn-mountains .dn-m-front{ fill:#3c5f42; transition:fill .6s ease; }
  html[data-theme="dark"] .dn-mountains .dn-m-back{ fill:#2a3550; }
  html[data-theme="dark"] .dn-mountains .dn-m-front{ fill:#1c2436; }

  .dn-knob{
    position:absolute; top:3px; left:3px;
    width:24px; height:24px; border-radius:50%;
    transform:translateX(0);
    transition:transform .45s var(--ease-spring), width .15s ease;
    box-shadow:0 1px 3px rgba(0,0,0,.4), inset 0 1px 1px rgba(255,255,255,.7);
  }
  html[data-theme="dark"] .dn-knob{ transform:translateX(32px); }

  .dn-face{ position:absolute; inset:0; border-radius:50%; transition:opacity .35s ease; overflow:hidden; }
  .dn-face-sun{ background:radial-gradient(circle at 35% 30%, #fff6d8, #ffc94a 75%); opacity:1; box-shadow:0 0 8px 1px rgba(255,201,74,.7); }
  .dn-face-moon{ background:radial-gradient(circle at 40% 32%, #ffffff, #d7e2ee 80%); opacity:0; }
  html[data-theme="dark"] .dn-face-sun{ opacity:0; box-shadow:none; }
  html[data-theme="dark"] .dn-face-moon{ opacity:1; }
  .dn-crater{ position:absolute; border-radius:50%; background:rgba(150,165,185,.55); }
  .logout-link{
    font-size:12.5px; color:var(--text); opacity:.55; text-decoration:none;
    padding:8px 12px; border-radius:20px; border:1px solid transparent; transition:.15s;
  }
  .logout-link:hover{ opacity:1; color:#B2504A; border-color:rgba(178,80,74,.35); background:rgba(178,80,74,.06); }

  /* ---- main split: owl | controls ---- */
  .main{
    flex:1 1 auto;
    min-height:0;
    display:flex;
    gap:26px;
  }

  .owl-col{
    flex:0 0 37%;
    position:relative;
    display:flex;
    align-items:flex-end;
    justify-content:center;
    min-height:0;
    transform:translateY(22px); /* съедает нижний padding .app — сова касается края экрана */
  }
  .owl-col img{
    max-height:calc(96% + 22px);
    max-width:100%;
    object-fit:contain;
    filter:drop-shadow(0 30px 40px rgba(40,35,30,.35));
  }
  .owl-caption{
    position:absolute;
    left:6px; bottom:14px;
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    letter-spacing:.06em;
    color:var(--text);
    opacity:.55;
    writing-mode:vertical-rl;
    text-transform:uppercase;
  }

  .control-col{
    flex:1 1 63%;
    min-height:0;
    display:flex;
    flex-direction:column;
    gap:14px;
  }

  /* ================= GLASS PANEL ================= */
  .panel{
    position:relative;
    border-radius:var(--radius);
    background:
      linear-gradient(155deg, var(--glass-fill-top), var(--glass-fill) 46%);
    border:1px solid var(--glass-border);
    backdrop-filter:blur(22px) saturate(130%);
    -webkit-backdrop-filter:blur(22px) saturate(130%);
    box-shadow:
      0 1px 0 rgba(var(--cream-rgb),.6) inset,
      0 2px 10px -4px var(--glass-shadow),
      0 22px 46px -18px var(--glass-shadow);
    overflow:hidden;
    display:flex;
    flex-direction:column;
    transition:flex-basis .38s var(--ease-spring), flex-grow .38s var(--ease-spring),
      background .45s ease, border-color .45s ease, box-shadow .45s ease;
  }

  .panel-head{
    flex:0 0 auto;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:14px 20px;
    border-bottom:1px solid rgba(var(--line-rgb),0.08);
  }
  .panel-head .title{
    display:flex; align-items:center; gap:9px;
    font-size:13.5px;
    font-weight:600;
    color:var(--text);
  }
  .panel-head .title svg{ opacity:.7; }

  .collapse-btn{
    width:28px; height:28px;
    border-radius:8px;
    border:none;
    background:transparent;
    cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    color:var(--text);
    opacity:.6;
    transition:transform .3s var(--ease-spring), background .18s, opacity .18s;
  }
  .collapse-btn:hover{ background:rgba(var(--line-rgb),0.07); opacity:1; }
  .collapse-btn svg{ transition:transform .32s var(--ease-spring); }
  .panel.collapsed .collapse-btn svg{ transform:rotate(180deg); }

  /* tabs */
  .tabs{
    display:flex;
    gap:4px;
    padding:0 20px;
    margin:12px 0 0;
    flex:0 0 auto;
  }
  .tab{
    appearance:none; border:none; cursor:pointer;
    font-family:'Inter', sans-serif;
    font-size:12.5px; font-weight:500;
    padding:8px 15px;
    border-radius:10px;
    color:var(--text);
    opacity:.6;
    background:transparent;
    transition:background .2s, opacity .2s, transform .12s var(--ease-spring);
  }
  .tab:hover{ opacity:.85; }
  .tab.active{
    opacity:1;
    color:var(--cream);
    background:linear-gradient(150deg, var(--amber-2), var(--amber));
    box-shadow:0 1px 0 rgba(255,255,255,.35) inset, 0 6px 14px -6px rgba(var(--amber-rgb),.6);
  }
  .tab.active:active{ transform:scale(.97); box-shadow:0 1px 3px -1px rgba(var(--amber-rgb),.5) inset; }

  .panel-body{
    flex:1 1 auto;
    min-height:0;
    overflow-y:auto;
    padding:14px 20px 14px;
  }
  .panel-body::-webkit-scrollbar{ width:6px; }
  .panel-body::-webkit-scrollbar-track{ background:transparent; }
  .panel-body::-webkit-scrollbar-thumb{ background:rgba(var(--amber-rgb),.35); border-radius:10px; }
  .panel-body::-webkit-scrollbar-thumb:hover{ background:rgba(var(--amber-rgb),.55); }

  .tab-pane{ display:none; height:100%; }
  .tab-pane.active{ display:block; animation:fadeUp .28s var(--ease-spring); }
  @keyframes fadeUp{ from{ opacity:0; transform:translateY(6px);} to{ opacity:1; transform:translateY(0);} }

  /* ---- collapse behaviour ---- */
  .panel-control{ flex:1 1 61%; }
  .panel-settings{ flex:0 0 39%; }
  .panel-settings.collapsed{ flex:0 0 auto; }
  .panel-settings.collapsed .tabs,
  .panel-settings.collapsed .panel-body{ display:none; }
  .panel-control.expanded{ flex:1 1 auto; }

  /* ================= small UI atoms ================= */
  .mini-card{
    background:rgba(var(--surface-rgb),.4);
    border:1px solid rgba(var(--surface-rgb),.5);
    border-radius:14px;
    padding:11px 14px;
    box-shadow:0 1px 0 rgba(var(--cream-rgb),.35) inset, 0 2px 8px -4px rgba(var(--graphite-rgb),.22);
  }
  .grid-2{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .row-between{ display:flex; align-items:center; justify-content:space-between; }
  .label-sm{ font-size:11.5px; color:var(--text); opacity:.6; }
  .value-lg{ font-family:'Fraunces', serif; font-size:20px; font-weight:520; margin-top:2px; }

  /* toggle switch */
  .switch{ position:relative; width:38px; height:22px; flex:0 0 auto; }
  .switch input{ opacity:0; width:0; height:0; }
  .switch .track{
    position:absolute; inset:0; border-radius:20px;
    background:rgba(var(--line-rgb),0.18);
    cursor:pointer;
    transition:background .25s;
    box-shadow:inset 0 2px 4px rgba(var(--graphite-rgb),.3), inset 0 -1px 0 rgba(var(--cream-rgb),.12);
  }
  .switch .track::after{
    content:''; position:absolute; left:3px; top:3px;
    width:16px; height:16px; border-radius:50%;
    background:linear-gradient(160deg, var(--cream), #ece6d8);
    box-shadow:0 1px 2px rgba(0,0,0,.35), 0 0 0 1px rgba(var(--graphite-rgb),.05), inset 0 1px 1px rgba(255,255,255,.9);
    transition:transform .25s var(--ease-spring);
  }
  .switch input:checked + .track{
    background:linear-gradient(150deg, var(--amber-2), var(--amber));
    box-shadow:inset 0 1px 3px rgba(0,0,0,.25), 0 0 9px rgba(var(--amber-rgb),.4);
  }
  .switch input:checked + .track::after{ transform:translateX(16px); }
  .switch input:disabled + .track{ opacity:.45; cursor:not-allowed; }

  /* range slider (servos) */
  input[type=range]{
    -webkit-appearance:none; appearance:none;
    width:100%; height:4px; border-radius:3px;
    background:rgba(var(--line-rgb),0.14);
    outline:none;
  }
  input[type=range]::-webkit-slider-thumb{
    -webkit-appearance:none;
    width:15px; height:15px; border-radius:50%;
    background:var(--cream);
    border:2px solid var(--amber);
    cursor:pointer;
    box-shadow:0 2px 6px rgba(var(--graphite-rgb),.25);
  }

  /* chat */
  .chat-log{ display:flex; flex-direction:column; gap:10px; margin-bottom:12px; min-height:120px; }
  .chat-empty{
    flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:10px; padding:40px 20px; text-align:center; color:var(--text); opacity:.5;
  }
  .chat-empty svg{ width:34px; height:34px; opacity:.7; }
  .chat-empty span{ font-size:12.5px; max-width:280px; line-height:1.5; }
  .msg{ max-width:78%; padding:9px 13px; border-radius:14px; font-size:13px; line-height:1.42; }
  .msg.bot{ align-self:flex-start; background:rgba(var(--surface-rgb),.55); border:1px solid rgba(var(--line-rgb),.25); }
  .msg.user{ align-self:flex-end; background:linear-gradient(150deg, var(--amber-2), var(--amber)); color:var(--cream); }
  .chat-input{
    display:flex; gap:8px;
    position:sticky; bottom:0;
  }
  .chat-input input{
    flex:1; border:1px solid rgba(var(--line-rgb),0.14);
    background:rgba(var(--surface-rgb),.5);
    border-radius:12px; padding:10px 13px;
    font-family:'Inter'; font-size:13px; color:var(--text);
    outline:none;
  }
  .send-btn{
    border:none; border-radius:12px; padding:0 16px;
    background:linear-gradient(150deg, var(--amber-2), var(--amber));
    color:var(--cream); font-weight:600; font-size:13px; cursor:pointer;
    box-shadow:0 1px 0 rgba(255,255,255,.4) inset, 0 4px 10px -4px rgba(var(--amber-rgb),.7);
    transition:transform .12s var(--ease-spring), box-shadow .15s;
  }
  .send-btn:hover{ transform:translateY(-1px); }
  .send-btn:active{ transform:translateY(0) scale(.97); box-shadow:0 1px 4px -1px rgba(var(--amber-rgb),.6) inset; }

  /* camera */
  .camera-frame{
    position:relative; border-radius:16px; overflow:hidden;
    width:100%; height:230px;
    background:#2a2622;
    display:flex; align-items:center; justify-content:center;
    color:rgba(251,249,244,.4); font-size:12px;
  }
  .camera-frame img{ width:100%; height:100%; object-fit:contain; }
  .live-badge{
    position:absolute; top:10px; left:10px;
    background:rgba(178,80,74,.9); color:var(--cream);
    font-size:10.5px; font-weight:600; letter-spacing:.03em;
    padding:3px 8px; border-radius:6px;
    display:flex; align-items:center; gap:5px;
  }
  .live-badge::before{ content:''; width:5px; height:5px; border-radius:50%; background:var(--cream); }

  /* servo grid */
  .servo-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
  .servo-card{ background:rgba(var(--surface-rgb),.4); border:1px solid rgba(var(--surface-rgb),.5); border-radius:12px; padding:6px 10px; }
  .servo-card .row-between{ margin-bottom:4px; }
  .servo-card .ch{ font-family:'IBM Plex Mono'; font-size:10.5px; opacity:.55; }
  .servo-card .deg{ font-size:12px; font-weight:600; }

  /* log list */
  .log-list{ display:flex; flex-direction:column; gap:8px; }
  .log-item{ display:flex; gap:10px; font-size:12.5px; padding:8px 0; border-bottom:1px solid rgba(var(--line-rgb),0.07); }
  .log-item .t{ font-family:'IBM Plex Mono'; font-size:10.5px; opacity:.5; flex:0 0 52px; padding-top:1px; }

  select.select-line{
    width:100%; padding:9px 30px 9px 11px; border-radius:10px;
    border:1px solid rgba(var(--line-rgb),0.14);
    background-color:rgba(var(--surface-rgb),.5);
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%2338332e' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
    background-repeat:no-repeat;
    background-position:right 12px center;
    appearance:none; -webkit-appearance:none; -moz-appearance:none;
    font-family:'Inter'; font-size:13px; color:var(--text);
    box-shadow:0 1px 0 rgba(var(--cream-rgb),.4) inset, 0 2px 6px -3px rgba(var(--graphite-rgb),.25);
    cursor:pointer;
  }
  html[data-theme="dark"] select.select-line{
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23fbf9f4' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  }
  select.select-line:disabled{ opacity:.5; cursor:not-allowed; }

  .hint-text{ font-size:11.5px; color:var(--text); opacity:.55; line-height:1.5; margin:12px 2px 0; }

  /* mic button + voice visualizer */
  .mic-btn{
    width:38px; height:38px; flex:0 0 auto; border-radius:12px;
    border:1px solid rgba(var(--line-rgb),0.14);
    background:rgba(var(--surface-rgb),.5);
    color:var(--text); cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 1px 0 rgba(var(--cream-rgb),.5) inset, 0 2px 6px -3px rgba(var(--graphite-rgb),.3);
    transition:background .18s, color .18s, border-color .18s, transform .12s var(--ease-spring);
  }
  .mic-btn:hover{ border-color:var(--amber); color:var(--amber); }
  .mic-btn:active{ transform:scale(.94); }
  .mic-btn.recording{
    background:linear-gradient(150deg, var(--amber-2), var(--amber));
    border-color:transparent; color:var(--cream);
    animation:micPulse 1.4s infinite;
  }
  @keyframes micPulse{ 0%,100%{ box-shadow:0 0 0 0 rgba(var(--amber-rgb),.4);} 50%{ box-shadow:0 0 0 8px rgba(var(--amber-rgb),0);} }
  .audio-visualizer{ width:100%; height:34px; margin-bottom:10px; border-radius:10px; background:rgba(var(--surface-rgb),.35); }

  /* chat quick-action chips */
  .chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .chip-btn{
    border:1px solid rgba(var(--line-rgb),0.14);
    background:rgba(var(--surface-rgb),.4);
    color:var(--text); font-family:'Inter'; font-size:12px; font-weight:500;
    padding:7px 13px; border-radius:20px; cursor:pointer;
    box-shadow:0 1px 0 rgba(var(--cream-rgb),.5) inset, 0 2px 6px -3px rgba(var(--graphite-rgb),.3);
    transition:border-color .18s, transform .14s var(--ease-spring), box-shadow .18s;
  }
  .chip-btn:hover{ border-color:var(--amber); transform:translateY(-1px); }
  .chip-btn:active{ transform:translateY(0) scale(.96); box-shadow:0 1px 3px -1px rgba(var(--graphite-rgb),.3) inset; }
  .chip-btn.chip-danger{ color:#B2504A; border-color:rgba(178,80,74,.3); }
  .chip-btn.chip-danger:hover{ border-color:#B2504A; }

  /* camera badges */
  .cam-badges{ display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:14px; }
  .cam-badge{
    display:flex; align-items:center; gap:7px;
    font-size:11px; letter-spacing:.03em; text-transform:uppercase;
    padding:6px 12px; border-radius:20px;
    border:1px solid rgba(var(--line-rgb),0.14);
    background:rgba(var(--surface-rgb),.35);
    color:var(--text); opacity:.6;
    transition:opacity .2s, border-color .2s, color .2s;
  }
  .cam-badge .dot{ width:6px; height:6px; border-radius:50%; background:currentColor; }
  .cam-badge.on{ opacity:1; color:var(--amber); border-color:rgba(var(--amber-rgb),.4); }
  #cameraPlaceholder{ font-size:12px; color:var(--cream); opacity:.5; text-align:center; padding:0 20px; }

  /* AI core segmented local/cloud */
  .ai-core-card{ display:grid; grid-template-columns:repeat(3,1fr); gap:0; padding:0; overflow:hidden; }
  .ai-core-col{ padding:9px 14px; }
  .ai-core-col + .ai-core-col{ border-left:1px solid rgba(var(--line-rgb),0.12); }
  .segmented{
    display:flex; border:1px solid rgba(var(--line-rgb),0.14); border-radius:9px; overflow:hidden;
    box-shadow:inset 0 1px 3px rgba(var(--graphite-rgb),.16);
  }
  .segmented button{
    flex:1; border:none; background:transparent; color:var(--text); opacity:.55;
    font-family:'Inter'; font-size:11.5px; font-weight:500;
    padding:7px 4px; cursor:pointer; transition:.15s;
  }
  .segmented button:first-child{ border-right:1px solid rgba(var(--line-rgb),0.14); }
  .segmented button.active{
    opacity:1; color:var(--cream);
    background:linear-gradient(150deg, var(--amber-2), var(--amber));
    box-shadow:0 1px 0 rgba(255,255,255,.35) inset, 0 2px 6px -2px rgba(var(--amber-rgb),.6);
  }
  .segmented button:active{ transform:scale(.97); }

  ::selection{ background:rgba(var(--amber-rgb),.35); }

  button:focus-visible, .tab:focus-visible, .chip-btn:focus-visible,
  input:focus-visible, select:focus-visible, .switch input:focus-visible + .track,
  .day-night-toggle:focus-visible, .mic-btn:focus-visible, .segmented button:focus-visible{
    outline:none;
    box-shadow:0 0 0 3px rgba(var(--amber-rgb),.45);
  }
  .tab.active:focus-visible{ box-shadow:0 1px 0 rgba(255,255,255,.35) inset, 0 0 0 3px rgba(var(--amber-rgb),.45); }
</style>

</head>
<body>

  <div class="bg-photo"></div>
  <div class="bg-vignette"></div>
  <div class="bg-noise"></div>

  <div class="app">

    <header>
      <div class="logo">
        <span class="mark">
          <svg width="26" height="26" viewBox="0 0 32 32" fill="none">
            <path d="M9.5 7.5 12 13 7 12.2Z" fill="currentColor"/>
            <path d="M22.5 7.5 20 13 25 12.2Z" fill="currentColor"/>
            <path d="M16 7C10 7 5.5 12.2 5.5 18.6 5.5 25.2 10.2 29.5 16 29.5S26.5 25.2 26.5 18.6C26.5 12.2 22 7 16 7Z" fill="currentColor" opacity=".1"/>
            <path d="M16 7C10 7 5.5 12.2 5.5 18.6 5.5 25.2 10.2 29.5 16 29.5S26.5 25.2 26.5 18.6C26.5 12.2 22 7 16 7Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
            <circle cx="11.9" cy="18.3" r="4" fill="currentColor" opacity=".1"/>
            <circle cx="20.1" cy="18.3" r="4" fill="currentColor" opacity=".1"/>
            <circle cx="11.9" cy="18.3" r="4" stroke="currentColor" stroke-width="1.4"/>
            <circle cx="20.1" cy="18.3" r="4" stroke="currentColor" stroke-width="1.4"/>
            <circle cx="11.9" cy="18.3" r="1.3" fill="currentColor"/>
            <circle cx="20.1" cy="18.3" r="1.3" fill="currentColor"/>
            <path d="M16 20.6 14.3 23.4h3.4Z" fill="currentColor"/>
          </svg>
        </span>
        Сорен
      </div>
      <div class="header-right">
        <div class="status offline" id="connectionIndicator">
          <span class="dot"></span>
          <span class="label" id="connectionLabel">Connection closed</span>
        </div>
        <div class="status offline" id="statusIndicator">
          <span class="dot"></span>
          <span class="label" id="statusLabel">Offline</span>
        </div>
        <button class="day-night-toggle" id="themeToggle" title="Тема: авто" onclick="toggleTheme()">
          <span class="dn-layer dn-sky-day"></span>
          <span class="dn-layer dn-sky-night"></span>
          <svg class="dn-stars" viewBox="0 0 62 30"><circle cx="10" cy="8" r="0.9"/><circle cx="18" cy="14" r="0.6"/><circle cx="6" cy="16" r="0.6"/><circle cx="15" cy="6" r="0.5"/></svg>
          <svg class="dn-mountains" viewBox="0 0 62 15" preserveAspectRatio="none">
            <path class="dn-m-back" d="M0,15 L0,9 L9,4 L16,8 L24,2 L33,8 L41,4 L49,9 L56,5 L62,8 L62,15 Z"/>
            <path class="dn-m-front" d="M0,15 L0,12 L7,7 L13,10 L21,5 L29,10 L38,6 L46,11 L53,7 L62,11 L62,15 Z"/>
          </svg>
          <span class="dn-knob" id="dnKnob">
            <span class="dn-face dn-face-sun"></span>
            <span class="dn-face dn-face-moon">
              <span class="dn-crater" style="width:5px;height:5px;top:4px;left:5px;"></span>
              <span class="dn-crater" style="width:3px;height:3px;top:12px;left:11px;"></span>
              <span class="dn-crater" style="width:2.5px;height:2.5px;top:7px;left:14px;"></span>
            </span>
          </span>
        </button>
        {{LOGOUT_LINK}}
      </div>
    </header>

    <div class="main">

      <div class="owl-col">
        <span class="owl-caption">SOREN · BARN OWL UNIT</span>
        <img src="/panel-assets/owl.png" alt="Сипуха">
      </div>

      <div class="control-col">

        <!-- ============ УПРАВЛЕНИЕ И ОТСЛЕЖИВАНИЕ ============ -->
        <section class="panel panel-control" id="panelControl">
          <div class="panel-head">
            <div class="title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/></svg>
              Управление и отслеживание
            </div>
          </div>
          <div class="tabs" data-group="control">
            <button class="tab active" data-tab="chat">Чат</button>
            <button class="tab" data-tab="camera">Камера</button>
            <button class="tab" data-tab="servos">Сервоприводы</button>
            <button class="tab" data-tab="light">Подсветка</button>
          </div>
          <div class="panel-body">

            <div class="tab-pane active" data-pane="chat">
              <div class="chat-log" id="chatHistory">
                <div class="chat-empty" id="chatEmpty">
                  <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.4">
                    <path d="M16 7C10 7 5.5 12.2 5.5 18.6 5.5 25.2 10.2 29.5 16 29.5S26.5 25.2 26.5 18.6C26.5 12.2 22 7 16 7Z"/>
                    <circle cx="11.9" cy="18.3" r="3.4"/><circle cx="20.1" cy="18.3" r="3.4"/>
                    <circle cx="11.9" cy="18.3" r="1" fill="currentColor" stroke="none"/>
                    <circle cx="20.1" cy="18.3" r="1" fill="currentColor" stroke="none"/>
                  </svg>
                  <span>Пока тихо. Напишите или скажите что-нибудь Сорену — он слушает.</span>
                </div>
              </div>
              <canvas id="audioVisualizer" class="audio-visualizer" style="display:none;"></canvas>
              <div class="chat-input">
                <button class="mic-btn" id="micBtn" title="Голосовой ввод" onclick="toggleRecording()">
                  <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z"/><path d="M19 11a7 7 0 0 1-14 0M12 19v3"/></svg>
                </button>
                <input type="text" id="chatInput" placeholder="Напишите Сорену…" onkeypress="if(event.key==='Enter') sendChat()">
                <button class="send-btn" onclick="sendChat()">Отправить</button>
              </div>
              <div class="chip-row">
                <button class="chip-btn" onclick="sendCmd({type:'animation',name:'wave'})">Помахать</button>
                <button class="chip-btn" onclick="sendCmd({type:'animation',name:'nod'})">Кивнуть</button>
                <button class="chip-btn" onclick="sendCmd({type:'animation',name:'shake_head'})">Качнуть головой</button>
                <button class="chip-btn" onclick="sendCmd({type:'animation',name:'idle'})">Покой</button>
                <button class="chip-btn chip-danger" onclick="sendCmd({type:'clear_history'})">Очистить историю</button>
              </div>
            </div>

            <div class="tab-pane" data-pane="camera">
              <div class="camera-frame" id="cameraFrame">
                <span class="live-badge" id="liveBadge" style="display:none;">LIVE</span>
                <span id="cameraPlaceholder">Нет видеопотока — ждём кадры с ESP32</span>
                <img id="cameraFeed" style="display:none;">
              </div>
              <div class="grid-2" style="margin-top:12px;">
                <div class="mini-card row-between">
                  <span class="label-sm">Показывать видео</span>
                  <label class="switch"><input type="checkbox" id="videoToggle" checked><span class="track"></span></label>
                </div>
                <div class="mini-card row-between">
                  <span class="label-sm">Кадров/сек</span>
                  <span class="value-lg" id="cameraFpsTag" style="font-size:14px;">—</span>
                </div>
              </div>
              <div class="cam-badges">
                <span class="cam-badge" id="badgeFace"><span class="dot"></span>Лицо не найдено</span>
                <span class="cam-badge" id="badgeTracking"><span class="dot"></span>Слежение выключено</span>
                <span class="cam-badge" id="badgeDialog"><span class="dot"></span>Диалог неактивен</span>
              </div>
            </div>

            <div class="tab-pane" data-pane="servos">
              <div class="servo-grid" id="servoGrid"></div>
            </div>

            <div class="tab-pane" data-pane="light">
              <div class="grid-2">
                <div class="mini-card row-between">
                  <span class="label-sm">Авто (закат–рассвет)</span>
                  <label class="switch"><input type="checkbox" id="backlightAutoToggle" onchange="setBacklightMode('auto', this.checked)"><span class="track"></span></label>
                </div>
                <div class="mini-card row-between">
                  <span class="label-sm">Вручную</span>
                  <label class="switch"><input type="checkbox" id="backlightManualToggle" onchange="setBacklightMode('manual', this.checked)"><span class="track"></span></label>
                </div>
              </div>
              <div class="mini-card" style="margin-top:12px;">
                <div class="row-between">
                  <span class="label-sm">Сейчас</span>
                  <span class="value-lg" id="backlightEffectiveTag" style="font-size:16px;">—</span>
                </div>
              </div>
              <p class="hint-text">Ручной тумблер работает, только пока авто-режим выключен. Координаты для авто — из config.yaml (backlight).</p>
            </div>

          </div>
        </section>

        <!-- ============ НАСТРОЙКИ ============ -->
        <section class="panel panel-settings" id="panelSettings">
          <div class="panel-head">
            <div class="title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1 1.55V21a2 2 0 0 1-4 0v-.09A1.7 1.7 0 0 0 9 19.4a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.55-1H3a2 2 0 0 1 0-4h.09A1.7 1.7 0 0 0 4.6 9a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.55V3a2 2 0 0 1 4 0v.09a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.7 1.7 0 0 0 19.4 9a1.7 1.7 0 0 0 1.55 1H21a2 2 0 0 1 0 4h-.09a1.7 1.7 0 0 0-1.51 1Z"/></svg>
              Настройки
            </div>
            <button class="collapse-btn" id="collapseBtn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M18 15l-6-6-6 6"/></svg>
            </button>
          </div>
          <div class="tabs" data-group="settings">
            <button class="tab active" data-tab="model">Модель ИИ</button>
            <button class="tab" data-tab="memory">Память</button>
            <button class="tab" data-tab="audio">Вывод звука</button>
            <button class="tab" data-tab="logs">Журнал</button>
          </div>
          <div class="panel-body">

            <div class="tab-pane active" data-pane="model">
              <div class="mini-card ai-core-card">
                <div class="ai-core-col">
                  <div class="label-sm" style="margin-bottom:6px;">Распознавание речи (STT)</div>
                  <div class="segmented" id="sttSegmented">
                    <button id="sttLocalBtn" onclick="setAIMode('stt','local')">Локально</button>
                    <button id="sttCloudBtn" onclick="setAIMode('stt','cloud')">Облако</button>
                  </div>
                  <select class="select-line" id="sttModelSelect" onchange="setSTTModel(this.value)" style="margin-top:6px;"></select>
                </div>
                <div class="ai-core-col">
                  <div class="label-sm" style="margin-bottom:6px;">Диалог (LLM)</div>
                  <div class="segmented" id="llmSegmented">
                    <button id="llmLocalBtn" onclick="setAIMode('llm','local')">Локально</button>
                    <button id="llmCloudBtn" onclick="setAIMode('llm','cloud')">Облако</button>
                  </div>
                  <select class="select-line" id="llmModelSelect" onchange="setLLMModel(this.value)" style="margin-top:6px;"></select>
                </div>
                <div class="ai-core-col">
                  <div class="label-sm" style="margin-bottom:6px;">Синтез речи (TTS)</div>
                  <div class="segmented" id="ttsSegmented">
                    <button id="ttsLocalBtn" onclick="setAIMode('tts','local')">Локально</button>
                    <button id="ttsCloudBtn" onclick="setAIMode('tts','cloud')">Облако</button>
                  </div>
                  <select class="select-line" id="ttsModelSelect" onchange="setTTSSpeaker(this.value)" style="margin-top:6px;"></select>
                </div>
              </div>
              <div class="mini-card row-between" style="margin-top:8px;">
                <span class="label-sm">Быстрые ответы — <span id="qaCountText">—</span></span>
                <button class="chip-btn" onclick="reloadQuickAnswers()">Обновить</button>
              </div>
            </div>

            <div class="tab-pane" data-pane="memory">
              <div class="grid-2">
                <div class="mini-card row-between"><span class="label-sm">Краткая память (STM)</span><label class="switch"><input type="checkbox" id="memStmToggle" onchange="toggleMemoryLevel('stm')"><span class="track"></span></label></div>
                <div class="mini-card row-between"><span class="label-sm">Долгая память (LTM)</span><label class="switch"><input type="checkbox" id="memLtmToggle" onchange="toggleMemoryLevel('ltm')"><span class="track"></span></label></div>
                <div class="mini-card row-between"><span class="label-sm">Эмоц. профиль</span><label class="switch"><input type="checkbox" id="memProfileToggle" onchange="toggleMemoryLevel('profile')"><span class="track"></span></label></div>
                <div class="mini-card row-between"><span class="label-sm">RAG канона</span><label class="switch"><input type="checkbox" id="memRagToggle" onchange="toggleMemoryLevel('rag')"><span class="track"></span></label></div>
              </div>
              <p class="hint-text">Изменения применяются сразу и сохраняются на диск — переживают перезапуск сервера.</p>
            </div>

            <div class="tab-pane" data-pane="audio">
              <div class="mini-card row-between" style="margin-bottom:12px;">
                <span class="label-sm">Микрофон (ввод)</span>
                <div class="row-between" style="gap:10px;">
                  <span class="label-sm" id="audioInputModeText">Робот · ESP32</span>
                  <label class="switch"><input type="checkbox" id="audioInputToggle" onchange="toggleAudioInputMode()"><span class="track"></span></label>
                </div>
              </div>
              <div class="mini-card row-between">
                <span class="label-sm">Динамик (вывод)</span>
                <div class="row-between" style="gap:10px;">
                  <span class="label-sm" id="audioOutputModeText">Робот · ESP32</span>
                  <label class="switch"><input type="checkbox" id="audioOutputToggle" onchange="toggleAudioOutputMode()"><span class="track"></span></label>
                </div>
              </div>
            </div>

            <div class="tab-pane" data-pane="logs">
              <div class="log-list" id="logList"></div>
            </div>

          </div>
        </section>

      </div>
    </div>
  </div>

  <audio id="audioPlayer" style="display:none;"></audio>

<script>
  // ==================== ТЕМА: авто (по часам устройства) + ручной day/night ====================
  // По умолчанию — режим "авто": ночная тема с 20:00 до 7:00 по локальному
  // времени устройства (без геолокации — запрос разрешения на каждой
  // перезагрузке раздражает, а часы устройства всегда доступны сразу).
  // Кнопка в шапке даёт только явный выбор "день"/"ночь": один клик один раз
  // переключает с текущего эффективного состояния на противоположное и
  // дальше держит его вручную, до перезагрузки страницы — обратно в "авто"
  // кнопка не возвращает.
  const NIGHT_START_HOUR = 20;
  const NIGHT_END_HOUR = 7;

  let themeMode = 'auto'; // 'auto' | 'light' | 'dark' — 'auto' достижим только программно при загрузке

  function isNightNow(){
    const h = new Date().getHours();
    return h >= NIGHT_START_HOUR || h < NIGHT_END_HOUR;
  }

  function applyTheme(){
    const effective = themeMode === 'auto' ? (isNightNow() ? 'dark' : 'light') : themeMode;
    document.documentElement.setAttribute('data-theme', effective);

    const modeLabel = effective === 'dark' ? 'ночная' : 'дневная';
    document.getElementById('themeToggle').title =
      themeMode === 'auto' ? `Тема: авто (сейчас ${modeLabel})` : `Тема: ${modeLabel}`;
  }

  function toggleTheme(){
    // Ручной выбор — только день/ночь; к "авто" кнопка больше не возвращает.
    const current = themeMode === 'auto' ? (isNightNow() ? 'dark' : 'light') : themeMode;
    themeMode = current === 'dark' ? 'light' : 'dark';
    applyTheme();
  }

  applyTheme();
  setInterval(()=>{ if (themeMode === 'auto') applyTheme(); }, 60000);

  // --- переключение табов внутри группы ---
  document.querySelectorAll('.tabs').forEach(group=>{
    const panelBody = group.parentElement.querySelector('.panel-body');
    group.addEventListener('click', e=>{
      const btn = e.target.closest('.tab');
      if(!btn) return;
      group.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
      btn.classList.add('active');
      panelBody.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
      panelBody.querySelector(`[data-pane="${btn.dataset.tab}"]`).classList.add('active');
    });
  });

  // --- сворачивание блока "Настройки" ---
  const panelSettings = document.getElementById('panelSettings');
  const panelControl = document.getElementById('panelControl');
  document.getElementById('collapseBtn').addEventListener('click', ()=>{
    panelSettings.classList.toggle('collapsed');
    panelControl.classList.toggle('expanded');
  });

  // ==================== СВЯЗЬ С СЕРВЕРОМ ====================
  const ws = new WebSocket(`ws://${window.location.host}/ws`);

  let audioInputMode = 'robot';
  let audioOutputMode = 'robot';
  let aiModes = { stt: 'local', tts: 'local', llm: 'local' };
  let memoryFlags = { stm: true, ltm: true, profile: true, rag: true };
  let modelConfig = { llm: {mode:'local', current:null, local_models:[], cloud_models:[]}, stt: {current:null, models:[]}, tts: {mode:'local', current:null, speakers:[]} };
  let quickAnswersStatus = { enabled: true, count: 0 };
  let backlightState = { auto: false, manual: false, effective: false };

  // "Connection open/closed" — собственный WS-канал этой панели.
  // "Online/Offline" — подключено ли к серверу само устройство (ESP32).
  function setConnection(open){
    const el = document.getElementById('connectionIndicator');
    document.getElementById('connectionLabel').textContent = open ? 'Connection open' : 'Connection closed';
    el.classList.toggle('offline', !open);
  }
  function setStatus(online, deviceCount){
    const el = document.getElementById('statusIndicator');
    document.getElementById('statusLabel').textContent = online ? 'Online' : 'Offline';
    el.classList.toggle('offline', !online);
    el.title = typeof deviceCount === 'number' ? `Устройств подключено: ${deviceCount}` : '';
  }

  ws.onopen = () => {
    setConnection(true);
    log('WebSocket подключен');
    ws.send(JSON.stringify({type:'hello', client:'panel'}));
    ws.send(JSON.stringify({type:'audio_mode'}));
    ws.send(JSON.stringify({type:'ai_mode'}));
    ws.send(JSON.stringify({type:'video_pref', enabled: videoPrefEnabled}));
    fetchStatus();
  };
  ws.onclose = () => {
    setConnection(false);
    setStatus(false, 0);
    log('WebSocket отключен');
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'video_frame') {
      handleVideoFrame(data);
      return; // не спамим журнал каждым кадром
    }
    log('← ' + JSON.stringify(data));
    if (data.angles) updateServoDisplay(data.angles);
    if (data.type === 'audio_mode') {
      if (data.input_mode) audioInputMode = data.input_mode;
      if (data.output_mode) audioOutputMode = data.output_mode;
      updateAudioModeUI();
    }
    if (data.type === 'ai_mode' && data.modes) { aiModes = data.modes; updateAIModeUI(); }
    // Ответ на {type:'text'} при audioOutputMode==='robot' — без поля type,
    // только status/response.
    if (!data.type && data.status === 'ok' && typeof data.response === 'string') {
      addMessage('bot', data.response, data.emotion);
      if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
    }
    if (data.modes && !data.type) { aiModes = data.modes; updateAIModeUI(); }
    if (typeof data.dialog_active === 'boolean') updateDialogBadges(data.face_detected, data.dialog_active);
  };

  async function fetchStatus(){
    try {
      const r = await fetch('/status');
      const data = await r.json();
      if (data.status === 'initializing') return;
      if (data.memory_flags) { memoryFlags = data.memory_flags; updateMemoryFlagsUI(); }
      if (data.model_config) { modelConfig = data.model_config; updateModelSelectsUI(); }
      if (data.quick_answers) { quickAnswersStatus = data.quick_answers; updateQuickAnswersUI(); }
      if (data.backlight) { backlightState = data.backlight; updateBacklightUI(); }
      if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
      if (Array.isArray(data.servo_angles)) updateServoDisplay(data.servo_angles);
      if (typeof data.audio_input_mode === 'string') audioInputMode = data.audio_input_mode;
      if (typeof data.audio_output_mode === 'string') audioOutputMode = data.audio_output_mode;
      updateAudioModeUI();
      setStatus((data.device_connections || 0) > 0, data.device_connections || 0);
    } catch(e) { log('Не удалось получить статус сервера: ' + e.message); }
  }

  function sendCmd(cmd){ ws.send(JSON.stringify(cmd)); log('→ ' + JSON.stringify(cmd)); }

  function log(msg){
    const el = document.getElementById('logList');
    if (!el) return;
    const row = document.createElement('div');
    row.className = 'log-item';
    row.innerHTML = `<span class="t">${new Date().toLocaleTimeString()}</span><span>${msg.length > 220 ? msg.slice(0,220)+'…' : msg}</span>`;
    el.appendChild(row);
    while (el.children.length > 200) el.removeChild(el.firstChild); // не даём журналу расти бесконечно
    el.scrollTop = el.scrollHeight;
  }

  // ==================== ЧАТ ====================
  function addMessage(sender, text, emotion){
    const chat = document.getElementById('chatHistory');
    const empty = document.getElementById('chatEmpty');
    if (empty) empty.remove();
    const div = document.createElement('div');
    div.className = 'msg ' + (sender === 'user' ? 'user' : 'bot');
    div.textContent = text + (emotion ? ` (${emotion})` : '');
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  async function sendChat(){
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage('user', text);
    if (audioOutputMode === 'robot') sendCmd({type:'text', text});
    else await sendLocal(text);
  }

  async function sendLocal(text){
    try {
      const fd = new FormData(); fd.append('text', text);
      const r = await fetch('/speak', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        addMessage('bot', data.response, data.emotion);
        if (data.audio_base64) playAudio(data.audio_base64);
        else if (data.tts_failed) log('⚠ Синтез речи не удался — ответ показан без озвучки');
        if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
      } else addMessage('bot', 'Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Ошибка сети: ' + e.message); }
  }

  function playAudio(b64){
    const audio = document.getElementById('audioPlayer');
    audio.src = 'data:audio/wav;base64,' + b64;
    audio.play().catch(e => log('Ошибка воспроизведения: ' + e.message));
  }

  // ==================== ГОЛОСОВОЙ ВВОД ====================
  let mediaRecorder=null, audioChunks=[], isRecording=false, audioCtx=null, analyser=null, visCanvas=null, visCtx=null;

  async function toggleRecording(){
    const btn = document.getElementById('micBtn');
    const visualizer = document.getElementById('audioVisualizer');
    if (!isRecording){
      try {
        const stream = await navigator.mediaDevices.getUserMedia({audio:true});
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
          const blob = new Blob(audioChunks, {type:'audio/wav'});
          await sendVoiceToServer(blob);
          stream.getTracks().forEach(t => t.stop());
        };
        mediaRecorder.start();
        isRecording = true;
        btn.classList.add('recording');
        visualizer.style.display = 'block';
        setupVisualizer(stream);
        log('Начало записи голоса');
      } catch(err){
        log('Ошибка доступа к микрофону: ' + err.message);
        alert('Разрешите доступ к микрофону в настройках браузера');
      }
    } else {
      mediaRecorder.stop();
      isRecording = false;
      btn.classList.remove('recording');
      visualizer.style.display = 'none';
      log('Конец записи, отправка…');
    }
  }

  function setupVisualizer(stream){
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    analyser.fftSize = 256;
    visCanvas = document.getElementById('audioVisualizer');
    visCtx = visCanvas.getContext('2d');
    visCanvas.width = visCanvas.offsetWidth;
    visCanvas.height = visCanvas.offsetHeight;
    drawVisualizer();
  }
  function drawVisualizer(){
    if (!isRecording || !analyser) return;
    requestAnimationFrame(drawVisualizer);
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    visCtx.clearRect(0,0,visCanvas.width, visCanvas.height);
    const barWidth = (visCanvas.width / bufferLength) * 2.5;
    let x = 0;
    for (let i=0;i<bufferLength;i++){
      const barHeight = dataArray[i] / 3;
      visCtx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--amber') || '#B8875A';
      visCtx.fillRect(x, visCanvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }

  async function sendVoiceToServer(blob){
    const fd = new FormData();
    fd.append('audio', blob, 'voice.wav');
    fd.append('audio_output_mode_param', audioOutputMode);
    try {
      const r = await fetch('/voice', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        addMessage('user', data.user_text);
        addMessage('bot', data.response, data.emotion);
        if (audioOutputMode === 'local' && data.audio_base64) playAudio(data.audio_base64);
        if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
      } else {
        addMessage('bot', 'Ошибка: ' + (data.message || 'неизвестная'));
      }
    } catch(e) { log('Ошибка сети (голос): ' + e.message); }
  }

  // ==================== КАМЕРА ====================
  const VIDEO_PREF_KEY = 'soren_panel_show_video';
  function getVideoPref(){ const v = localStorage.getItem(VIDEO_PREF_KEY); return v === null ? true : v === '1'; }
  function setVideoPref(enabled){ localStorage.setItem(VIDEO_PREF_KEY, enabled ? '1' : '0'); }
  let videoPrefEnabled = getVideoPref();
  let hasReceivedFrame = false;
  let frameCount = 0, fpsWindowStart = Date.now();

  function updateCameraFrameVisibility(){
    const img = document.getElementById('cameraFeed');
    const placeholder = document.getElementById('cameraPlaceholder');
    const liveBadge = document.getElementById('liveBadge');
    if (!videoPrefEnabled){
      img.style.display = 'none'; liveBadge.style.display = 'none';
      placeholder.style.display = 'block';
      placeholder.textContent = 'Видео отключено — слежение продолжает работать';
    } else if (hasReceivedFrame){
      img.style.display = 'block'; liveBadge.style.display = 'flex';
      placeholder.style.display = 'none';
    } else {
      img.style.display = 'none'; liveBadge.style.display = 'none';
      placeholder.style.display = 'block';
      placeholder.textContent = 'Нет видеопотока — ждём кадры с ESP32';
    }
  }

  const videoToggle = document.getElementById('videoToggle');
  videoToggle.checked = videoPrefEnabled;
  updateCameraFrameVisibility();
  videoToggle.addEventListener('change', () => {
    videoPrefEnabled = videoToggle.checked;
    setVideoPref(videoPrefEnabled);
    updateCameraFrameVisibility();
    if (ws.readyState === WebSocket.OPEN) sendCmd({type:'video_pref', enabled: videoPrefEnabled});
  });

  function handleVideoFrame(data){
    document.getElementById('cameraFeed').src = 'data:image/jpeg;base64,' + data.image;
    hasReceivedFrame = true;
    updateCameraFrameVisibility();
    updateDialogBadges(data.face_detected, data.dialog_active);

    frameCount++;
    const now = Date.now();
    if (now - fpsWindowStart >= 1000){
      const facesTxt = data.faces_count > 1 ? ` (лиц: ${data.faces_count})` : '';
      document.getElementById('cameraFpsTag').textContent = frameCount + facesTxt;
      frameCount = 0; fpsWindowStart = now;
    }
  }

  function updateDialogBadges(faceDetected, dialogActive){
    const faceBadge = document.getElementById('badgeFace');
    const trackBadge = document.getElementById('badgeTracking');
    const dialogBadge = document.getElementById('badgeDialog');
    faceBadge.classList.toggle('on', !!faceDetected);
    faceBadge.lastChild.textContent = faceDetected ? 'Лицо найдено' : 'Лицо не найдено';
    const tracking = !!faceDetected && !!dialogActive;
    trackBadge.classList.toggle('on', tracking);
    trackBadge.lastChild.textContent = tracking ? 'Слежение активно' : 'Слежение выключено';
    dialogBadge.classList.toggle('on', !!dialogActive);
    dialogBadge.lastChild.textContent = dialogActive ? 'Диалог активен' : 'Диалог неактивен';
  }

  // ==================== AI CORE (STT/LLM/TTS) ====================
  async function setAIMode(module, mode){
    log(`Переключение ${module.toUpperCase()} → ${mode}…`);
    const fd = new FormData(); fd.append('module', module); fd.append('mode', mode);
    try {
      const r = await fetch('/ai_mode', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok'){
        if (data.modes) { aiModes = data.modes; updateAIModeUI(); }
        log(`${module.toUpperCase()} → ${mode} ✓`);
        if (module === 'llm' || module === 'tts') fetchStatus();
      } else {
        log('Ошибка: ' + (data.message || 'неизвестная'));
        if (data.modes) { aiModes = data.modes; updateAIModeUI(); }
      }
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  function updateAIModeUI(){
    ['stt','llm','tts'].forEach(mod => {
      const mode = aiModes[mod] || 'local';
      const localBtn = document.getElementById(mod+'LocalBtn');
      const cloudBtn = document.getElementById(mod+'CloudBtn');
      if (localBtn) localBtn.classList.toggle('active', mode === 'local');
      if (cloudBtn) cloudBtn.classList.toggle('active', mode === 'cloud');
    });
    updateModelSelectsUI();
  }

  function _fillSelect(selectEl, options, currentId, placeholder){
    if (!selectEl) return;
    selectEl.innerHTML = '';
    if (!options || !options.length){
      const opt = document.createElement('option');
      opt.value = ''; opt.textContent = placeholder || '— нет вариантов —';
      selectEl.appendChild(opt);
      selectEl.disabled = true;
      return;
    }
    selectEl.disabled = false;
    let matched = false;
    options.forEach(o => {
      const opt = document.createElement('option');
      opt.value = o.id; opt.textContent = o.label || o.id;
      if (o.id === currentId) { opt.selected = true; matched = true; }
      selectEl.appendChild(opt);
    });
    if (!matched && currentId){
      const opt = document.createElement('option');
      opt.value = currentId; opt.textContent = currentId + ' (вручную из config.yaml)';
      opt.selected = true;
      selectEl.insertBefore(opt, selectEl.firstChild);
    }
  }

  function updateModelSelectsUI(){
    const llmMode = (aiModes.llm || modelConfig.llm.mode || 'local');
    const llmOptions = llmMode === 'cloud' ? modelConfig.llm.cloud_models : modelConfig.llm.local_models;
    _fillSelect(document.getElementById('llmModelSelect'), llmOptions, modelConfig.llm.current, 'Список моделей пуст — задайте llm.local_models/cloud_models в config.yaml');
    _fillSelect(document.getElementById('sttModelSelect'), modelConfig.stt.models, modelConfig.stt.current, 'Список пуст — задайте stt.whisper_models в config.yaml');
    const ttsSpeakers = (modelConfig.tts.speakers || []).map(s => ({id:s, label:s}));
    _fillSelect(document.getElementById('ttsModelSelect'), ttsSpeakers, modelConfig.tts.current, 'Нет доступных голосов');
  }

  async function setLLMModel(modelId){
    if (!modelId) return;
    log(`Модель LLM → ${modelId}…`);
    const fd = new FormData(); fd.append('model_id', modelId);
    try {
      const r = await fetch('/llm_model', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok'){
        if (data.model_config) modelConfig = data.model_config;
        aiModes.llm = modelConfig.llm.mode; updateAIModeUI(); updateModelSelectsUI();
        log(`LLM модель → ${modelId} ✓ (режим: ${modelConfig.llm.mode})`);
      } else log('Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  async function setSTTModel(modelId){
    if (!modelId) return;
    log(`Модель STT (Whisper) → ${modelId}…`);
    const fd = new FormData(); fd.append('model_id', modelId);
    try {
      const r = await fetch('/stt_model', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok'){
        if (data.model_config) modelConfig = data.model_config;
        log(`STT модель → ${modelId} ✓`);
      } else log('Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  async function setTTSSpeaker(speaker){
    if (!speaker) return;
    log(`Голос TTS → ${speaker}…`);
    const fd = new FormData(); fd.append('speaker', speaker);
    try {
      const r = await fetch('/tts_speaker', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok'){
        if (data.model_config) modelConfig = data.model_config;
        log(`Голос TTS → ${speaker} ✓`);
      } else log('Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  async function reloadQuickAnswers(){
    log('Перезагружаю словарь быстрых ответов…');
    try {
      const r = await fetch('/quick_answers/reload', {method:'POST'});
      const data = await r.json();
      if (data.status === 'ok'){
        quickAnswersStatus = data.quick_answers;
        updateQuickAnswersUI();
        log(`Словарь быстрых ответов обновлён ✓ (${quickAnswersStatus.count} записей)`);
      } else log('Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }
  function updateQuickAnswersUI(){
    const txt = document.getElementById('qaCountText');
    if (!txt) return;
    txt.textContent = quickAnswersStatus.enabled ? `${quickAnswersStatus.count} записей` : 'выключено (config.yaml)';
  }

  // ==================== АУДИО-МАРШРУТИЗАЦИЯ ====================
  async function toggleAudioInputMode(){
    const newMode = document.getElementById('audioInputToggle').checked ? 'local' : 'robot';
    const fd = new FormData(); fd.append('mode', newMode); fd.append('type', 'input');
    const r = await fetch('/audio_mode', {method:'POST', body:fd});
    const data = await r.json();
    if (data.status === 'ok') { audioInputMode = data.audio_input_mode; updateAudioModeUI(); }
  }
  async function toggleAudioOutputMode(){
    const newMode = document.getElementById('audioOutputToggle').checked ? 'local' : 'robot';
    const fd = new FormData(); fd.append('mode', newMode); fd.append('type', 'output');
    const r = await fetch('/audio_mode', {method:'POST', body:fd});
    const data = await r.json();
    if (data.status === 'ok') { audioOutputMode = data.audio_output_mode; updateAudioModeUI(); }
  }
  function updateAudioModeUI(){
    const it = document.getElementById('audioInputToggle');
    it.checked = (audioInputMode === 'local');
    document.getElementById('audioInputModeText').textContent = audioInputMode === 'local' ? 'Локально · микрофон ПК' : 'Робот · ESP32';
    const ot = document.getElementById('audioOutputToggle');
    ot.checked = (audioOutputMode === 'local');
    document.getElementById('audioOutputModeText').textContent = audioOutputMode === 'local' ? 'Локально · наушники ПК' : 'Робот · ESP32';
  }

  // ==================== ПАМЯТЬ ====================
  async function toggleMemoryLevel(level){
    const enabled = document.getElementById('mem' + level[0].toUpperCase()+level.slice(1) + 'Toggle').checked;
    log(`Уровень памяти '${level}' → ${enabled ? 'вкл' : 'выкл'}…`);
    const fd = new FormData(); fd.append('level', level); fd.append('enabled', enabled ? '1' : '0');
    try {
      const r = await fetch('/memory_config', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok'){
        if (data.memory_flags) memoryFlags = data.memory_flags;
        log(`Память '${level}' → ${enabled ? 'вкл' : 'выкл'} ✓`);
      } else log('Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
    updateMemoryFlagsUI();
  }
  function updateMemoryFlagsUI(){
    ['Stm','Ltm','Profile','Rag'].forEach(level => {
      const key = level.toLowerCase();
      const box = document.getElementById('mem' + level + 'Toggle');
      if (box) box.checked = !!memoryFlags[key];
    });
  }

  // ==================== ПОДСВЕТКА ПОДСТАВКИ ====================
  function updateBacklightUI(){
    const autoBox = document.getElementById('backlightAutoToggle');
    const manualBox = document.getElementById('backlightManualToggle');
    const tag = document.getElementById('backlightEffectiveTag');
    if (autoBox) autoBox.checked = !!backlightState.auto;
    if (manualBox){
      manualBox.checked = !!backlightState.manual;
      manualBox.disabled = !!backlightState.auto;
    }
    if (tag) tag.textContent = backlightState.effective ? 'горит' : 'выкл';
  }
  async function setBacklightMode(mode, enabled){
    log(`Подсветка: ${mode} → ${enabled ? 'вкл' : 'выкл'}…`);
    const fd = new FormData(); fd.append('mode', mode); fd.append('enabled', enabled ? '1' : '0');
    try {
      const r = await fetch('/backlight', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok'){
        backlightState = {...data.backlight, effective: data.effective};
        updateBacklightUI();
        log(`Подсветка обновлена ✓ (${backlightState.effective ? 'горит' : 'выкл'})`);
      } else log('Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  // ==================== СЕРВОПРИВОДЫ ====================
  const servoGridEl = document.getElementById('servoGrid');
  for (let i=0;i<18;i++) servoGridEl.appendChild(makeServo(i));
  function makeServo(i){
    const div = document.createElement('div');
    div.className = 'servo-card';
    div.innerHTML = `
      <div class="row-between">
        <span class="ch">CH ${String(i).padStart(2,'0')}</span>
        <span class="deg" id="val-${i}">90°</span>
      </div>
      <input type="range" id="servo-${i}" min="0" max="180" value="90" oninput="onServoSlide(${i}, this.value)">
    `;
    return div;
  }
  // Троттлинг на слайдер: угол шлётся сразу, но не чаще раза в SERVO_SLIDER_THROTTLE_MS —
  // иначе перетаскивание ползунка заваливает сокет очередью устаревших команд.
  // Финальное значение при отпускании досылается гарантированно отдельным таймером.
  const SERVO_SLIDER_THROTTLE_MS = 60;
  const _servoSlideState = {};
  function onServoSlide(i, value){
    const angle = parseInt(value);
    document.getElementById(`val-${i}`).textContent = angle + '°';
    let st = _servoSlideState[i];
    if (!st) st = _servoSlideState[i] = { lastSentTs: 0, pendingTimeout: null, pendingAngle: null };
    const sendNow = () => { st.lastSentTs = Date.now(); st.pendingAngle = null; sendCmd({type:'servo', id:i, angle}); };
    const sinceLast = Date.now() - st.lastSentTs;
    if (sinceLast >= SERVO_SLIDER_THROTTLE_MS){
      if (st.pendingTimeout) { clearTimeout(st.pendingTimeout); st.pendingTimeout = null; }
      sendNow();
    } else {
      st.pendingAngle = angle;
      if (!st.pendingTimeout){
        st.pendingTimeout = setTimeout(() => {
          st.pendingTimeout = null;
          if (st.pendingAngle !== null){
            const finalAngle = st.pendingAngle;
            st.lastSentTs = Date.now(); st.pendingAngle = null;
            sendCmd({type:'servo', id:i, angle:finalAngle});
          }
        }, SERVO_SLIDER_THROTTLE_MS - sinceLast);
      }
    }
  }
  function updateServoDisplay(angles){
    for (let i=0;i<angles.length;i++){
      const slider = document.getElementById(`servo-${i}`);
      const val = document.getElementById(`val-${i}`);
      if (slider && val) { slider.value = angles[i]; val.textContent = angles[i] + '°'; }
    }
  }
</script>
</body>
</html>

"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)