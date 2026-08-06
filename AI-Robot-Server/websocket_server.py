"""WebSocket сервер - связь с ESP32 и веб-интерфейсом + голосовое общение"""
import os
import asyncio
import json
import logging
import base64
import io
import wave
import tempfile
from typing import Set, Optional, Dict, Tuple, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from modules.robot_brain import RobotBrain
from modules.auth import create_session_token, verify_session_token, is_valid_device_ping
from config.settings import (
    SERVER_HOST, SERVER_PORT,
    VIDEO_PANEL_MIN_INTERVAL_SEC, VIDEO_PANEL_JPEG_QUALITY,
    SERVO_UPDATE_MIN_INTERVAL_SEC,
    PANEL_PASSWORD, DEVICE_KEY,
    SESSION_COOKIE_NAME, SESSION_SHORT_MAX_AGE_SEC, SESSION_REMEMBER_MAX_AGE_SEC,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot AI Server - Soren", version="3.0")

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

        robot_brain.mark_dialog_active()

        llm_result = robot_brain.llm.generate(user_text, robot_brain.vision_context)
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
            asyncio.create_task(robot_brain.servos.play_animation(action))
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
            "text": user_text,
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

        robot_brain.mark_dialog_active()

        llm_result = robot_brain.llm.generate(user_text, robot_brain.vision_context)
        response_text = llm_result.get("text", "")
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")

        servo_angles = robot_brain.emotion_engine.get_servo_angles(emotion)
        eye_led = robot_brain.emotion_engine.get_eye_led(emotion)

        tts_audio = robot_brain.tts.synthesize(response_text)

        if action:
            asyncio.create_task(robot_brain.servos.play_animation(action))
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
            "user_text": user_text,
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
                    audio_packet = b"AUDI" + result["audio"]
                    await websocket.send_bytes(audio_packet)

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
    servo_angles = vision_result["servo_angles"]
    now = asyncio.get_event_loop().time()
    last_send_ts, last_sent_angles = _last_servo_send.get(websocket, (None, None))

    if last_send_ts is not None and (now - last_send_ts) < SERVO_UPDATE_MIN_INTERVAL_SEC:
        return

    if last_sent_angles is None:
        # Первая отправка этому устройству — шлём всё, дальше только дельты
        delta = {str(i): a for i, a in enumerate(servo_angles)}
    else:
        delta = {str(i): a for i, a in enumerate(servo_angles) if a != last_sent_angles[i]}

    if not delta:
        # Ничего не изменилось — не спамим сеть, но время последней проверки обновляем
        _last_servo_send[websocket] = (now, last_sent_angles)
        return

    servo_cmd = {
        "type": "servo_update",
        "angles": delta,
        "face_detected": vision_result["face_detected"],
        "face_offset": vision_result["face_offset"],
        "dialog_active": vision_result.get("dialog_active", False)
    }
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
<meta charset="utf-8">
<title>Soren — Instrument Panel</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0b0d0b;
    --panel:rgba(28,34,29,0.6);
    --panel-solid:#161c18;
    --panel-alt:rgba(255,255,255,0.035);
    --raised:rgba(255,255,255,0.06);
    --border:rgba(255,255,255,0.09);
    --border-soft:rgba(255,255,255,0.055);
    --text:#eeece5;
    --text-dim:#9ba5a0;
    --text-faint:#666f68;
    --amber:#e0b94a;
    --amber-soft:#8a6f2a;
    --sage:#8bb385;
    --slate:#6fa3c7;
    --rust:#d97a63;
    --radius:16px;
    --radius-sm:10px;
    --shadow:0 14px 40px rgba(0,0,0,0.32);
  }
  :root.light{
    --bg:#f2efe7;
    --panel:rgba(255,255,255,0.72);
    --panel-solid:#ffffff;
    --panel-alt:rgba(0,0,0,0.03);
    --raised:rgba(0,0,0,0.05);
    --border:rgba(0,0,0,0.09);
    --border-soft:rgba(0,0,0,0.06);
    --text:#1c1e1b;
    --text-dim:#52584f;
    --text-faint:#8a8f87;
    --amber:#a9790b;
    --amber-soft:#d4a537;
    --sage:#3d7a3d;
    --slate:#2a5a7a;
    --rust:#a04030;
    --shadow:0 14px 40px rgba(30,30,20,0.08);
  }
  body.light{ background:var(--bg); color:var(--text); }
  *{box-sizing:border-box;}
  body{
    margin:0; min-height:100vh; color:var(--text);
    font-family:'IBM Plex Sans', sans-serif;
    padding:32px 24px 60px; -webkit-font-smoothing:antialiased;
    background:
      radial-gradient(1100px 560px at 12% -8%, rgba(224,185,74,0.10), transparent 55%),
      radial-gradient(1000px 620px at 100% 15%, rgba(111,163,199,0.08), transparent 55%),
      radial-gradient(900px 700px at 40% 120%, rgba(139,179,133,0.08), transparent 60%),
      var(--bg);
    transition:background .25s;
  }
  .wrap{ max-width:1360px; margin:0 auto; }

  @keyframes fadeUp{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }

  /* ===== NAMEPLATE ===== */
  .nameplate{
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;
    border:1px solid var(--border); border-radius:20px;
    padding:20px 26px; margin-bottom:24px;
    background:var(--panel);
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,0.05);
    position:relative; overflow:hidden;
    animation:fadeUp .45s cubic-bezier(.2,.8,.2,1);
  }
  .nameplate::before{
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg, transparent, var(--amber), var(--sage), transparent);
    opacity:.65; pointer-events:none;
  }
  .brand{ display:flex; align-items:center; gap:16px; }
  .eye{ width:40px; height:40px; flex-shrink:0; filter:drop-shadow(0 0 8px rgba(224,185,74,0.35)); }
  .brand-text h1{
    font-family:'Spectral', serif; font-weight:600; font-size:27px;
    margin:0; letter-spacing:.4px; color:var(--text);
  }
  .brand-text .subtitle{
    font-family:'IBM Plex Mono', monospace; font-size:10.5px;
    letter-spacing:2px; color:var(--text-faint); text-transform:uppercase;
    margin-top:4px;
  }
  .nameplate-meta{ display:flex; align-items:center; gap:22px; flex-wrap:wrap; }
  .meta-item{ text-align:right; }
  .meta-item .label{
    font-family:'IBM Plex Mono', monospace; font-size:9.5px; letter-spacing:1.5px;
    color:var(--text-faint); text-transform:uppercase; display:block; margin-bottom:4px;
  }
  .meta-item .value{ font-family:'IBM Plex Mono', monospace; font-size:13.5px; font-weight:500; }
  .status-dot{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:7px; background:var(--rust); box-shadow:0 0 8px var(--rust); }
  .status-dot.online{ background:var(--sage); box-shadow:0 0 8px var(--sage); animation:pulseDot 2s infinite; }
  @keyframes pulseDot{ 0%,100%{ opacity:1; } 50%{ opacity:.55; } }

  /* ===== SECTION FRAME ===== */
  .section{
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--panel); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
    padding:22px 24px; margin-bottom:18px;
    box-shadow:var(--shadow), inset 0 1px 0 rgba(255,255,255,0.04);
    position:relative; animation:fadeUp .5s cubic-bezier(.2,.8,.2,1) both;
    transition:border-color .2s, transform .2s;
  }
  .section-head{
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
    margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid var(--border-soft);
  }
  .section-head h2{
    font-family:'IBM Plex Mono', monospace; font-size:11.5px; font-weight:600;
    letter-spacing:2px; text-transform:uppercase; color:var(--text-dim); margin:0;
    display:flex; align-items:center; gap:9px;
  }
  .section-head h2::before{
    content:""; width:6px; height:6px; border-radius:50%; background:var(--amber);
    box-shadow:0 0 6px var(--amber); flex-shrink:0;
  }
  .section-head .tag{
    font-family:'IBM Plex Mono', monospace; font-size:10px; color:var(--text-faint);
    background:var(--panel-alt); padding:4px 10px; border-radius:20px; border:1px solid var(--border-soft);
  }
  .row2{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }

  /* ===== AI CORE — segmented controls ===== */
  .core-grid{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }
  .core-card{
    border:1px solid var(--border-soft); border-radius:var(--radius-sm); background:var(--panel-alt);
    padding:16px 18px; transition:border-color .2s, transform .2s;
  }
  .core-card:hover{ border-color:var(--border); transform:translateY(-2px); }
  .core-card .name{ font-size:13.5px; font-weight:600; margin-bottom:3px; }
  .core-card .desc{ font-size:11px; color:var(--text-faint); margin-bottom:14px; line-height:1.55; }
  .segmented{
    display:flex; border:1px solid var(--border); border-radius:10px; overflow:hidden;
    font-family:'IBM Plex Mono', monospace; font-size:11.5px; background:var(--panel-solid);
  }
  .segmented button{
    flex:1; border:none; background:transparent; color:var(--text-faint);
    padding:9px 6px; cursor:pointer; letter-spacing:.5px; transition:.15s;
    border-radius:0; font-family:inherit; font-size:inherit; font-weight:inherit;
  }
  .segmented button:first-child{ border-right:1px solid var(--border); }
  .segmented button.active.local{ background:rgba(139,179,133,.18); color:var(--sage); font-weight:600; }
  .segmented button.active.cloud{ background:rgba(111,163,199,.18); color:var(--slate); font-weight:600; }
  .segmented button:hover:not(.active){ color:var(--text-dim); background:var(--raised); }
  .model-select{
    width:100%; margin-top:10px; border:1px solid var(--border); border-radius:8px;
    background:var(--panel-solid); color:var(--text-dim); font-family:'IBM Plex Mono', monospace;
    font-size:11px; padding:7px 8px; cursor:pointer; transition:border-color .15s;
  }
  .model-select:hover{ border-color:var(--border); color:var(--text); }
  .model-select:focus{ outline:none; border-color:var(--sage); }
  .core-note{ font-size:10.5px; color:var(--text-faint); margin-top:16px; text-align:center; border-top:1px solid var(--border-soft); padding-top:12px; }

  /* ===== AUDIO ROUTING ===== */
  .audio-grid{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .audio-row{
    display:flex; align-items:center; justify-content:space-between; gap:14px;
    border:1px solid var(--border-soft); border-radius:var(--radius-sm); background:var(--panel-alt);
    padding:14px 18px; transition:border-color .2s;
  }
  .audio-row:hover{ border-color:var(--border); }
  .audio-row .label{ font-size:12.5px; color:var(--text-dim); }
  .audio-state{ font-family:'IBM Plex Mono', monospace; font-size:11px; }
  .switch{ position:relative; width:46px; height:25px; flex-shrink:0; }
  .switch input{ opacity:0; width:0; height:0; }
  .switch .track{
    position:absolute; inset:0; background:var(--raised); border:1px solid var(--border);
    border-radius:20px; transition:.2s; cursor:pointer;
  }
  .switch .track::before{
    content:""; position:absolute; width:17px; height:17px; left:3px; top:3px;
    background:var(--text-faint); border-radius:50%; transition:.2s cubic-bezier(.2,.8,.2,1);
    box-shadow:0 2px 4px rgba(0,0,0,0.25);
  }
  .switch input:checked + .track{ background:rgba(111,163,199,.2); border-color:var(--slate); }
  .switch input:checked + .track::before{ transform:translateX(21px); background:var(--slate); }

  /* ===== VOICE ===== */
  .voice-panel{ display:flex; align-items:center; gap:24px; padding:6px 4px; }
  .mic-btn{
    width:68px; height:68px; border-radius:50%; flex-shrink:0;
    border:1.5px solid var(--amber-soft); background:var(--panel-alt); color:var(--amber);
    cursor:pointer; display:flex; align-items:center; justify-content:center; transition:.2s;
    box-shadow:0 6px 18px rgba(0,0,0,0.2);
  }
  .mic-btn:hover{ border-color:var(--amber); transform:translateY(-2px); box-shadow:0 8px 22px rgba(224,185,74,0.2); }
  .mic-btn.recording{ background:rgba(217,122,99,.16); border-color:var(--rust); color:var(--rust); animation:breathe 1.4s infinite; }
  @keyframes breathe{ 0%,100%{ box-shadow:0 0 0 0 rgba(217,122,99,.35);} 50%{ box-shadow:0 0 0 10px rgba(217,122,99,0);} }
  .voice-info{ flex:1; min-width:0; }
  .voice-status{ font-size:12.5px; color:var(--text-dim); margin-bottom:9px; }
  .voice-status .rec{ color:var(--rust); font-family:'IBM Plex Mono', monospace; margin-left:8px; display:none; }
  .voice-status .rec.active{ display:inline; }
  canvas.visualizer{ width:100%; height:36px; background:var(--panel-alt); border:1px solid var(--border-soft); border-radius:var(--radius-sm); display:none; }
  canvas.visualizer.active{ display:block; }
  :root.light canvas.visualizer{ background:var(--raised); }

  /* ===== CHAT ===== */
  .chat-log{ max-height:280px; overflow-y:auto; display:flex; flex-direction:column; gap:10px; margin-bottom:14px; padding-right:4px; }
  .msg{
    font-size:12.5px; line-height:1.55; padding:10px 14px; border-radius:14px;
    border:1px solid var(--border-soft); max-width:88%;
  }
  .msg.user{ background:rgba(111,163,199,0.1); border-color:rgba(111,163,199,0.25); align-self:flex-end; border-bottom-right-radius:4px; }
  :root.light .msg.user{ background:rgba(42,90,122,0.06); border-color:rgba(42,90,122,0.18); }
  .msg.robot{ background:rgba(224,185,74,.08); border-color:rgba(224,185,74,.22); align-self:flex-start; border-bottom-left-radius:4px; }
  :root.light .msg.robot{ background:rgba(184,134,11,0.07); border-color:rgba(184,134,11,.2); }
  .msg .who{ font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:1px; color:var(--text-faint); text-transform:uppercase; display:block; margin-bottom:4px; }
  .emotion-tag{ display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:9.5px; letter-spacing:1px; text-transform:uppercase; padding:2px 8px; border-radius:10px; margin-left:8px; border:1px solid; }
  .em-calm{ color:#7fa8c9; border-color:#7fa8c9; }
  .em-sad{ color:#8686b0; border-color:#8686b0; }
  .em-angry{ color:var(--rust); border-color:var(--rust); }
  .em-loving{ color:#d9a04a; border-color:#d9a04a; }
  .em-determined{ color:var(--sage); border-color:var(--sage); }
  .em-surprised{ color:#c17fd9; border-color:#c17fd9; }
  .em-tired{ color:var(--text-faint); border-color:var(--text-faint); }
  .chat-input-row{ display:flex; gap:10px; }
  input[type=text]{
    flex:1; background:var(--panel-alt); border:1px solid var(--border); color:var(--text);
    padding:11px 14px; border-radius:var(--radius-sm); font-family:'IBM Plex Sans', sans-serif; font-size:13px;
    transition:border-color .15s, box-shadow .15s;
  }
  input[type=text]:focus{ outline:none; border-color:var(--amber-soft); box-shadow:0 0 0 3px rgba(224,185,74,0.12); }
  :root.light input[type=text]{ background:var(--raised); }

  button.btn{
    background:var(--panel-alt); border:1px solid var(--border); color:var(--text-dim);
    padding:10px 18px; border-radius:var(--radius-sm); cursor:pointer; font-size:12px;
    font-family:'IBM Plex Sans', sans-serif; font-weight:600; transition:.15s;
  }
  button.btn:hover{ border-color:var(--amber-soft); color:var(--text); transform:translateY(-1px); }
  button.btn.primary{ border-color:var(--amber-soft); color:#171208; background:linear-gradient(180deg, #ecc766, var(--amber)); }
  button.btn.primary:hover{ filter:brightness(1.05); box-shadow:0 6px 16px rgba(224,185,74,0.25); }
  button.btn.danger{ border-color:rgba(217,122,99,.5); color:var(--rust); }
  button.btn.danger:hover{ background:rgba(217,122,99,.1); }

  .gesture-grid{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px; }

  /* ===== CAMERA / FACE TRACKING ===== */
  .camera-frame{
    position:relative; width:100%; max-width:480px; aspect-ratio:4/3; margin:0 auto;
    background:#05070500; background-color:rgba(0,0,0,0.28); border:1px solid var(--border); border-radius:var(--radius-sm);
    overflow:hidden; display:flex; align-items:center; justify-content:center;
  }
  .camera-frame img{ width:100%; height:100%; object-fit:contain; display:block; }
  .camera-placeholder{
    font-family:'IBM Plex Mono', monospace; font-size:11px; letter-spacing:1px;
    color:var(--text-faint); text-transform:uppercase; text-align:center; padding:0 20px;
  }
  .camera-badges{ display:flex; gap:10px; justify-content:center; margin-top:14px; flex-wrap:wrap; }
  .camera-badge{
    font-family:'IBM Plex Mono', monospace; font-size:10.5px; letter-spacing:1px; text-transform:uppercase;
    padding:6px 13px; border-radius:20px; border:1px solid var(--border); color:var(--text-faint);
    display:flex; align-items:center; gap:7px; background:var(--panel-alt); transition:.2s;
  }
  .camera-badge .dot{ width:6px; height:6px; border-radius:50%; background:var(--text-faint); }
  .camera-badge.on{ color:var(--sage); border-color:rgba(139,179,133,.4); }
  .camera-badge.on .dot{ background:var(--sage); box-shadow:0 0 6px var(--sage); }
  .camera-badge.tracking.on{ color:var(--amber); border-color:rgba(224,185,74,.4); }
  .camera-badge.tracking.on .dot{ background:var(--amber); box-shadow:0 0 6px var(--amber); }
  .video-toggle{
    display:flex; align-items:center; gap:7px; font-family:'IBM Plex Mono', monospace;
    font-size:11px; letter-spacing:0.5px; color:var(--text-faint); cursor:pointer; user-select:none;
  }
  .video-toggle input{ cursor:pointer; accent-color:var(--amber); }

  /* ===== SERVOS ===== */
  .servo-block-label{
    font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:1.5px; color:var(--text-faint);
    text-transform:uppercase; margin:16px 0 10px; padding-bottom:8px; border-bottom:1px solid var(--border-soft);
  }
  .servo-block-label:first-child{ margin-top:0; }
  .servo-grid{ display:grid; grid-template-columns:repeat(8, 1fr); gap:10px; }
  .servo{
    border:1px solid var(--border-soft); background:var(--panel-alt); border-radius:var(--radius-sm);
    padding:10px 6px; text-align:center; transition:border-color .2s;
  }
  .servo:hover{ border-color:var(--border); }
  .servo .ch{ font-family:'IBM Plex Mono', monospace; font-size:9.5px; color:var(--text-faint); letter-spacing:.5px; }
  .servo .deg{ font-family:'IBM Plex Mono', monospace; font-size:13px; color:var(--amber); margin:4px 0; font-weight:600; }
  .servo input[type=range]{ width:100%; accent-color:var(--amber); height:14px; cursor:pointer; }
  .servo-actions{ display:flex; gap:10px; margin-top:16px; }

  /* ===== LOG ===== */
  .log{
    background:rgba(0,0,0,0.28); border:1px solid var(--border-soft); border-radius:var(--radius-sm);
    height:210px; overflow-y:auto; padding:12px 14px;
    font-family:'IBM Plex Mono', monospace; font-size:11px; line-height:1.75; color:#a3b3aa;
  }
  :root.light .log{
    background:#ffffff;
    color:#1a1a1a;
    border-color:var(--border);
  }
  .log::-webkit-scrollbar, .chat-log::-webkit-scrollbar{ width:6px; }
  .log::-webkit-scrollbar-thumb, .chat-log::-webkit-scrollbar-thumb{ background:var(--border); border-radius:3px; }
  :root.light .log::-webkit-scrollbar-thumb{ background:var(--border-soft); }

  button.theme-btn{
    display:flex; align-items:center; gap:8px;
    background:var(--panel-alt); border:1px solid var(--border); color:var(--text-dim);
    padding:8px 15px; border-radius:20px; cursor:pointer; font-family:'IBM Plex Sans', sans-serif;
    font-size:12px; font-weight:500; transition:.15s;
    flex:none; letter-spacing:normal;
  }
  button.theme-btn:hover{ border-color:var(--amber-soft); color:var(--amber); background:var(--panel-alt); transform:translateY(-1px); }
  button.theme-btn svg{ flex-shrink:0; }
  .logout-link{
    font-family:'IBM Plex Sans', sans-serif; font-size:12px; color:var(--text-faint);
    text-decoration:none; padding:8px 14px; border-radius:20px; border:1px solid transparent;
    transition:.15s;
  }
  .logout-link:hover{ color:var(--rust); border-color:rgba(217,122,99,.35); background:rgba(217,122,99,.06); }
  @media (max-width: 860px){
    .core-grid{ grid-template-columns:1fr; }
    .audio-grid{ grid-template-columns:1fr; }
    .row2{ grid-template-columns:1fr; }
    .servo-grid{ grid-template-columns:repeat(4,1fr); }
    .nameplate{ flex-direction:column; align-items:flex-start; gap:14px; }
    .nameplate-meta{ gap:16px; }
    body{ padding:20px 14px 48px; }
    .section{ padding:18px 16px; }
  }
</style>
</head>
<body>
<div class="wrap">

  <!-- NAMEPLATE -->
  <div class="nameplate">
    <div class="brand">
      <svg class="eye" viewBox="0 0 40 40" fill="none">
        <circle cx="20" cy="20" r="18.5" stroke="#8a6f2a" stroke-width="1.2"/>
        <circle cx="20" cy="20" r="11" stroke="#d4a537" stroke-width="1.4"/>
        <circle cx="20" cy="20" r="4.2" fill="#d4a537"/>
      </svg>
      <div class="brand-text">
        <h1>Сорен</h1>
        <div class="subtitle">Strigiformes Companion Unit · Server v3.0</div>
      </div>
    </div>
    <div class="nameplate-meta">
      <div class="meta-item">
        <span class="label">Connections</span>
        <span class="value" id="conn-count">0</span>
      </div>
      <div class="meta-item">
        <span class="label">Link</span>
        <span class="value"><span class="status-dot" id="status-dot"></span><span id="status-text">OFFLINE</span></span>
      </div>

      <button class="theme-btn" onclick="toggleTheme()" title="Переключить тему">
        <svg id="theme-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>
        <span id="theme-label">Тёмная</span>
      </button>
      {{LOGOUT_LINK}}
    </div>
  </div>

  <!-- CAMERA / FACE TRACKING -->
  <div class="section">
    <div class="section-head">
      <h2>Камера · Слежение за лицом (OpenCV)</h2>
      <label class="video-toggle">
        <input type="checkbox" id="video-toggle-checkbox" checked>
        Показывать видео
      </label>
      <span class="tag" id="camera-fps-tag">ожидание кадров…</span>
    </div>
    <div class="camera-frame" id="camera-frame">
      <div class="camera-placeholder" id="camera-placeholder">Нет видеопотока —<br>ждём кадры с ESP32</div>
      <div class="camera-placeholder" id="camera-video-off" style="display:none;">Видео отключено —<br>слежение продолжает работать</div>
      <img id="camera-feed" style="display:none;">
    </div>
    <div class="camera-badges">
      <span class="camera-badge" id="badge-face"><span class="dot"></span>Лицо не найдено</span>
      <span class="camera-badge tracking" id="badge-tracking"><span class="dot"></span>Слежение выключено</span>
      <span class="camera-badge" id="badge-dialog"><span class="dot"></span>Диалог неактивен</span>
    </div>
  </div>

  <!-- AI CORE -->
  <div class="section">
    <div class="section-head">
      <h2>AI Core — Обработка</h2>
      <span class="tag">STT / LLM / TTS</span>
    </div>
    <div class="core-grid" id="ai-mode-panel">
      <div class="core-card">
        <div class="name">Распознавание речи</div>
        <div class="desc">Локально: faster-whisper (~500 МБ)<br>Облако: OpenAI Whisper API</div>
        <div class="segmented">
          <button id="stt-local-btn" onclick="setAIMode('stt','local')">Локально</button>
          <button id="stt-cloud-btn" onclick="setAIMode('stt','cloud')">Облако</button>
        </div>
        <select class="model-select" id="stt-model-select" onchange="setSTTModel(this.value)"></select>
      </div>
      <div class="core-card">
        <div class="name">Языковая модель</div>
        <div class="desc">Локально: Qwen 7B GGUF (~4.5 ГБ)<br>Облако: GPT-4o-mini</div>
        <div class="segmented">
          <button id="llm-local-btn" onclick="setAIMode('llm','local')">Локально</button>
          <button id="llm-cloud-btn" onclick="setAIMode('llm','cloud')">Облако</button>
        </div>
        <select class="model-select" id="llm-model-select" onchange="setLLMModel(this.value)"></select>
      </div>
      <div class="core-card">
        <div class="name">Синтез речи</div>
        <div class="desc">Локально: Silero (~120 МБ)<br>Облако: OpenAI TTS API</div>
        <div class="segmented">
          <button id="tts-local-btn" onclick="setAIMode('tts','local')">Локально</button>
          <button id="tts-cloud-btn" onclick="setAIMode('tts','cloud')">Облако</button>
        </div>
        <select class="model-select" id="tts-model-select" onchange="setTTSSpeaker(this.value)"></select>
      </div>
    </div>
    <div class="core-note">Для облачных режимов требуется ключ API в .env — см. README</div>
  </div>

  <!-- AUDIO ROUTING -->
  <div class="section">
    <div class="section-head">
      <h2>Маршрутизация звука</h2>
      <span class="tag">Robot / Local</span>
    </div>
    <div class="audio-grid">
      <div class="audio-row">
        <span class="label">Микрофон (ввод)</span>
        <span class="audio-state" id="audio-input-mode-text">Робот · ESP32</span>
        <label class="switch">
          <input type="checkbox" id="audio-input-toggle" onchange="toggleAudioInputMode()">
          <span class="track"></span>
        </label>
      </div>
      <div class="audio-row">
        <span class="label">Динамик (вывод)</span>
        <span class="audio-state" id="audio-output-mode-text">Робот · ESP32</span>
        <label class="switch">
          <input type="checkbox" id="audio-output-toggle" onchange="toggleAudioOutputMode()">
          <span class="track"></span>
        </label>
      </div>
    </div>
  </div>

  <!-- MEMORY LEVELS -->
  <div class="section">
    <div class="section-head">
      <h2>Уровни памяти</h2>
      <span class="tag">LLM</span>
    </div>
    <div class="audio-grid">
      <div class="audio-row">
        <span class="label">Краткосрочная (STM)</span>
        <span class="audio-state" id="mem-stm-text">—</span>
        <label class="switch">
          <input type="checkbox" id="mem-stm-toggle" onchange="toggleMemoryLevel('stm')">
          <span class="track"></span>
        </label>
      </div>
      <div class="audio-row">
        <span class="label">Долгосрочная (LTM, Qdrant)</span>
        <span class="audio-state" id="mem-ltm-text">—</span>
        <label class="switch">
          <input type="checkbox" id="mem-ltm-toggle" onchange="toggleMemoryLevel('ltm')">
          <span class="track"></span>
        </label>
      </div>
      <div class="audio-row">
        <span class="label">Эмоциональный профиль</span>
        <span class="audio-state" id="mem-profile-text">—</span>
        <label class="switch">
          <input type="checkbox" id="mem-profile-toggle" onchange="toggleMemoryLevel('profile')">
          <span class="track"></span>
        </label>
      </div>
      <div class="audio-row">
        <span class="label">RAG канона мира</span>
        <span class="audio-state" id="mem-rag-text">—</span>
        <label class="switch">
          <input type="checkbox" id="mem-rag-toggle" onchange="toggleMemoryLevel('rag')">
          <span class="track"></span>
        </label>
      </div>
    </div>
    <div class="core-note">Изменения применяются сразу и сохраняются на диск — переживают перезапуск сервера</div>
  </div>

  <!-- VOICE -->
  <div class="section">
    <div class="section-head"><h2>Голосовое общение</h2></div>
    <div class="voice-panel">
      <button id="mic-btn" class="mic-btn" onclick="toggleRecording()">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z"/><path d="M19 11a7 7 0 0 1-14 0M12 19v3"/></svg>
      </button>
      <div class="voice-info">
        <div class="voice-status">
          <span id="voice-status-text">Нажмите и говорите</span>
          <span id="recording-indicator" class="rec">● Запись</span>
        </div>
        <canvas id="audio-visualizer" class="visualizer"></canvas>
      </div>
    </div>
  </div>

  <!-- CHAT + GESTURES -->
  <div class="row2">
    <div class="section">
      <div class="section-head"><h2>Текстовый чат</h2></div>
      <div class="chat-log" id="chat-history"></div>
      <div class="chat-input-row">
        <input type="text" id="chat-input" placeholder="Напишите Сорену…" onkeypress="if(event.key==='Enter') sendChat()">
        <button class="btn primary" onclick="sendChat()">Отправить</button>
      </div>
    </div>
    <div class="section">
      <div class="section-head"><h2>Жесты</h2></div>
      <div class="gesture-grid">
        <button class="btn" onclick="sendCmd({type:'animation',name:'wave'})">Помахать</button>
        <button class="btn" onclick="sendCmd({type:'animation',name:'nod'})">Кивнуть</button>
        <button class="btn" onclick="sendCmd({type:'animation',name:'shake_head'})">Качнуть головой</button>
        <button class="btn" onclick="sendCmd({type:'animation',name:'idle'})">Покой</button>
      </div>
      <button class="btn danger" onclick="sendCmd({type:'clear_history'})">Очистить историю диалога</button>
    </div>
  </div>

  <!-- SERVOS -->
  <div class="section">
    <div class="section-head">
      <h2>Сервоприводы</h2>
      <span class="tag">18 каналов</span>
    </div>
    <div class="servo-block-label">PCA9685 · Каналы 0–15</div>
    <div class="servo-grid" id="servo-grid-main"></div>
    <div class="servo-block-label">Прямое подключение · Каналы 16–17</div>
    <div class="servo-grid" id="servo-grid-direct" style="grid-template-columns:repeat(8,1fr);"></div>
    <div class="servo-actions">
      <button class="btn primary" onclick="setAllServos()">Применить все</button>
      <button class="btn" onclick="resetServos()">Сбросить в 90°</button>
    </div>
  </div>

  <!-- LOG -->
  <div class="section">
    <div class="section-head"><h2>Системный журнал</h2></div>
    <div class="log" id="log"></div>
  </div>

  <audio id="audio-player" style="display:none;"></audio>
</div>

<script>
  const ws = new WebSocket(`ws://${window.location.host}/ws`);
  let audioInputMode = 'robot';
  let audioOutputMode = 'robot';
  let aiModes = { stt: 'local', tts: 'local', llm: 'local' };
  let memoryFlags = { stm: true, ltm: true, profile: true, rag: true };
  let modelConfig = { llm: {mode:'local', current:null, local_models:[], cloud_models:[]}, stt: {current:null, models:[]}, tts: {mode:'local', current:null, speakers:[]} };

  ws.onopen = () => {
    document.getElementById('status-dot').classList.add('online');
    document.getElementById('status-text').textContent = 'ONLINE';
    log('WebSocket подключен');
    ws.send(JSON.stringify({type:'hello', client:'panel'}));
    ws.send(JSON.stringify({type:'audio_mode'}));
    ws.send(JSON.stringify({type:'ai_mode'}));
    ws.send(JSON.stringify({type:'video_pref', enabled: videoPrefEnabled}));
    fetchMemoryConfig();
    fetchModelConfig();
  };
  ws.onclose = () => {
    document.getElementById('status-dot').classList.remove('online');
    document.getElementById('status-text').textContent = 'OFFLINE';
    log('WebSocket отключен');
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'video_frame') {
      handleVideoFrame(data);
      return; // не спамим системный журнал каждым кадром
    }
    log('← ' + JSON.stringify(data));
    if (data.angles) updateServoDisplay(data.angles);
    if (data.emotion) log('Эмоция: ' + data.emotion);
    if (data.type === 'audio_mode') {
      if (data.input_mode) audioInputMode = data.input_mode;
      if (data.output_mode) audioOutputMode = data.output_mode;
      updateAudioModeUI();
    }
    if (data.type === 'ai_mode' && data.modes) { aiModes = data.modes; updateAIModeUI(); }
    // Ответ на команду {type:'text'} (чат при audioOutputMode === 'robot') — у него нет
    // поля type, только status/response, поэтому раньше молча терялся тут и не попадал в чат.
    if (!data.type && data.status === 'ok' && typeof data.response === 'string') {
      addMessage('robot', data.response, data.emotion);
      if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
    }
    if (data.modes && !data.type) { aiModes = data.modes; updateAIModeUI(); }
    if (typeof data.dialog_active === 'boolean') updateDialogBadges(data.face_detected, data.dialog_active);
  };

  // ===== Показывать/скрывать видео в панели (не влияет на слежение — только на вывод) =====
  const VIDEO_PREF_KEY = 'soren_panel_show_video';
  function getVideoPref() {
    const stored = localStorage.getItem(VIDEO_PREF_KEY);
    return stored === null ? true : stored === '1';
  }
  function setVideoPref(enabled) {
    localStorage.setItem(VIDEO_PREF_KEY, enabled ? '1' : '0');
  }
  let videoPrefEnabled = getVideoPref();
  let hasReceivedFrame = false;

  function updateCameraFrameVisibility() {
    const img = document.getElementById('camera-feed');
    const waitingPlaceholder = document.getElementById('camera-placeholder');
    const offPlaceholder = document.getElementById('camera-video-off');

    if (!videoPrefEnabled) {
      img.style.display = 'none';
      waitingPlaceholder.style.display = 'none';
      offPlaceholder.style.display = 'block';
    } else if (hasReceivedFrame) {
      img.style.display = 'block';
      waitingPlaceholder.style.display = 'none';
      offPlaceholder.style.display = 'none';
    } else {
      img.style.display = 'none';
      waitingPlaceholder.style.display = 'block';
      offPlaceholder.style.display = 'none';
    }
  }

  const videoToggleCheckbox = document.getElementById('video-toggle-checkbox');
  videoToggleCheckbox.checked = videoPrefEnabled;
  updateCameraFrameVisibility();
  videoToggleCheckbox.addEventListener('change', () => {
    videoPrefEnabled = videoToggleCheckbox.checked;
    setVideoPref(videoPrefEnabled);
    updateCameraFrameVisibility();
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({type:'video_pref', enabled: videoPrefEnabled}));
    }
  });

  let lastFrameTs = 0, frameCount = 0, fpsWindowStart = Date.now();
  function handleVideoFrame(data) {
    const img = document.getElementById('camera-feed');
    img.src = 'data:image/jpeg;base64,' + data.image;
    hasReceivedFrame = true;
    updateCameraFrameVisibility();

    updateDialogBadges(data.face_detected, data.dialog_active);

    frameCount++;
    const now = Date.now();
    if (now - fpsWindowStart >= 1000) {
      const facesTxt = data.faces_count > 1 ? (', лиц в кадре: ' + data.faces_count) : '';
      document.getElementById('camera-fps-tag').textContent = frameCount + ' fps (панель)' + facesTxt;
      frameCount = 0;
      fpsWindowStart = now;
    }
  }

  function updateDialogBadges(faceDetected, dialogActive) {
    const faceBadge = document.getElementById('badge-face');
    const trackBadge = document.getElementById('badge-tracking');
    const dialogBadge = document.getElementById('badge-dialog');

    faceBadge.classList.toggle('on', !!faceDetected);
    faceBadge.childNodes[1].textContent = faceDetected ? ' Лицо найдено' : ' Лицо не найдено';

    const tracking = !!faceDetected && !!dialogActive;
    trackBadge.classList.toggle('on', tracking);
    trackBadge.childNodes[1].textContent = tracking ? ' Слежение активно' : ' Слежение выключено';

    dialogBadge.classList.toggle('on', !!dialogActive);
    dialogBadge.childNodes[1].textContent = dialogActive ? ' Диалог активен' : ' Диалог неактивен';
  }

  async function setAIMode(module, mode) {
    log(`Переключение ${module.toUpperCase()} → ${mode}…`);
    const fd = new FormData(); fd.append('module', module); fd.append('mode', mode);
    try {
      const r = await fetch('/ai_mode', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        if (data.modes) { aiModes = data.modes; updateAIModeUI(); }
        log(`${module.toUpperCase()} → ${mode} ✓`);
        if (module === 'llm' || module === 'tts') { fetchModelConfig(); }
      } else {
        log('Ошибка: ' + (data.message || 'неизвестная'));
        if (data.modes) { aiModes = data.modes; updateAIModeUI(); }
      }
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  function updateAIModeUI() {
    ['stt','llm','tts'].forEach(mod => {
      const mode = aiModes[mod] || 'local';
      const localBtn = document.getElementById(mod+'-local-btn');
      const cloudBtn = document.getElementById(mod+'-cloud-btn');
      if (localBtn) localBtn.className = mode === 'local' ? 'active local' : '';
      if (cloudBtn) cloudBtn.className = mode === 'cloud' ? 'active cloud' : '';
    });
    updateModelSelectsUI();
  }

  async function toggleAudioInputMode() {
    const t = document.getElementById('audio-input-toggle');
    const newMode = t.checked ? 'local' : 'robot';
    const fd = new FormData(); fd.append('mode', newMode); fd.append('type', 'input');
    const r = await fetch('/audio_mode', {method:'POST', body:fd});
    const data = await r.json();
    if (data.status === 'ok') { audioInputMode = data.audio_input_mode; updateAudioModeUI(); }
  }
  async function toggleAudioOutputMode() {
    const t = document.getElementById('audio-output-toggle');
    const newMode = t.checked ? 'local' : 'robot';
    const fd = new FormData(); fd.append('mode', newMode); fd.append('type', 'output');
    const r = await fetch('/audio_mode', {method:'POST', body:fd});
    const data = await r.json();
    if (data.status === 'ok') { audioOutputMode = data.audio_output_mode; updateAudioModeUI(); }
  }
  function updateAudioModeUI() {
    const it = document.getElementById('audio-input-toggle');
    const itx = document.getElementById('audio-input-mode-text');
    it.checked = (audioInputMode === 'local');
    itx.textContent = audioInputMode === 'local' ? 'Локально · микрофон ПК' : 'Робот · ESP32';
    const ot = document.getElementById('audio-output-toggle');
    const otx = document.getElementById('audio-output-mode-text');
    ot.checked = (audioOutputMode === 'local');
    otx.textContent = audioOutputMode === 'local' ? 'Локально · наушники ПК' : 'Робот · ESP32';
  }

  async function fetchMemoryConfig() {
    try {
      const r = await fetch('/status');
      const data = await r.json();
      if (data.memory_flags) { memoryFlags = data.memory_flags; updateMemoryFlagsUI(); }
    } catch(e) { log('Не удалось получить уровни памяти: ' + e.message); }
  }

  async function fetchModelConfig() {
    try {
      const r = await fetch('/status');
      const data = await r.json();
      if (data.model_config) { modelConfig = data.model_config; updateModelSelectsUI(); }
    } catch(e) { log('Не удалось получить список моделей: ' + e.message); }
  }

  function _fillSelect(selectEl, options, currentId, placeholder) {
    if (!selectEl) return;
    selectEl.innerHTML = '';
    if (!options || !options.length) {
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
    if (!matched && currentId) {
      const opt = document.createElement('option');
      opt.value = currentId; opt.textContent = currentId + ' (вручную из config.yaml)';
      opt.selected = true;
      selectEl.insertBefore(opt, selectEl.firstChild);
    }
  }

  function updateModelSelectsUI() {
    const llmMode = (aiModes.llm || modelConfig.llm.mode || 'local');
    const llmOptions = llmMode === 'cloud' ? modelConfig.llm.cloud_models : modelConfig.llm.local_models;
    _fillSelect(document.getElementById('llm-model-select'), llmOptions, modelConfig.llm.current, 'Список моделей пуст — задайте llm.local_models/cloud_models в config.yaml');

    _fillSelect(document.getElementById('stt-model-select'), modelConfig.stt.models, modelConfig.stt.current, 'Список пуст — задайте stt.whisper_models в config.yaml');

    const ttsSpeakers = (modelConfig.tts.speakers || []).map(s => ({id: s, label: s}));
    _fillSelect(document.getElementById('tts-model-select'), ttsSpeakers, modelConfig.tts.current, 'Нет доступных голосов');
  }

  async function setLLMModel(modelId) {
    if (!modelId) return;
    log(`Модель LLM → ${modelId}…`);
    const fd = new FormData(); fd.append('model_id', modelId);
    try {
      const r = await fetch('/llm_model', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        if (data.model_config) { modelConfig = data.model_config; }
        aiModes.llm = modelConfig.llm.mode; updateAIModeUI(); updateModelSelectsUI();
        log(`LLM модель → ${modelId} ✓ (режим: ${modelConfig.llm.mode})`);
      } else {
        log('Ошибка: ' + (data.message || 'неизвестная'));
      }
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  async function setSTTModel(modelId) {
    if (!modelId) return;
    log(`Модель STT (Whisper) → ${modelId}…`);
    const fd = new FormData(); fd.append('model_id', modelId);
    try {
      const r = await fetch('/stt_model', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        if (data.model_config) { modelConfig = data.model_config; }
        log(`STT модель → ${modelId} ✓`);
      } else {
        log('Ошибка: ' + (data.message || 'неизвестная'));
      }
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  async function setTTSSpeaker(speaker) {
    if (!speaker) return;
    log(`Голос TTS → ${speaker}…`);
    const fd = new FormData(); fd.append('speaker', speaker);
    try {
      const r = await fetch('/tts_speaker', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        if (data.model_config) { modelConfig = data.model_config; }
        log(`Голос TTS → ${speaker} ✓`);
      } else {
        log('Ошибка: ' + (data.message || 'неизвестная'));
      }
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
  }

  async function toggleMemoryLevel(level) {
    const t = document.getElementById('mem-' + level + '-toggle');
    const enabled = t.checked;
    log(`Уровень памяти '${level}' → ${enabled ? 'вкл' : 'выкл'}…`);
    const fd = new FormData(); fd.append('level', level); fd.append('enabled', enabled ? '1' : '0');
    try {
      const r = await fetch('/memory_config', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        if (data.memory_flags) { memoryFlags = data.memory_flags; }
        log(`Память '${level}' → ${enabled ? 'вкл' : 'выкл'} ✓`);
      } else {
        log('Ошибка: ' + (data.message || 'неизвестная'));
      }
    } catch(e) { log('Сетевая ошибка: ' + e.message); }
    updateMemoryFlagsUI();
  }

  function updateMemoryFlagsUI() {
    const labels = { stm: 'реплики хранятся', ltm: 'записи идут в Qdrant', profile: 'профиль обновляется', rag: 'канон подключён' };
    ['stm','ltm','profile','rag'].forEach(level => {
      const box = document.getElementById('mem-' + level + '-toggle');
      const txt = document.getElementById('mem-' + level + '-text');
      const on = !!memoryFlags[level];
      if (box) box.checked = on;
      if (txt) txt.textContent = on ? ('Включено · ' + labels[level]) : 'Выключено';
    });
  }

  let mediaRecorder=null, audioChunks=[], isRecording=false, audioContext=null, analyser=null, visCanvas=null, visCtx=null;

  async function toggleRecording() {
    const btn = document.getElementById('mic-btn');
    const statusText = document.getElementById('voice-status-text');
    const indicator = document.getElementById('recording-indicator');
    const visualizer = document.getElementById('audio-visualizer');
    if (!isRecording) {
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
        statusText.textContent = 'Идёт запись…';
        indicator.classList.add('active');
        visualizer.classList.add('active');
        setupVisualizer(stream);
        log('Начало записи голоса');
      } catch(err) {
        log('Ошибка доступа к микрофону: ' + err.message);
        alert('Разрешите доступ к микрофону в настройках браузера');
      }
    } else {
      mediaRecorder.stop();
      isRecording = false;
      btn.classList.remove('recording');
      statusText.textContent = 'Обработка…';
      indicator.classList.remove('active');
      visualizer.classList.remove('active');
      log('Конец записи, отправка…');
    }
  }

  function setupVisualizer(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 256;
    visCanvas = document.getElementById('audio-visualizer');
    visCtx = visCanvas.getContext('2d');
    visCanvas.width = visCanvas.offsetWidth;
    visCanvas.height = visCanvas.offsetHeight;
    drawVisualizer();
  }
  function drawVisualizer() {
    if (!isRecording || !analyser) return;
    requestAnimationFrame(drawVisualizer);
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);
    visCtx.fillStyle = '#0a0e0b';
    visCtx.fillRect(0,0,visCanvas.width, visCanvas.height);
    const barWidth = (visCanvas.width / bufferLength) * 2.5;
    let x = 0;
    for (let i=0;i<bufferLength;i++){
      const barHeight = dataArray[i] / 3;
      visCtx.fillStyle = '#d4a537';
      visCtx.fillRect(x, visCanvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }
  }

  async function sendVoiceToServer(blob) {
    const statusText = document.getElementById('voice-status-text');
    statusText.textContent = 'Отправка на сервер…';
    const fd = new FormData();
    fd.append('audio', blob, 'voice.wav');
    fd.append('audio_output_mode_param', audioOutputMode);
    try {
      const r = await fetch('/voice', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        addMessage('user', data.user_text);
        addMessage('robot', data.response, data.emotion);
        if (audioOutputMode === 'local' && data.audio_base64) playAudio(data.audio_base64);
        if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
        statusText.textContent = 'Готово — нажмите для новой записи';
      } else {
        addMessage('robot', 'Ошибка: ' + (data.message || 'неизвестная'));
        statusText.textContent = 'Ошибка: ' + data.message;
      }
    } catch(e) { log('Ошибка сети: ' + e); statusText.textContent = 'Ошибка сети'; }
  }

  async function sendChat() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage('user', text);
    if (audioOutputMode === 'robot') sendCmd({type:'text', text:text});
    else await sendLocal(text);
  }
  async function sendLocal(text) {
    try {
      const fd = new FormData(); fd.append('text', text);
      const r = await fetch('/speak', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        addMessage('robot', data.response, data.emotion);
        if (data.audio_base64) playAudio(data.audio_base64);
        else if (data.tts_failed) log('⚠ Синтез речи не удался — ответ показан без озвучки');
        if (data.ai_modes) { aiModes = data.ai_modes; updateAIModeUI(); }
      } else addMessage('robot', 'Ошибка: ' + (data.message || 'неизвестная'));
    } catch(e) { log('Ошибка сети: ' + e); }
  }
  function playAudio(b64) {
    const audio = document.getElementById('audio-player');
    audio.src = 'data:audio/wav;base64,' + b64;
    audio.play().catch(e => log('Ошибка воспроизведения: ' + e.message));
  }
  function addMessage(sender, text, emotion) {
    const chat = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = 'msg ' + sender;
    const who = sender === 'user' ? 'Вы' : 'Сорен';
    let emTag = '';
    if (emotion) emTag = `<span class="emotion-tag em-${emotion}">${emotion}</span>`;
    div.innerHTML = `<span class="who">${who}</span>${text}${emTag}`;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }
  function sendCmd(cmd) { ws.send(JSON.stringify(cmd)); log('→ ' + JSON.stringify(cmd)); }
  function log(msg) {
    const el = document.getElementById('log');
    el.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${msg}</div>`;
    el.scrollTop = el.scrollHeight;
  }

  function createServoGrid() {
    const main = document.getElementById('servo-grid-main');
    for (let i=0;i<16;i++) main.appendChild(makeServo(i));
    const direct = document.getElementById('servo-grid-direct');
    for (let i=16;i<18;i++) direct.appendChild(makeServo(i));
  }
  function makeServo(i) {
    const div = document.createElement('div');
    div.className = 'servo';
    div.innerHTML = `<div class="ch">CH ${String(i).padStart(2,'0')}</div><div class="deg" id="val-${i}">90°</div><input type="range" id="servo-${i}" min="0" max="180" value="90" oninput="document.getElementById('val-${i}').textContent=this.value+'°'">`;
    return div;
  }
  function setAllServos() {
    const angles = [];
    for (let i=0;i<18;i++) angles.push(parseInt(document.getElementById(`servo-${i}`).value));
    sendCmd({type:'servo_multi', angles});
  }
  function resetServos() {
    for (let i=0;i<18;i++) {
      document.getElementById(`servo-${i}`).value = 90;
      document.getElementById(`val-${i}`).textContent = '90°';
    }
    sendCmd({type:'servo_multi', angles:new Array(18).fill(90)});
  }
  function updateServoDisplay(angles) {
    for (let i=0;i<angles.length;i++) {
      const slider = document.getElementById(`servo-${i}`);
      const val = document.getElementById(`val-${i}`);
      if (slider && val) { slider.value = angles[i]; val.textContent = angles[i] + '°'; }
    }
  }
  createServoGrid();

  function toggleTheme() {
    const body = document.body;
    const root = document.documentElement;
    const label = document.getElementById('theme-label');
    const icon = document.getElementById('theme-icon');
    if (body.classList.contains('light')) {
      body.classList.remove('light');
      root.classList.remove('light');
      label.textContent = 'Тёмная';
      icon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
      localStorage.setItem('soren-theme', 'dark');
    } else {
      body.classList.add('light');
      root.classList.add('light');
      label.textContent = 'Светлая';
      icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
      localStorage.setItem('soren-theme', 'light');
    }
  }
  // Восстановление темы при загрузке
  (function() {
    const saved = localStorage.getItem('soren-theme');
    const icon = document.getElementById('theme-icon');
    if (saved === 'light') {
      document.body.classList.add('light');
      document.documentElement.classList.add('light');
      document.getElementById('theme-label').textContent = 'Светлая';
      icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
    }
  })();
</script>
</body>
</html>

"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)