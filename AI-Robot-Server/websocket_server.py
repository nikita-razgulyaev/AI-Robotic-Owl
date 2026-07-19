"""WebSocket сервер - связь с ESP32 и веб-интерфейсом + голосовое общение"""
import os
import asyncio
import json
import logging
import base64
import io
import wave
import tempfile
from typing import Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from modules.robot_brain import RobotBrain
from config.settings import SERVER_HOST, SERVER_PORT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot AI Server - Soren", version="3.0")

robot_brain: RobotBrain = None
active_connections: Set[WebSocket] = set()

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
        "connections": len(active_connections),
        "servo_angles": robot_brain.servos.get_current_angles(),
        "vision_context": robot_brain.vision_context,
        "current_emotion": robot_brain.current_emotion
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

        llm_result = robot_brain.llm.generate(user_text, robot_brain.vision_context)
        response_text = llm_result.get("text", "")
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")

        if not response_text:
            return JSONResponse({"status": "error", "message": "LLM вернул пустой ответ"})

        servo_angles = robot_brain.emotion_engine.get_servo_angles(emotion)
        eye_led = robot_brain.emotion_engine.get_eye_led(emotion)

        tts_audio = robot_brain.tts.synthesize(response_text)
        if not tts_audio:
            return JSONResponse({"status": "error", "message": "Ошибка синтеза речи"})

        if action:
            asyncio.create_task(robot_brain.servos.play_animation(action))
        else:
            robot_brain.servos.set_all_servos(servo_angles)

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"ESP32 подключен: {client_info}")

    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                await handle_text_message(websocket, message["text"])
            elif "bytes" in message:
                await handle_binary_message(websocket, message["bytes"])
    except WebSocketDisconnect:
        logger.info(f"ESP32 отключен: {client_info}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
    finally:
        active_connections.discard(websocket)


async def handle_text_message(websocket: WebSocket, text: str):
    try:
        data = json.loads(text)
        msg_type = data.get("type")

        if msg_type in ["servo", "servo_multi", "animation", "text", "get_status", "clear_history", "set_mode"]:
            result = await robot_brain.handle_command(data)
            await websocket.send_json(result)
        elif msg_type == "ping":
            await websocket.send_json({"type": "pong", "timestamp": data.get("timestamp")})
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


async def handle_binary_message(websocket: WebSocket, data: bytes):
    try:
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
            servo_cmd = {
                "type": "servo_update",
                "angles": vision_result["servo_angles"],
                "face_detected": vision_result["face_detected"],
                "face_offset": vision_result["face_offset"]
            }
            await websocket.send_json(servo_cmd)
    except Exception as e:
        logger.error(f"Ошибка бинарных данных: {e}")


# ===== ПАНЕЛЬ УПРАВЛЕНИЯ =====

@app.get("/panel", response_class=HTMLResponse)
async def control_panel():
    return PANEL_HTML


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
    --bg:#0e130f;
    --panel:#161c18;
    --panel-alt:#1b2220;
    --raised:#1f2723;
    --border:#2a332c;
    --border-soft:#212a24;
    --text:#e8e5da;
    --text-dim:#94a099;
    --text-faint:#5c6961;
    --amber:#d4a537;
    --amber-soft:#8a6f2a;
    --sage:#7a9b76;
    --slate:#5f8fb0;
    --rust:#b5533c;
    --radius:3px;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:'IBM Plex Sans', sans-serif;
    padding:28px; -webkit-font-smoothing:antialiased;
  }
  .wrap{ max-width:1360px; margin:0 auto; }

  /* ===== NAMEPLATE ===== */
  .nameplate{
    display:flex; align-items:center; justify-content:space-between;
    border:1px solid var(--border); border-radius:var(--radius);
    padding:18px 24px; margin-bottom:22px;
    background:linear-gradient(180deg, var(--panel-alt), var(--panel));
    position:relative;
  }
  .nameplate::before{
    content:""; position:absolute; inset:0; border-radius:var(--radius);
    background:
      linear-gradient(90deg, var(--amber-soft) 0%, transparent 2px) 0 0,
      linear-gradient(90deg, var(--amber-soft) 0%, transparent 2px) 100% 0;
    background-size: 2px 100%; background-repeat:no-repeat; opacity:.6;
  }
  .brand{ display:flex; align-items:center; gap:16px; }
  .eye{ width:38px; height:38px; flex-shrink:0; }
  .brand-text h1{
    font-family:'Spectral', serif; font-weight:600; font-size:26px;
    margin:0; letter-spacing:.5px; color:var(--text);
  }
  .brand-text .subtitle{
    font-family:'IBM Plex Mono', monospace; font-size:10.5px;
    letter-spacing:2px; color:var(--text-faint); text-transform:uppercase;
    margin-top:3px;
  }
  .nameplate-meta{ display:flex; align-items:center; gap:28px; }
  .meta-item{ text-align:right; }
  .meta-item .label{
    font-family:'IBM Plex Mono', monospace; font-size:9.5px; letter-spacing:1.5px;
    color:var(--text-faint); text-transform:uppercase; display:block; margin-bottom:3px;
  }
  .meta-item .value{ font-family:'IBM Plex Mono', monospace; font-size:13px; }
  .status-dot{ width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:7px; background:var(--rust); box-shadow:0 0 6px var(--rust);}
  .status-dot.online{ background:var(--sage); box-shadow:0 0 6px var(--sage); }

  /* ===== SECTION FRAME ===== */
  .section{
    border:1px solid var(--border); border-radius:var(--radius);
    background:var(--panel); padding:18px 20px; margin-bottom:16px;
    position:relative;
  }
  .section-head{
    display:flex; align-items:baseline; justify-content:space-between;
    margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--border-soft);
  }
  .section-head h2{
    font-family:'IBM Plex Mono', monospace; font-size:11.5px; font-weight:600;
    letter-spacing:2px; text-transform:uppercase; color:var(--text-dim); margin:0;
  }
  .section-head .tag{
    font-family:'IBM Plex Mono', monospace; font-size:10px; color:var(--text-faint);
  }
  .row2{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }

  /* ===== AI CORE — segmented controls ===== */
  .core-grid{ display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }
  .core-card{
    border:1px solid var(--border-soft); border-radius:var(--radius); background:var(--panel-alt);
    padding:14px 16px;
  }
  .core-card .name{ font-size:13.5px; font-weight:600; margin-bottom:2px; }
  .core-card .desc{ font-size:11px; color:var(--text-faint); margin-bottom:12px; line-height:1.5; }
  .segmented{
    display:flex; border:1px solid var(--border); border-radius:var(--radius); overflow:hidden;
    font-family:'IBM Plex Mono', monospace; font-size:11.5px;
  }
  .segmented button{
    flex:1; border:none; background:transparent; color:var(--text-faint);
    padding:8px 6px; cursor:pointer; letter-spacing:.5px; transition:.15s;
  }
  .segmented button:first-child{ border-right:1px solid var(--border); }
  .segmented button.active.local{ background:rgba(122,155,118,.16); color:var(--sage); }
  .segmented button.active.cloud{ background:rgba(95,143,176,.16); color:var(--slate); }
  .segmented button:hover:not(.active){ color:var(--text-dim); background:var(--raised); }
  .core-note{ font-size:10.5px; color:var(--text-faint); margin-top:14px; text-align:center; border-top:1px solid var(--border-soft); padding-top:10px; }

  /* ===== AUDIO ROUTING ===== */
  .audio-grid{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .audio-row{
    display:flex; align-items:center; justify-content:space-between; gap:14px;
    border:1px solid var(--border-soft); border-radius:var(--radius); background:var(--panel-alt);
    padding:12px 16px;
  }
  .audio-row .label{ font-size:12.5px; color:var(--text-dim); }
  .audio-state{ font-family:'IBM Plex Mono', monospace; font-size:11px; }
  .switch{ position:relative; width:46px; height:24px; flex-shrink:0; }
  .switch input{ opacity:0; width:0; height:0; }
  .switch .track{
    position:absolute; inset:0; background:var(--raised); border:1px solid var(--border);
    border-radius:20px; transition:.2s; cursor:pointer;
  }
  .switch .track::before{
    content:""; position:absolute; width:16px; height:16px; left:3px; top:2.5px;
    background:var(--text-faint); border-radius:50%; transition:.2s;
  }
  .switch input:checked + .track{ background:rgba(95,143,176,.16); border-color:var(--slate); }
  .switch input:checked + .track::before{ transform:translateX(21px); background:var(--slate); }

  /* ===== VOICE ===== */
  .voice-panel{ display:flex; align-items:center; gap:22px; padding:6px 4px; }
  .mic-btn{
    width:64px; height:64px; border-radius:50%; flex-shrink:0;
    border:1.5px solid var(--amber-soft); background:var(--panel-alt); color:var(--amber);
    cursor:pointer; display:flex; align-items:center; justify-content:center; transition:.2s;
  }
  .mic-btn:hover{ border-color:var(--amber); }
  .mic-btn.recording{ background:rgba(181,83,60,.15); border-color:var(--rust); color:var(--rust); animation:breathe 1.4s infinite; }
  @keyframes breathe{ 0%,100%{ box-shadow:0 0 0 0 rgba(181,83,60,.35);} 50%{ box-shadow:0 0 0 8px rgba(181,83,60,0);} }
  .voice-info{ flex:1; }
  .voice-status{ font-size:12.5px; color:var(--text-dim); margin-bottom:8px; }
  .voice-status .rec{ color:var(--rust); font-family:'IBM Plex Mono', monospace; margin-left:8px; display:none; }
  .voice-status .rec.active{ display:inline; }
  canvas.visualizer{ width:100%; height:34px; background:var(--panel-alt); border:1px solid var(--border-soft); border-radius:var(--radius); display:none; }
  canvas.visualizer.active{ display:block; }

  /* ===== CHAT ===== */
  .chat-log{ max-height:260px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; margin-bottom:12px; padding-right:4px; }
  .msg{ font-size:12.5px; line-height:1.5; padding:9px 12px; border-radius:var(--radius); border:1px solid var(--border-soft); }
  .msg.user{ background:var(--panel-alt); }
  .msg.robot{ background:rgba(212,165,55,.06); border-color:rgba(212,165,55,.2); }
  .msg .who{ font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:1px; color:var(--text-faint); text-transform:uppercase; display:block; margin-bottom:3px; }
  .emotion-tag{ display:inline-block; font-family:'IBM Plex Mono', monospace; font-size:9.5px; letter-spacing:1px; text-transform:uppercase; padding:1px 7px; border-radius:10px; margin-left:8px; border:1px solid; }
  .em-calm{ color:#7fa8c9; border-color:#7fa8c9; }
  .em-sad{ color:#8686b0; border-color:#8686b0; }
  .em-angry{ color:var(--rust); border-color:var(--rust); }
  .em-loving{ color:#d9a04a; border-color:#d9a04a; }
  .em-determined{ color:var(--sage); border-color:var(--sage); }
  .em-surprised{ color:#c17fd9; border-color:#c17fd9; }
  .em-tired{ color:var(--text-faint); border-color:var(--text-faint); }
  .chat-input-row{ display:flex; gap:8px; }
  input[type=text]{
    flex:1; background:var(--panel-alt); border:1px solid var(--border); color:var(--text);
    padding:10px 12px; border-radius:var(--radius); font-family:'IBM Plex Sans', sans-serif; font-size:13px;
  }
  input[type=text]:focus{ outline:none; border-color:var(--amber-soft); }

  button.btn{
    background:transparent; border:1px solid var(--border); color:var(--text-dim);
    padding:9px 16px; border-radius:var(--radius); cursor:pointer; font-size:12px;
    font-family:'IBM Plex Sans', sans-serif; font-weight:500; transition:.15s;
  }
  button.btn:hover{ border-color:var(--amber-soft); color:var(--text); }
  button.btn.primary{ border-color:var(--amber-soft); color:var(--amber); }
  button.btn.primary:hover{ background:rgba(212,165,55,.08); }
  button.btn.danger{ border-color:rgba(181,83,60,.5); color:var(--rust); }
  button.btn.danger:hover{ background:rgba(181,83,60,.08); }

  .gesture-grid{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:14px; }

  /* ===== SERVOS ===== */
  .servo-block-label{
    font-family:'IBM Plex Mono', monospace; font-size:10px; letter-spacing:1.5px; color:var(--text-faint);
    text-transform:uppercase; margin:14px 0 8px; padding-bottom:6px; border-bottom:1px solid var(--border-soft);
  }
  .servo-block-label:first-child{ margin-top:0; }
  .servo-grid{ display:grid; grid-template-columns:repeat(8, 1fr); gap:8px; }
  .servo{
    border:1px solid var(--border-soft); background:var(--panel-alt); border-radius:var(--radius);
    padding:8px 6px; text-align:center;
  }
  .servo .ch{ font-family:'IBM Plex Mono', monospace; font-size:9.5px; color:var(--text-faint); letter-spacing:.5px; }
  .servo .deg{ font-family:'IBM Plex Mono', monospace; font-size:13px; color:var(--amber); margin:3px 0; }
  .servo input[type=range]{ width:100%; accent-color:var(--amber); height:14px; }
  .servo-actions{ display:flex; gap:8px; margin-top:14px; }

  /* ===== LOG ===== */
  .log{
    background:#0a0e0b; border:1px solid var(--border-soft); border-radius:var(--radius);
    height:200px; overflow-y:auto; padding:10px 12px;
    font-family:'IBM Plex Mono', monospace; font-size:11px; line-height:1.7; color:#8fa397;
  }
  .log::-webkit-scrollbar, .chat-log::-webkit-scrollbar{ width:6px; }
  .log::-webkit-scrollbar-thumb, .chat-log::-webkit-scrollbar-thumb{ background:var(--border); border-radius:3px; }

  @media (max-width: 860px){
    .core-grid{ grid-template-columns:1fr; }
    .audio-grid{ grid-template-columns:1fr; }
    .row2{ grid-template-columns:1fr; }
    .servo-grid{ grid-template-columns:repeat(4,1fr); }
    .nameplate{ flex-direction:column; align-items:flex-start; gap:14px; }
    .nameplate-meta{ gap:18px; }
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
      </div>
      <div class="core-card">
        <div class="name">Языковая модель</div>
        <div class="desc">Локально: Qwen 7B GGUF (~4.5 ГБ)<br>Облако: GPT-4o-mini</div>
        <div class="segmented">
          <button id="llm-local-btn" onclick="setAIMode('llm','local')">Локально</button>
          <button id="llm-cloud-btn" onclick="setAIMode('llm','cloud')">Облако</button>
        </div>
      </div>
      <div class="core-card">
        <div class="name">Синтез речи</div>
        <div class="desc">Локально: Silero (~120 МБ)<br>Облако: OpenAI TTS API</div>
        <div class="segmented">
          <button id="tts-local-btn" onclick="setAIMode('tts','local')">Локально</button>
          <button id="tts-cloud-btn" onclick="setAIMode('tts','cloud')">Облако</button>
        </div>
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

  ws.onopen = () => {
    document.getElementById('status-dot').classList.add('online');
    document.getElementById('status-text').textContent = 'ONLINE';
    log('WebSocket подключен');
    ws.send(JSON.stringify({type:'audio_mode'}));
    ws.send(JSON.stringify({type:'ai_mode'}));
  };
  ws.onclose = () => {
    document.getElementById('status-dot').classList.remove('online');
    document.getElementById('status-text').textContent = 'OFFLINE';
    log('WebSocket отключен');
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    log('← ' + JSON.stringify(data));
    if (data.angles) updateServoDisplay(data.angles);
    if (data.emotion) log('Эмоция: ' + data.emotion);
    if (data.type === 'audio_mode') {
      if (data.input_mode) audioInputMode = data.input_mode;
      if (data.output_mode) audioOutputMode = data.output_mode;
      updateAudioModeUI();
    }
    if (data.type === 'ai_mode' && data.modes) { aiModes = data.modes; updateAIModeUI(); }
    if (data.modes && !data.type) { aiModes = data.modes; updateAIModeUI(); }
  };

  async function setAIMode(module, mode) {
    log(`Переключение ${module.toUpperCase()} → ${mode}…`);
    const fd = new FormData(); fd.append('module', module); fd.append('mode', mode);
    try {
      const r = await fetch('/ai_mode', {method:'POST', body:fd});
      const data = await r.json();
      if (data.status === 'ok') {
        if (data.modes) { aiModes = data.modes; updateAIModeUI(); }
        log(`${module.toUpperCase()} → ${mode} ✓`);
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
</script>
</body>
</html>

"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)