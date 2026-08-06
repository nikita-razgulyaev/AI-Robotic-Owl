"""Конфигурация сервера робота Сорена — v3.5 (YAML + память + fine-tuning + Vector RAG)"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# === Загрузка YAML конфига ===
CONFIG_PATH = BASE_DIR / "config.yaml"
_config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f)

# === ПРОИЗВОДИТЕЛЬНОСТЬ ===
# fast_mode: true — отключает Vector RAG, долгосрочную память (Qdrant) и
# чтение/запись JSON-профиля на диск; system prompt берётся короткий
# (character/Soren_short.txt вместо Soren.txt). См. комментарий в config.yaml.
FAST_MODE = bool(_config.get("performance", {}).get("fast_mode", False))

# === Character / Knowledge System ===
CHARACTER_DIR = BASE_DIR / "character"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# === VECTOR RAG ===
QDRANT_PATH = _config.get("qdrant", {}).get("path", "./qdrant_storage")
RAG_ENCODER_MODEL = _config.get("qdrant", {}).get("encoder_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# === ПАМЯТЬ ===
MEMORY_DIR = BASE_DIR / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
SHORT_TERM_MEMORY_MAX = _config.get("memory", {}).get("short_term_max", 10)
# fast_mode принудительно выключает Qdrant LTM, даже если в config.yaml забыли поставить false
LONG_TERM_MEMORY_ENABLED = False if FAST_MODE else _config.get("memory", {}).get("long_term_enabled", True)
QDRANT_HOST = _config.get("memory", {}).get("qdrant_host", "localhost")
QDRANT_PORT = _config.get("memory", {}).get("qdrant_port", 6333)
QDRANT_COLLECTIONS = _config.get("memory", {}).get("collections", {
    "emotional_moments": "soren_emotional_moments",
    "dialogue_turns": "soren_dialogue_turns",
    "facts": "soren_facts",
    "rag_chunks": "soren_rag_chunks",
})

# === FINE-TUNING / LoRA ===
LORA_ENABLED = _config.get("lora", {}).get("enabled", False)
LORA_PATH = BASE_DIR / _config.get("lora", {}).get("path", "models/soren_lora_v1.gguf")
LORA_SCALE = float(_config.get("lora", {}).get("scale", 1.0))
LORA_RANK = int(_config.get("lora", {}).get("rank", 16))
LORA_ALPHA = int(_config.get("lora", {}).get("alpha", 32))
DATASET_DIR = BASE_DIR / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Server
SERVER_HOST = _config.get("server", {}).get("host", "0.0.0.0")
SERVER_PORT = _config.get("server", {}).get("port", 8765)

# Paths
MODELS_DIR = BASE_DIR / "models"
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# WiFi (из .env)
WIFI_SSID = os.getenv("WIFI_SSID", "")
WIFI_PASSWORD = os.getenv("WIFI_PASSWORD", "")

# ========== STT ==========
STT_MODE = _config.get("stt", {}).get("mode", "local")
WHISPER_MODEL_SIZE = _config.get("stt", {}).get("whisper_model", "small")
WHISPER_DEVICE = _config.get("stt", {}).get("device", "cpu")
WHISPER_COMPUTE_TYPE = _config.get("stt", {}).get("compute_type", "int8")
WHISPER_MODELS = _config.get("stt", {}).get("whisper_models", [])

# ========== TTS ==========
TTS_MODE = _config.get("tts", {}).get("mode", "local")
SILERO_SPEAKER = _config.get("tts", {}).get("silero_speaker", "aidar")
SILERO_PITCH_SEMITONES = float(_config.get("tts", {}).get("silero_pitch_semitones", 1.5))
SILERO_RATE = float(_config.get("tts", {}).get("silero_rate", 1.05))
FREETTS_VOICE = _config.get("tts", {}).get("edge_voice", "ru-RU-DmitryNeural")
FREETTS_SPEED = float(_config.get("tts", {}).get("edge_speed", 1.0))
FREETTS_PITCH = _config.get("tts", {}).get("edge_pitch", "+15Hz")

# ========== LLM ==========
LLM_MODE = _config.get("llm", {}).get("mode", "local")

# Путь к локальной модели (.gguf) — берётся строго из config.yaml (llm.local_model_path),
# никакого захардкоженного имени файла по умолчанию. Обязателен, только если реально
# используется локальный режим — если сейчас mode="cloud", требовать его смысла нет.
_LOCAL_MODEL_PATH_STR = _config.get("llm", {}).get("local_model_path")
if LLM_MODE == "local" and not _LOCAL_MODEL_PATH_STR:
    raise ValueError(
        "config.yaml: llm.mode = 'local', но llm.local_model_path не задан. "
        "Укажи путь к .gguf файлу локальной модели (например: models/qwen2.5-7b-instruct-Q4_K_M.gguf)."
    )
LLM_MODEL_PATH = (BASE_DIR / _LOCAL_MODEL_PATH_STR) if _LOCAL_MODEL_PATH_STR else None

LLM_N_CTX = int(_config.get("llm", {}).get("n_ctx", 4096))
LLM_N_THREADS = int(_config.get("llm", {}).get("n_threads", 4))
LLM_TEMPERATURE = float(_config.get("llm", {}).get("temperature", 0.6))
LLM_REPEAT_PENALTY = float(_config.get("llm", {}).get("repeat_penalty", 1.15))
CLOUD_API_KEY = os.getenv("HF_TOKEN", _config.get("llm", {}).get("cloud_api_key", ""))

# Имя облачной модели — тоже строго из config.yaml (llm.cloud_model), без захардкоженного
# значения по умолчанию. Обязательно, только если реально используется cloud-режим.
# Формат — как в Hugging Face Inference Providers: "<model_id>:<провайдер|fastest|cheapest>",
# например "openai/gpt-oss-120b:fastest".
CLOUD_MODEL_NAME = _config.get("llm", {}).get("cloud_model")
if LLM_MODE == "cloud" and not CLOUD_MODEL_NAME:
    raise ValueError(
        "config.yaml: llm.mode = 'cloud', но llm.cloud_model не задан. "
        "Укажи имя облачной модели (например: openai/gpt-oss-120b:fastest)."
    )
CLOUD_MODEL_NAME = CLOUD_MODEL_NAME or ""

# Куратированные списки моделей для выпадающего списка в /panel (см. config.yaml
# llm.local_models / llm.cloud_models). Каждый элемент — dict с "id"/"label" и
# "path" (для local) или "model" (для cloud).
LLM_LOCAL_MODELS = _config.get("llm", {}).get("local_models", [])
LLM_CLOUD_MODELS = _config.get("llm", {}).get("cloud_models", [])

# Vision
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
ENABLE_POSE_TRACKING = _config.get("vision", {}).get("pose_tracking", True)

# === Слежение за лицом (OpenCV) ===
# Робот поворачивает "голову" (серво pan/tilt) к лицу собеседника ТОЛЬКО пока
# активен диалог. Вне диалога камера продолжает работать (для панели
# мониторинга), но серво головы не дёргаются.
FACE_TRACKING_ENABLED = _config.get("vision", {}).get("face_tracking", True)
DIALOG_ACTIVE_TIMEOUT_SEC = float(_config.get("vision", {}).get("dialog_active_timeout_sec", 20))
# "Предварительное" окно — продлевается уже на VAD speech_start/speech, ДО распознавания.
# Даёт голове сразу начать реагировать, но быстро отпускает, если ничего не подтвердилось
# (шум/кашель). Настоящий диалог держится через DIALOG_ACTIVE_TIMEOUT_SEC (см. is_dialog_active).
PROVISIONAL_DIALOG_TIMEOUT_SEC = float(_config.get("vision", {}).get("provisional_dialog_timeout_sec", 4))
FACE_PAN_SERVO = int(_config.get("vision", {}).get("face_pan_servo", 16))
FACE_TILT_SERVO = int(_config.get("vision", {}).get("face_tilt_servo", 17))
FACE_PAN_GAIN = float(_config.get("vision", {}).get("face_pan_gain", 45))
FACE_TILT_GAIN = float(_config.get("vision", {}).get("face_tilt_gain", 30))

# --- Сглаживание движения головы + "мёртвая зона" (имитация поля зрения) ---
HEAD_SMOOTHING_ALPHA = float(_config.get("vision", {}).get("head_smoothing_alpha", 0.3))
FACE_DEADZONE_X = float(_config.get("vision", {}).get("face_deadzone_x", 0.12))
FACE_DEADZONE_Y = float(_config.get("vision", {}).get("face_deadzone_y", 0.12))

# --- Плавный выход из диалога: голова сама едет обратно в центр, а не замирает резко ---
EXIT_EASE_ALPHA = float(_config.get("vision", {}).get("exit_ease_alpha", 0.15))
EXIT_EASE_EPSILON = float(_config.get("vision", {}).get("exit_ease_epsilon", 0.02))

# --- Детектор лица: OpenCV DNN (основной) с откатом на Haar Cascade ---
FACE_DETECTOR_PROTOTXT = MODELS_DIR / "face_detector" / "deploy.prototxt"
FACE_DETECTOR_MODEL = MODELS_DIR / "face_detector" / "res10_300x300_ssd_iter_140000.caffemodel"
FACE_DETECTOR_CONFIDENCE = float(_config.get("vision", {}).get("face_detector_confidence", 0.6))
FACE_DETECTOR_INPUT_SIZE = int(_config.get("vision", {}).get("face_detector_input_size", 300))
# Haar Cascade используется только если DNN-модель не скачана (см. scripts/download_face_detector.py)
FACE_CASCADE_SCALE_FACTOR = float(_config.get("vision", {}).get("face_cascade_scale_factor", 1.1))
FACE_CASCADE_MIN_NEIGHBORS = int(_config.get("vision", {}).get("face_cascade_min_neighbors", 5))
FACE_CASCADE_MIN_SIZE = int(_config.get("vision", {}).get("face_cascade_min_size", 60))

# --- Трекер лиц между кадрами (простой, по расстоянию центров) ---
FACE_TRACKER_MAX_DISTANCE = int(_config.get("vision", {}).get("face_tracker_max_distance", 90))
FACE_TRACKER_MAX_MISSED_FRAMES = int(_config.get("vision", {}).get("face_tracker_max_missed_frames", 12))

# --- Выбор говорящего по движению губ ---
LIP_ACTIVITY_WINDOW_SEC = float(_config.get("vision", {}).get("lip_activity_window_sec", 0.8))
MIN_LIP_SAMPLES_FOR_DECISION = int(_config.get("vision", {}).get("min_lip_samples_for_decision", 3))

# --- Троттлинг тяжёлых детекторов (YOLO/Pose не нужно гонять на каждый кадр) ---
# Детекция лица + трекер + активность губ НЕ троттлятся — от их частоты напрямую
# зависит плавность слежения (сглаживание) и скорость выбора говорящего по губам.
YOLO_DETECTION_INTERVAL_SEC = float(_config.get("vision", {}).get("yolo_detection_interval_sec", 1.0))
POSE_DETECTION_INTERVAL_SEC = float(_config.get("vision", {}).get("pose_detection_interval_sec", 0.2))

# === Видео в панели мониторинга ===
VIDEO_PANEL_MIN_INTERVAL_SEC = float(_config.get("vision", {}).get("panel_frame_interval_sec", 0.15))
VIDEO_PANEL_JPEG_QUALITY = int(_config.get("vision", {}).get("panel_jpeg_quality", 70))

# --- Частота отправки servo_update на ESP32 ---
# Интерполяция углов уже сглаживается на самой прошивке (interpolateServos), поэтому
# слать новую цель на КАЖДЫЙ видеокадр избыточно — троттлим отправку по сети,
# сам расчёт (детекция/трекинг/сглаживание) при этом остаётся на полной частоте.
SERVO_UPDATE_MIN_INTERVAL_SEC = float(_config.get("vision", {}).get("servo_update_min_interval_sec", 0.1))

# Audio
SAMPLE_RATE = int(_config.get("audio", {}).get("sample_rate", 16000))
CHUNK_DURATION_MS = int(_config.get("audio", {}).get("chunk_duration_ms", 30))
VAD_AGGRESSIVENESS = int(_config.get("audio", {}).get("vad_aggressiveness", 2))
SILENCE_TIMEOUT_MS = int(_config.get("audio", {}).get("silence_timeout_ms", 1500))

# Servo config
SERVO_CFG = _config.get("servos", {})
SERVO_CONFIG = {
    "pca9685_channels": SERVO_CFG.get("pca9685_channels", 16),
    "pca9685_address": SERVO_CFG.get("pca9685_address", 0x40),
    "pca9685_freq": SERVO_CFG.get("pca9685_freq", 50),
    "extra_servos_pins": SERVO_CFG.get("extra_pins", [17, 18]),
    "min_angle": SERVO_CFG.get("min_angle", 0),
    "max_angle": SERVO_CFG.get("max_angle", 180),
}

# Анимации
ANIMATIONS = {
    "wave": [
        {"time": 0, "servos": [90]*18},
        {"time": 200, "servos": [90, 45, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 400, "servos": [90, 135, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 600, "servos": [90, 45, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 800, "servos": [90]*18},
    ],
    "nod": [
        {"time": 0, "servos": [90]*18},
        {"time": 300, "servos": [90, 90, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 600, "servos": [90, 90, 120, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 900, "servos": [90]*18},
    ],
    "shake_head": [
        {"time": 0, "servos": [90]*18},
        {"time": 200, "servos": [90, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 400, "servos": [90, 90, 90, 120, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 600, "servos": [90, 90, 90, 60, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]},
        {"time": 800, "servos": [90]*18},
    ],
    "idle": [
        {"time": 0, "servos": [90]*18},
    ],
}

# === Авторизация панели мониторинга / устройства ===
# Один общий пароль (это личный робот, а не сервис на много пользователей).
# Пусто по умолчанию = авторизация выключена (как было раньше — для удобства
# первоначальной настройки, включается явно заданием пароля в .env).
PANEL_PASSWORD = os.getenv("PANEL_PASSWORD", "")

# Общий секрет, который ESP32 присылает вместе с "ping" для подтверждения,
# что это доверенное устройство, а не случайный WS-клиент, притворяющийся им.
# Пусто = не проверяем (совместимость со старой прошивкой без DEVICE_KEY).
DEVICE_KEY = os.getenv("DEVICE_KEY", "")

# Секрет для подписи cookie-сессии панели (HMAC). Если не задан явно в .env —
# генерируется один раз и сохраняется в файл, чтобы перезапуск сервера
# не разлогинивал всех пользователей панели.
_SESSION_SECRET_FILE = BASE_DIR / "config" / ".session_secret"


def _load_or_create_session_secret() -> str:
    env_secret = os.getenv("SESSION_SECRET", "")
    if env_secret:
        return env_secret
    if _SESSION_SECRET_FILE.exists():
        return _SESSION_SECRET_FILE.read_text(encoding="utf-8").strip()
    import secrets
    new_secret = secrets.token_hex(32)
    _SESSION_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_SECRET_FILE.write_text(new_secret, encoding="utf-8")
    return new_secret


SESSION_SECRET = _load_or_create_session_secret()
SESSION_COOKIE_NAME = "soren_session"
SESSION_SHORT_MAX_AGE_SEC = 12 * 3600            # без "запомнить это устройство" — 12 часов
SESSION_REMEMBER_MAX_AGE_SEC = 90 * 24 * 3600    # с галочкой — 90 дней