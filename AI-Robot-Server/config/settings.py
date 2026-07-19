"""Конфигурация сервера робота Сорена"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# === Character / Knowledge System ===
CHARACTER_DIR = Path(os.getenv("CHARACTER_DIR", "./character"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Server
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8765"))

# Paths
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))
AUDIO_CACHE_DIR = Path(os.getenv("AUDIO_CACHE_DIR", "./audio_cache"))
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# WiFi
WIFI_SSID = os.getenv("WIFI_SSID", "")
WIFI_PASSWORD = os.getenv("WIFI_PASSWORD", "")

# ========== STT ==========
# Режим: только "local" (faster-whisper) — облачные STT без VPN не работают
STT_MODE = os.getenv("STT_MODE", "local")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ========== TTS ==========
# Режим: "local" (Silero) или "cloud" (FreeTTS / edge-tts)
TTS_MODE = os.getenv("TTS_MODE", "local")
SILERO_SPEAKER = os.getenv("SILERO_SPEAKER", "xenia")
# Облачный TTS (FreeTTS — edge-tts, бесплатный Microsoft Edge TTS)
FREETTS_VOICE = os.getenv("FREETTS_VOICE", "ru-RU-SvetlanaNeural")
FREETTS_SPEED = float(os.getenv("FREETTS_SPEED", "1.0"))

# ========== LLM ==========
# Режим: "local" (llama.cpp) или "cloud" (GitHub Models API)
LLM_MODE = os.getenv("LLM_MODE", "local")
# Локальный LLM
LLM_MODEL_PATH = Path(os.getenv("LLM_MODEL_PATH", "./models/qwen2.5-7b-instruct-q4_k_m.gguf"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "4"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
# Облачный LLM (GitHub Models API — бесплатный, без VPN)
GITHUB_MODELS_KEY = os.getenv("GITHUB_MODELS_KEY", "")
GITHUB_MODELS_NAME = os.getenv("GITHUB_MODELS_NAME", "gpt-4o-mini")

# Vision
YOLO_MODEL = Path(os.getenv("YOLO_MODEL", "./models/yolov8n.pt"))
ENABLE_POSE_TRACKING = os.getenv("ENABLE_POSE_TRACKING", "true").lower() == "true"

# Audio
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHUNK_DURATION_MS = int(os.getenv("CHUNK_DURATION_MS", "30"))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
SILENCE_TIMEOUT_MS = int(os.getenv("SILENCE_TIMEOUT_MS", "1500"))

# Servo config (18 сервоприводов: 16 на PCA9685 + 2 на GPIO)
SERVO_CONFIG = {
    "pca9685_channels": 16,
    "pca9685_address": 0x40,
    "pca9685_freq": 50,
    "extra_servos_pins": [17, 18],
    "min_angle": 0,
    "max_angle": 180,
}

# Анимации (ключевые кадры для сервоприводов)
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