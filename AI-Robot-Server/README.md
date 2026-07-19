# 🤖 Robot AI Server — Сорен v3.0

Локальный сервер ИИ для роботизированной совы **Сорен** на ESP32-S3 с голосовым управлением, компьютерным зрением и управлением 18 сервоприводами.

Сорен — амбарная сова, cтраж Великого Древа Га'Хула. Персонаж из книг Кэтрин Ласки «Легенды ночных стражей», воплощённый в роботе с полноценным AI-ассистентом.

---

## ✨ Переключение Local / Cloud

Каждый AI модуль (STT, LLM, TTS) имеет **отдельную кнопку** переключения между локальным и облачным режимом.

| Модуль     | Локально                | Облачно                                   |
| ---------- | ----------------------- | ----------------------------------------- |
| **STT** 🎤 | faster-whisper (~500MB) | Нет                                       |
| **LLM** 🧠 | Qwen 7B GGUF (~4.5GB)   | GitHub Models API (gpt-4o-mini)           |
| **TTS** 🔊 | Silero (~120MB)         | edge-tts (Microsoft Edge TTS, бесплатный) |

---

## 🛠 Технологический стек

**FastAPI + faster-whisper + llama-cpp-python + PyTorch/Silero + edge-tts + YOLOv8 + MediaPipe + OpenCV + webrtcvad**

| Компонент                | Технология                                          | Назначение                           |
| ------------------------ | --------------------------------------------------- | ------------------------------------ |
| WebSocket/HTTP сервер    | **FastAPI + Uvicorn + websockets**                  | API для ESP32 и веб-панели           |
| **STT** (Speech-to-Text) | **faster-whisper**                                  | Распознавание речи (только локально) |
| **LLM** (Language Model) | **llama-cpp-python** / **GitHub Models API**        | Генерация ответов Сорена             |
| **TTS** (Text-to-Speech) | **Silero** (PyTorch) / **edge-tts**                 | Синтез речи                          |
| **Vision**               | **Ultralytics YOLOv8** + **MediaPipe** + **OpenCV** | Детекция объектов, позы, лица        |
| **Audio VAD**            | **webrtcvad**                                       | Определение начала/конца речи        |
| **Audio processing**     | **soundfile + librosa**                             | Конвертация форматов, ресемплинг     |
| **Config**               | **python-dotenv**                                   | Переменные окружения                 |

---

## 📁 Структура проекта

```
robot_server/
│
├── websocket_server.py      # Главный WebSocket сервер + веб-панель
├── requirements.txt         # Зависимости
├── .env                     # Конфигурация (создаётся из .env.example)
├── .env.example             # Шаблон конфигурации
├── .gitattributes
├── .gitignore
├── README.md                # Этот файл
│
├── config/                  # Конфигурация
│   ├── __init__.py
│   └── settings.py          # Настройки сервера, AI, сервоприводов, анимаций
│
├── modules/                 # AI модули
│   ├── __init__.py
│   ├── stt.py               # STT: local (faster-whisper)
│   ├── tts.py               # TTS: local (Silero) + cloud (edge-tts)
│   ├── llm.py               # LLM: local (llama.cpp) + cloud (GitHub Models)
│   ├── robot_brain.py       # Оркестратор + эмоциональный движок
│   ├── vision.py            # YOLO + MediaPipe + OpenCV
│   ├── servo_controller.py  # Управление 18 сервоприводами
│   ├── audio_buffer.py      # Аудио буфер с VAD
│   └── fuzzy_matcher.py     # Адаптивная коррекция речи
│
├── character/               # Данные персонажа Сорена
│   ├── Soren.txt            # System prompt (личность, стиль, философия)
│   ├── Soren_rag_chunks.jsonl  # RAG: биография, отношения, мир
│   └── Soren_emotions.json  # Движения
│
├── models/                  # AI модели (создаётся автоматически)
│   ├── qwen2.5-7b-instruct-q4_k_m.gguf  # LLM (~4.5GB)
│   ├── silero_v4_ru.pt      # TTS (~120MB)
│   └── yolov8n.pt           # Vision (~6MB)
│
├── audio_cache/             # Кэш синтезированной речи
│
├── venv/                    # Виртуальное окружение
│
├── check_model.py       # Проверка целостности GGUF модели
├── download_models.py   # Загрузка всех моделей
├── download_qwen.py     # Скачивание Qwen GGUF (авто / ручное)
├── download_yolo.py     # Скачивание YOLOv8
├── merge_gguf.py        # Сборка GGUF из частей
├── test_server.py       # Тест сервера без ESP32
├── test_github.py       # Тест GitHub Models API
└── test_all.py          # Полное тестирование всех модулей
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Настройка конфигурации

```bash
cp .env.example .env
# Отредактируй .env — укажи GITHUB_MODELS_KEY если хочешь облачный LLM
```

### 3. Загрузка моделей

```bash
# Qwen LLM (автоматически через huggingface-cli или вручную)
python download_qwen.py

# YOLO (автоматически через ultralytics)
python download_yolo.py

# Whisper и Silero скачаются при первом запуске сервера
```

### 4. Запуск сервера

```bash
python websocket_server.py
```

Открой панель управления: **http://localhost:8765/panel**

---

## ⚙️ Переменные окружения (.env)

```bash
# === Режимы AI ===
STT_MODE=local              # Только local (облачный недоступен без VPN)
TTS_MODE=local              # local (Silero) или cloud (edge-tts)
LLM_MODE=local              # local (llama.cpp) или cloud (GitHub Models)

# === GitHub Models API (для облачного LLM) ===
GITHUB_MODELS_KEY=ghp_xxx   # Токен с https://github.com/settings/tokens
GITHUB_MODELS_NAME=gpt-4o-mini

# === Локальные модели ===
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
LLM_MODEL_PATH=./models/qwen2.5-7b-instruct-q4_k_m.gguf
LLM_N_CTX=4096
LLM_N_THREADS=4
LLM_TEMPERATURE=0.6
SILERO_SPEAKER=xenia

# === Облачный TTS (edge-tts) ===
FREETTS_VOICE=ru-RU-SvetlanaNeural
FREETTS_SPEED=1.0

# === Vision ===
YOLO_MODEL=./models/yolov8n.pt
ENABLE_POSE_TRACKING=true

# === Audio ===
SAMPLE_RATE=16000
CHUNK_DURATION_MS=30
VAD_AGGRESSIVENESS=2
SILENCE_TIMEOUT_MS=1500

# === Сервер ===
SERVER_HOST=0.0.0.0
SERVER_PORT=8765

# === Сервоприводы ===
# 16 каналов PCA9685 (I2C 0x40) + 2 на GPIO (17, 18)
```

---

## 🔌 API Endpoints

| Endpoint      | Метод | Описание                             |
| ------------- | ----- | ------------------------------------ |
| `/`           | GET   | Статус сервера                       |
| `/status`     | GET   | Полный статус + режимы AI            |
| `/audio_mode` | POST  | Переключение аудио (input/output)    |
| `/ai_mode`    | POST  | Переключение AI модуля (stt/tts/llm) |
| `/speak`      | POST  | Текст → LLM → TTS → аудио            |
| `/voice`      | POST  | Аудио → STT → LLM → TTS → аудио      |
| `/panel`      | GET   | HTML панель управления               |
| `/ws`         | WS    | WebSocket для ESP32                  |

### WebSocket команды (ESP32)

```json
// Управление сервоприводом
{"type": "servo", "id": 0, "angle": 90}

// Управление всеми сервоприводами
{"type": "servo_multi", "angles": [90,90,90,...]}

// Анимация
{"type": "animation", "name": "wave"}

// Текстовый ввод
{"type": "text", "text": "Привет, Сорен!"}

// Статус
{"type": "get_status"}

// Очистка истории
{"type": "clear_history"}

// Переключение режима
{"type": "set_mode", "module": "llm", "mode": "cloud"}
```

---

## 🎛️ Панель управления

Веб-панель (`/panel`) содержит:

- **3 блока переключения AI** — STT, LLM, TTS (Local / Cloud)
- **2 переключателя аудио** — Ввод (микрофон) и Вывод (динамик)
- **Голосовой чат** — кнопка микрофона для разговора
- **Текстовый чат** — поле ввода сообщений
- **Анимации** — wave, nod, shake_head, idle
- **18 сервоприводов** — индивидуальное управление каждым

---

## 🦉 Персонаж Сорен

### Эмоциональный движок

7 эмоций с уникальными позами сервоприводов и LED-глаз:

| Эмоция          | Триггеры                       | LED                   | Поза                              |
| --------------- | ------------------------------ | --------------------- | --------------------------------- |
| **Спокойствие** | обычный разговор               | soft_white_low        | нейтральная                       |
| **Печаль**      | клудд, потеря, смерть          | dim_blue_pulse        | голова опущена, крылья прижаты    |
| **Гнев**        | угроза, предательство          | bright_orange_flicker | голова наклонена, когти сжаты     |
| **Нежность**    | пеллиппер, друг, благодарность | warm_yellow_glow      | мягкий наклон, крылья расслаблены |
| **Решимость**   | битва, миссия, защита          | steady_white_bright   | взгляд прямо, крылья готовы       |
| **Удивление**   | неожиданность, вещий сон       | bright_white_flash    | голова поднята, глаза широко      |
| **Усталость**   | долгий разговор, рана          | dim_amber_slow        | голова опущена, всё расслаблено   |

### RAG — Знания Сорена

49 чанков знаний в `character/Soren_rag_chunks.jsonl`:

- **Биография** (18 чанков) — детство, Сант-Эголиус, путешествие, Древо
- **Отношения** (20 чанков) — Клудд, Гильфи, Эзилриб, Пеллиппер, Эглантина
- **Мир** (7 чанков) — Га'Хул, Сант-Эголиус, Чистые, крупинки
- **Философия** (7 чанков) — серебряная душа, надежда, сила, любовь

### Fuzzy Matching

Адаптивная коррекция распознанной речи — исправляет ошибки Whisper

---

## 🔧 Аппаратная часть

### Сервоприводы (18 шт.)

| ID      | Назначение                         | Диапазон |
| ------- | ---------------------------------- | -------- |
| S0–S2   | Голова (поворот, наклон, клюв)     | 0–180°   |
| S3–S8   | Левое крыло (3) + Правое крыло (3) | 0–180°   |
| S9–S12  | Тело + хвост                       | 0–180°   |
| S13–S15 | Лапы                               | 0–180°   |
| S16–S17 | Глаза (LED-управление)             | 0–180°   |

### Подключение

- **PCA9685** — I2C (SCL/SDA), адрес `0x40`, частота 50Hz → 16 сервоприводов
- **GPIO 17, 18** — 2 дополнительных сервопривода (ESP32-S3)
- **Микрофон** — I2S или USB
- **Динамик** — I2S или PWM
- **Камера** — ESP32-CAM или USB

---

## 🧪 Тестирование

```bash
# Проверка целостности LLM модели
python check_model.py

# Тест сервера без ESP32
python test_server.py

# Тест GitHub Models API
python test_github.py

# Полное тестирование всех модулей
python test_all.py
```

---

## 📋 Требования

- **Python**: 3.10 – 3.11
- **ОЗУ**: минимум 8 GB (для Qwen 7B)
- **Диск**: ~5 GB под модели
- **OS**: Windows 10/11, Linux, macOS

---

## 📜 Лицензия

MIT License
