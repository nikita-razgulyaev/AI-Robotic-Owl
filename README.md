# 🦉 AI Robotic Owl

Интерактивный робот-сова с искусственным интеллектом, объединяющий 3D-печатную механику, сервоприводное управление и серверную часть с AI.

---

## 📁 Структура проекта

```
ai-robotic-owl/
├── AI-Robot-Server/          # Серверная часть (AI, API, WebSocket)
├── 3d-model-for-owl/         # 3D-модели для печати
├── arduino-owl-model/        # Прошивка для микроконтроллера (ESP32)
├── .gitattributes
└── README.md

```

---

## 🧠 Описание компонентов

### 1. `ai-robot-server/`

Серверная часть проекта, отвечающая за:

- Web-интерфейс для мониторинга и ручного управления
- Распознавание голосовых команд (STT) и воспроизведение речи (TTS)
- Интеграцию с языковыми моделями (LLM) для генерации ответов

**Стек:**

- **Model:** Arduino
- **Server:** React, TypeScript, Python, FastAPI, Uvicorn, WebSocket, Llama-cpp-python, Faster-whisper, Edge-tts, OpenCV

### 2. `3d-model-for-owl/`

Набор 3D-моделей для FDM печати:

```
├── fusion/
│      ├── head.f3d
│      ├── body.f3d
│      ├── wing box.f3d
│      ├── wings.f3d
│      ├── tail box.f3d
│      ├── tail.f3d
│      ├── leg box.f3d
│      ├── legs.f3d
└── stl/
│      ├── head/...
│      ├── body/...
│      ├── wing box/...
│      ├── wings/...
│      ├── tail box/...
│      ├── tail/...
│      ├── leg box/...
│      └── legs/...
```

**Форматы:** `.stl`

### 3. `arduino-owl-model/`

Прошивка для микроконтроллера ESP32 с драйвером сервоприводов PCA9685.

**Особенности:**

- Управление **18 сервоприводами**
- Поддержка Wi-Fi для связи с сервером
- Протокол обмена: JSON через Serial / WebSocket
- Плавные интерполированные движения

**Аппаратная платформа:**
| Компонент | Спецификация |
|-----------|-------------|
| МК | ESP32 (Wi-Fi + Bluetooth) |
| Драйвер серво | PCA9685 (16-канальный, I2C) |
| Доп. серво | 1 сервопривод клюва напрямую на ESP32 |
| Сервоприводы | 18× SG90 / MG90S |
| Питание серво | Внешний BEC 5V 6A+ |

---

## 🚀 Быстрый старт

### Аппаратная сборка

1. Распечатайте все детали из `3d-model-for-owl/` (рекомендуемый материал: PETG/PLA+, заполнение 20–30%).
2. Установите сервоприводы в корпус согласно схеме сборки.
3. Подключите PCA9685 к ESP32 по I2C (SDA → GPIO21, SCL → GPIO22).
4. Подключите сервопривод клюва напрямую к GPIO ESP32.
5. Обеспечьте отдельное питание 5V для сервоприводов (общая земля с ESP32).

### Загрузка прошивки

```bash
cd arduino-owl-model
# Откройте проект в Arduino IDE / PlatformIO
# Установите библиотеки: Adafruit PWM Servo Driver, ESP32Servo
# Выберите плату: ESP32 Dev Module
# Загрузите прошивку
```

### Запуск сервера

```bash
cd ai-robot-server
# Установите зависимости (см. README внутри папки)
pip install -r requirements.txt  # или npm install
# Запустите сервер
python app.py  # или npm start
```

---

## 🔌 Схема подключения

```
ESP32
 ├── GPIO21 (SDA)  ──► PCA9685 SDA
 ├── GPIO22 (SCL)  ──► PCA9685 SCL
 ├── 3.3V/GND      ──► PCA9685 VCC/GND (логика)
 ├── GPIO_X        ──► Серво клюва (шИМ)
 └── USB/5V        ──► Питание ESP32

PCA9685
 ├── V+ / GND      ──► Внешний BEC 5V (питание серво)
 └── PWM 0–15      ──► 16 сервоприводов
```

---

## 📡 Протокол взаимодействия

Сервер ↔ МК обмениваются JSON-сообщениями:

**Команда от сервера:**

```json
{
    "type": "set_pose",
    "servos": {
        "neck": 90,
        "wing_left": 45,
        "wing_right": 135,
        "beak": 20,
        "eyelid_left": 100
    },
    "duration": 500
}
```

**Статус от МК:**

```json
{
  "type": "status",
  "battery": 4.15,
  "servo_positions": { "neck": 90, ... },
  "connected": true
}
```

---

## 🎯 Возможности

- 🎙️ Голосовое управление и диалоги через AI
- 🦉 Реалистичные движения: поворот головы, моргание, взмах крыльев, открытие клюва
- 🧠 Генерация эмоций и анимаций на основе контекста разговора
- 📱 Web-интерфейс для ручного управления и калибровки
- 🔧 OTA-обновление прошивки ESP32

---

## 📋 Зависимости

### Прошивка (Arduino)

- [Adafruit PWM Servo Driver Library](https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library)
- [ESP32Servo](https://github.com/madhephaestus/ESP32Servo)
- ArduinoJson

### Сервер

- См. `ai-robot-server/requirements.txt`

---

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку: `git checkout -b feature/awesome-feature`
3. Закоммитьте изменения: `git commit -m 'Add awesome feature'`
4. Отправьте в ветку: `git push origin feature/awesome-feature`
5. Откройте Pull Request

---

## 📄 Лицензия

MIT License
