/* 🦉 Robot ESP32-S3 Firmware - Soren */

#include <WiFi.h>
#include <ETH.h>
#include <SPI.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <driver/i2s.h>
#include <esp_camera.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>
#include <esp_task_wdt.h>

// ==================== КОНФИГУРАЦИЯ ====================
const char* WIFI_SSID = "TP-Link_208E";
const char* WIFI_PASSWORD = "31267649";
const char* SERVER_HOST = "192.168.50.10";  // статический IP ПК на адаптере TP-LINK USB-Ethernet
const char* DEVICE_KEY = "";
const int SERVER_PORT = 8765;

// ---------- ETHERNET (модуль W5500 по SPI) ----------
// GPIO0/45/46 — strapping-пины ESP32-S3: их уровень чип читает В МОМЕНТ
// СБРОСА, чтобы выбрать режим загрузки. Внешнее устройство на них (было тут
// раньше — 0=MOSI, 46=CS) срывает это определение при каждом
// авто-сбросе/заливке ("Wrong boot mode detected"). Поэтому SPI переехал на
// GPIO19/20/43/44 — ни один из них не strapping и не PSRAM (33-37).
//
// ВАЖНО: GPIO43/44 по умолчанию — консольный UART на CP2102 ("UART"-порт).
// Заняв их под Ethernet, Serial-вывод для отладки нужно смотреть через
// ВТОРОЙ, нативный USB-порт платы ("USB"), и в Arduino IDE включить
// Tools → USB CDC On Boot → Enabled — иначе Serial.print() будет уходить в
// никуда.
// ПЛАТА: ESP32-S3-DevKitC-1, модуль WROOM-1 -N16R8 (Octal PSRAM).
// ЭКСПЕРИМЕНТ: пробуем вернуть SCK/MISO на strapping-пины 0/46, но НЕ в
// тех же ролях, что раньше сорвали загрузку (было 0=MOSI, 46=CS).
// Теория: SCK всегда ведёт ESP32 (мастер), W5500 (ведомый) его не тянет;
// MISO — вход для ESP32, GPIO46 к тому же input-only. Раньше проблема
// была именно с MOSI/CS на strapping-пинах — не факт, что SCK/MISO
// поведут себя так же. ПРОВЕРИТЬ: после заливки следить за выводом
// esptool — если снова "Wrong boot mode detected"/сбои автосброса на
// каждой заливке, откатывать на 43/44 (см. предыдущую версию комментария
// в истории правок).
#define ETH_PHY_TYPE   ETH_PHY_W5500
#define ETH_PHY_ADDR   1
#define ETH_PHY_CS     20
#define ETH_PHY_IRQ    -1
#define ETH_PHY_RST    48
#define ETH_SPI_SCK    0
#define ETH_SPI_MISO   46
#define ETH_SPI_MOSI   19

#define ETH_CONNECT_TIMEOUT_MS 4000  // сколько ждём линк+DHCP на Ethernet при старте,
                                      // прежде чем откатиться на Wi-Fi
#define WIFI_FALLBACK_TIMEOUT_MS 20000

bool ethConnected = false;      // Ethernet реально поднят и получил IP прямо сейчас
bool wifiFallbackActive = false; // на этом старте работаем через Wi-Fi (Ethernet не найден/не подключен)

// ---------- СЕРВОПРИВОДЫ ----------
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);
const int PCA_FREQ = 50;
const int SERVO_17_PIN = 38;
const int SERVO_18_PIN = 39;

// ESP32-S3 аппаратно поддерживает МАКСИМУМ 14 бит разрешения на канал LEDC
// (в отличие от классического ESP32, где было 20 бит). Раньше здесь стояло 16 —
// ledcAttach() с недопустимым разрешением тихо возвращает false и канал вообще
// не настраивается, поэтому ledcWrite() на GPIO 38/39 ничего не делал — именно
// поэтому сервы HEAD_ROLL/BEAK "не работали".
#define SERVO_GPIO_LEDC_RES_BITS 14
#define SERVO_GPIO_LEDC_MAX_DUTY ((1 << SERVO_GPIO_LEDC_RES_BITS) - 1)  // 16383

// -------------------- ИМЕНОВАННЫЕ ИНДЕКСЫ СЕРВО --------------------
#define L_FLAP          0   // Левое крыло — взмах
#define L_FOLD          1   // Левое крыло — складывание
#define R_FLAP          2   // Правое крыло — взмах
#define R_FOLD          3   // Правое крыло — складывание
#define WING_TILT       4   // Наклон крыльев (общий)
#define L_LEG_TURN      5   // Левая нога — поворот
#define L_LEG_FOLD      6   // Левая нога — сгиб
#define L_LEG_LIFT      7   // Левая нога — подъём
#define R_LEG_TURN      8   // Правая нога — поворот
#define R_LEG_FOLD      9   // Правая нога — сгиб
#define R_LEG_LIFT      10  // Правая нога — подъём
#define TAIL_YAW        11  // Хвост — поворот
#define TAIL_SPREAD     12  // Хвост — раскрытие
#define TAIL_LIFT       13  // Хвост — подъём
#define HEAD_YAW        14  // Голова — влево/вправо (канал PCA9685)
#define HEAD_PITCH      15  // Голова — вперёд/назад (канал PCA9685)
#define HEAD_ROLL_INDEX 16  // Голова — крен вбок. Физически SERVO_17_PIN = GPIO 38
#define BEAK_INDEX      17  // Открытие клюва.       Физически SERVO_18_PIN = GPIO 39

#define PCA_SERVO_COUNT 16  // сколько первых индексов (0-15) идут через PCA9685

#define SERVO_MAX_STEP 3        // потолок скорости (град/тик = 10мс) — верхний предел даже для больших скачков цели
#define SERVO_EASE_FACTOR 0.12  // доля оставшегося расстояния до цели, проходимая за один тик (плавное подтормаживание к концу)
#define SERVO_UPDATE_MS 10
#define SERVO_DEAD_ZONE 1

// ---------- ПОДСВЕТКА ----------
#define STAND_LED_PIN   47
#define STAND_LED_COUNT 60
Adafruit_NeoPixel standLed(STAND_LED_COUNT, STAND_LED_PIN, NEO_GRB + NEO_KHZ800);

#define WARM_WHITE_R 255
#define WARM_WHITE_G 200
#define WARM_WHITE_B 140
bool standBacklightOn = false;

// ---------- I2S ПИНЫ ----------
#define I2S_MIC_BCLK    1
#define I2S_MIC_WS      3
#define I2S_MIC_DIN     2

#define I2S_AMP_BCLK    40
#define I2S_AMP_LRC     41
#define I2S_AMP_DOUT    42

#define SAMPLE_RATE 16000
#define SPEAKER_SAMPLE_RATE 48000

// ---------- КАМЕРА ----------
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   15
#define XCLK_GPIO_NUM    -1
#define SIOD_GPIO_NUM    14
#define SIOC_GPIO_NUM    21

// Возвращено на QVGA (320x240) — на HQVGA (240x176) детектор лица на
// зашумлённой ИК-картинке стабильно не проходил порог уверенности
// (FACE_DETECTOR_CONFIDENCE). Прирост FPS слежения берём не за счёт
// разрешения, а за счёт освобождённого CV-потока (убрано слежение за
// губами) и CAM_FPS_FACE ниже.
#define DETECT_FRAME_SIZE  FRAMESIZE_QVGA

#define CAM_FPS_IDLE  3
#define CAM_FPS_FACE  15
unsigned long videoFrameIntervalMs = 1000 / CAM_FPS_IDLE;
unsigned long lastVideoFrameMs = 0;
bool lastFaceDetected = false;

// ==================== ГЛОБАЛЬНЫЕ ====================
WebSocketsClient webSocket;
bool isConnected = false;

int currentServoAngles[18];
int targetServoAngles[18];
unsigned long lastServoUpdateMs = 0;

String currentEyeLed = "soft_white_low";
String audioMode = "robot";

// ---------- МИКРОФОН ----------
#define AUDIO_BUFFER_SIZE 1024
int16_t audioBuffer[AUDIO_BUFFER_SIZE];

#define AUDIO_CHUNK_SAMPLES 480
#define AUDIO_CHUNK_BYTES (AUDIO_CHUNK_SAMPLES * 2)

uint8_t micAccum[AUDIO_CHUNK_BYTES * 8];
size_t micAccumLen = 0;

#define MIC_NOISE_GATE 150
#define MIC_SILENCE_TIMEOUT 1000
unsigned long silenceStartMs = 0;
bool micStreamConfirmed = false;

// ---------- ДИНАМИК (ring buffer) ----------
#define SPK_BUFFER_SIZE 131072
uint8_t* spkRingBuffer = nullptr;
volatile size_t spkWriteIdx = 0;
volatile size_t spkReadIdx = 0;
volatile size_t spkDataLen = 0;

#define SPK_PREBUFFER_BYTES     12000  // ~125 мс на 48кГц/16-бит — запас на джиттер сети
#define SPK_PREBUFFER_STALL_MS  350    // столько ждём тишину, прежде чем считать накопление законченным
#define SPK_END_SILENCE_MS      900    // тишина дольше этого = фраза реально закончилась
bool spkPrebuffering = true;
bool spkEnded = true;               // true = сейчас полная тишина, новых данных не ждём
volatile unsigned long lastSpkPushMs = 0;

// ---------- ВОССТАНОВЛЕНИЕ ПОСЛЕ ОБРЫВА WI-FI ----------
bool utteranceInterrupted = false;

// ---------- КАМЕРА ----------
static uint8_t videPacket[16384];

// ---------- HEARTBEAT ----------
#define HEARTBEAT_INTERVAL_MS 30000
#define HEARTBEAT_TIMEOUT_MS  60000
unsigned long lastHeartbeatSentMs = 0;
unsigned long lastHeartbeatRecvMs = 0;

// ---------- ЭХО-ПОДАВЛЕНИЕ ----------
// Время (мс), на которое микрофон "глухнет" после окончания аудио в буфере,
// чтобы не схватить реверберацию из динамика
#define MIC_DEAF_MS_AFTER_AUDIO 400
static unsigned long lastSpkActiveMs = 0;

// ==================== СЕТЬ (Ethernet первым, Wi-Fi — резерв) ====================
void onNetworkEvent(arduino_event_id_t event, arduino_event_info_t info) {
  switch (event) {
    case ARDUINO_EVENT_ETH_START:
      Serial.println("[ETH] Интерфейс стартовал");
      ETH.setHostname("soren-owl");
      break;
    case ARDUINO_EVENT_ETH_CONNECTED:
      Serial.println("[ETH] Кабель на линке");
      break;
    case ARDUINO_EVENT_ETH_GOT_IP:
      Serial.print("[ETH] ✅ IP получен: ");
      Serial.println(ETH.localIP());
      ethConnected = true;
      break;
    case ARDUINO_EVENT_ETH_LOST_IP:
    case ARDUINO_EVENT_ETH_DISCONNECTED:
    case ARDUINO_EVENT_ETH_STOP:
      if (ethConnected) Serial.println("[ETH] Соединение потеряно");
      ethConnected = false;
      break;
    default:
      break;
  }
}

// Пробует поднять Ethernet (модуль W5500); если за ETH_CONNECT_TIMEOUT_MS
// линк/IP не появился (кабель не воткнут или модуль не распаян) — откатывается
// на Wi-Fi, как раньше. Выбор делается один раз при старте.
void connectNetwork() {
  Network.onEvent(onNetworkEvent);

  Serial.println("[NET] Пробую Ethernet (W5500)...");
  SPI.begin(ETH_SPI_SCK, ETH_SPI_MISO, ETH_SPI_MOSI, ETH_PHY_CS);
  ETH.begin(ETH_PHY_TYPE, ETH_PHY_ADDR, ETH_PHY_CS, ETH_PHY_IRQ, ETH_PHY_RST, SPI);

  // ETH.config() обязательно ПОСЛЕ ETH.begin() — до begin() сетевой
  // интерфейс ещё не создан, вызов молча игнорируется (или роняет в
  // панику на некоторых версиях core), и плата тихо продолжает ждать
  // DHCP, как будто config() не было. Статика: прямое соединение
  // ПК↔робот через USB-Ethernet переходник без DHCP-сервера.
  IPAddress staticIP(192, 168, 50, 20);
  IPAddress gateway(192, 168, 50, 10);
  IPAddress subnet(255, 255, 255, 0);
  ETH.config(staticIP, gateway, subnet);

  unsigned long ethWaitStart = millis();
  while (!ethConnected && millis() - ethWaitStart < ETH_CONNECT_TIMEOUT_MS) {
    delay(100);
  }

  if (ethConnected) {
    Serial.println("[NET] ✅ Работаем через Ethernet");
    return;
  }

  Serial.println("[NET] Ethernet не поднялся (нет кабеля/модуля) — переключаюсь на Wi-Fi");
  wifiFallbackActive = true;
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[NET] Connecting to WiFi");
  unsigned long wifiWaitStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiWaitStart < WIFI_FALLBACK_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[NET] ✅ Работаем через Wi-Fi, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[NET] ⚠️ Wi-Fi тоже не подключился за таймаут — "
                    "продолжаю пробовать в фоне (WiFi сам переподключается)");
  }
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n🦉 Soren ESP32-S3 v4.2 Starting...");

  Serial.printf("PSRAM: %d KB total, %d KB free\n",
    ESP.getPsramSize() / 1024, ESP.getFreePsram() / 1024);

  standLed.begin();
  standLedBootAnimation();

  Wire.begin(4, 5);

  Serial.println("[I2C] Скан шины...");
  int i2cFound = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("[I2C]   найдено устройство на 0x%02X%s\n", addr,
                    addr == 0x40 ? "  ← ожидаемый адрес PCA9685" : "");
      i2cFound++;
    }
  }
  if (i2cFound == 0) {
    Serial.println("[I2C] ⚠️ НИЧЕГО не найдено на шине! Проверь SDA/SCL (пины 4/5), "
                    "питание VCC самого PCA9685 (не путать с V+ для серв) и общую землю.");
  }

  pca.begin();

  Wire.end();
  Wire.begin(4, 5);

  pca.setPWMFreq(PCA_FREQ);

  bool attach17 = ledcAttach(SERVO_17_PIN, 50, SERVO_GPIO_LEDC_RES_BITS);
  bool attach18 = ledcAttach(SERVO_18_PIN, 50, SERVO_GPIO_LEDC_RES_BITS);
  if (!attach17) Serial.println("❌ ledcAttach GPIO38 (HEAD_ROLL) FAILED");
  if (!attach18) Serial.println("❌ ledcAttach GPIO39 (BEAK) FAILED");

  for (int i = 0; i < 18; i++) {
    currentServoAngles[i] = 90;
    targetServoAngles[i] = 90;
  }
  for (int i = 0; i < PCA_SERVO_COUNT; i++) setServoAngle(i, 90);
  setServoAngle(HEAD_ROLL_INDEX, 90);
  setServoAngle(BEAK_INDEX, 90);

  initI2SMic();
  initI2SSpeaker();

  if (ESP.getPsramSize() > 0) {
    spkRingBuffer = (uint8_t*)ps_malloc(SPK_BUFFER_SIZE);
    if (spkRingBuffer) {
      Serial.printf("✅ Speaker ring buffer: %d bytes in PSRAM\n", SPK_BUFFER_SIZE);
    } else {
      Serial.println("⚠️ PSRAM alloc failed for speaker buffer");
    }
  }

  connectNetwork();

  webSocket.begin(SERVER_HOST, SERVER_PORT, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);

  initCamera();

  lastHeartbeatRecvMs = millis();

  Serial.println("✅ Setup complete!");
  Serial.println("🔊 Audio mode: ROBOT (ESP32 speaker)");
}

bool isAudioPlaying() {
  return !spkEnded;
}

// ==================== LOOP ====================
void loop() {
  webSocket.loop();

  // Если стартовали в резервном Wi-Fi-режиме, а Ethernet потом всё же
  // появился (воткнули кабель) — просто перезагружаемся: чище, чем на лету
  // тушить Wi-Fi и пересобирать сокеты, а плата и так переживает рестарт
  // без потери состояния серв/подсветки (см. device_state на сервере).
  static unsigned long lastEthRecheckMs = 0;
  if (wifiFallbackActive && ethConnected && millis() - lastEthRecheckMs > 3000) {
    lastEthRecheckMs = millis();
    Serial.println("[NET] Ethernet появился во время работы на Wi-Fi — перезагружаюсь, чтобы переехать на него");
    delay(200);
    ESP.restart();
  }

  if (isConnected) {
    // --- Микрофон: глушим во время TTS + 400 мс после, чтобы не поймать эхо ---
    bool micDeaf = isAudioPlaying() || (millis() - lastSpkActiveMs < MIC_DEAF_MS_AFTER_AUDIO);
    if (micDeaf) {
      // Сбрасываем накопленное и сливаем I2S-буфер, чтобы DMA не переполнился
      micAccumLen = 0;
      micStreamConfirmed = false;
      size_t bytesRead = 0;
      i2s_read(I2S_NUM_0, audioBuffer, sizeof(audioBuffer), &bytesRead, 0);
      if (isAudioPlaying()) {
        lastSpkActiveMs = millis();  // продлеваем глухой период
      }
    } else {
      sendAudioChunk();
    }

    feedSpeakerFromRingBuffer();

    unsigned long now = millis();
    if (isAudioPlaying()) {
      lastVideoFrameMs = now;  // не копим "долг" кадров на время озвучки
    } else if (now - lastVideoFrameMs >= videoFrameIntervalMs) {
      lastVideoFrameMs = now;
      sendVideoFrame();
    }

    interpolateServos();

    if (now - lastHeartbeatSentMs >= HEARTBEAT_INTERVAL_MS) {
      sendPing();
      lastHeartbeatSentMs = now;
    }
    if (now - lastHeartbeatRecvMs >= HEARTBEAT_TIMEOUT_MS) {
      Serial.println("[WS] ❌ Heartbeat timeout — forcing reconnect");
      isConnected = false;
      webSocket.disconnect();
    }
  }

  delay(2);
}

// ==================== WEBSOCKET ====================
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected");
      isConnected = false;
      if (isAudioPlaying()) {
        utteranceInterrupted = true;
      }

      spkWriteIdx = 0;
      spkReadIdx = 0;
      spkDataLen = 0;
      spkPrebuffering = true;
      spkEnded = true;
      if (spkRingBuffer) {
        i2s_zero_dma_buffer(I2S_NUM_1);
      }
      micAccumLen = 0;
      micStreamConfirmed = false;
      break;

    case WStype_CONNECTED:
      Serial.println("[WS] Connected to server");
      isConnected = true;
      lastHeartbeatRecvMs = millis();
      sendPing();
      webSocket.sendTXT("{\"type\":\"audio_mode\"}");

      if (utteranceInterrupted) {
        webSocket.sendTXT("{\"type\":\"audio_resume_request\"}");
        Serial.println("[WS] Запрошен повтор прерванной фразы (audio_resume_request)");
        utteranceInterrupted = false;
      }
      break;

    case WStype_TEXT: {
      String text = String((char*)payload);
      handleServerCommand(text);
      break;
    }

    case WStype_BIN: {
      if (length >= 4) {
        if (memcmp(payload, "AUDI", 4) == 0) {
          size_t audioLen = length - 4;
          Serial.printf("[WS BIN] AUDI received: %u bytes\n", (unsigned)audioLen);
          if (audioMode == "robot") {
            if (spkRingBuffer) {
              pushToSpeakerRing(payload + 4, audioLen);
            } else {
              playAudioDirect(payload + 4, audioLen);
            }
          } else {
            Serial.printf("[Audio] REJECTED — mode=%s\n", audioMode.c_str());
          }
        } else {
          char tag[5] = {0};
          memcpy(tag, payload, 4);
          static unsigned long lastOtherBin = 0;
          if (millis() - lastOtherBin > 3000) {
            Serial.printf("[WS BIN] tag=%s len=%u\n", tag, (unsigned)length);
            lastOtherBin = millis();
          }
        }
      }
      break;
    }

    default:
      break;
  }
}

void handleServerCommand(String& json) {
  StaticJsonDocument<4096> doc;
  DeserializationError error = deserializeJson(doc, json);

  if (error) {
    Serial.print("[WS] JSON parse error: ");
    Serial.println(error.c_str());
    return;
  }

  const char* cmdType = doc["type"];
  if (!cmdType) {
    Serial.println("[WS] WARN: received JSON without 'type' field");
    return;
  }

  if (strcmp(cmdType, "pong") == 0) {
    lastHeartbeatRecvMs = millis();
    Serial.println("[RX pong] heartbeat OK");
    return;
  }

  if (strcmp(cmdType, "servo_update") == 0) {
    JsonObject angles = doc["angles"];
    if (angles.isNull()) {
      Serial.println("[RX servo_update] WARN: empty angles");
      return;
    }
    String logIds = "";
    int updatedCount = 0;
    for (JsonPair kv : angles) {
      int servoId = atoi(kv.key().c_str());
      int angle = kv.value().as<int>();
      if (servoId >= 0 && servoId < 18) {
        targetServoAngles[servoId] = angle;
        updatedCount++;
        logIds += String(servoId) + "=" + String(angle) + " ";
      }
    }
    Serial.printf("[RX servo_update] %d servo(s): %s\n", updatedCount, logIds.c_str());

    bool faceDetected = doc["face_detected"] | false;
    bool dialogActive = doc["dialog_active"] | false;
    lastFaceDetected = faceDetected || dialogActive;
    videoFrameIntervalMs = lastFaceDetected ? (1000 / CAM_FPS_FACE) : (1000 / CAM_FPS_IDLE);
  }
  else if (strcmp(cmdType, "response") == 0) {
    const char* robotText = doc["robot_text"];
    const char* action = doc["action"];
    const char* emotion = doc["emotion"];

    if (robotText) {
      Serial.print("🦉 Soren: ");
      Serial.println(robotText);
    }
    if (emotion) {
      Serial.printf("   Emotion: %s\n", emotion);
    }
    if (action && strlen(action) > 0) {
      Serial.printf("🎬 Action: %s\n", action);
    }

    JsonArray servoAngles = doc["servo_angles"];
    if (!servoAngles.isNull()) {
      int i = 0;
      String logAngles = "";
      for (JsonVariant v : servoAngles) {
        if (i >= 18) break;
        int a = v.as<int>();
        targetServoAngles[i] = a;
        logAngles += String(a) + " ";
        i++;
      }
      Serial.printf("[RX response] servo_angles[%d]: %s\n", i, logAngles.c_str());
    }
  }
  else if (strcmp(cmdType, "audio_mode") == 0) {
    const char* mode = doc["output_mode"];
    if (mode) {
      audioMode = String(mode);
      Serial.printf("🔊 Audio mode changed to: %s\n", audioMode.c_str());
    }
  }
  else if (strcmp(cmdType, "backlight") == 0) {
    bool enabled = doc["enabled"] | false;
    setStandBacklight(enabled);
  }
  else if (strcmp(cmdType, "request_state") == 0) {
    // Сервер только что (пере)подключил нас и просит реальное состояние —
    // серв и подсветки — чтобы не работать по сбитым/дефолтным данным после
    // своего рестарта. Отвечаем тем, что физически выставлено ПРЯМО СЕЙЧАС.
    sendDeviceState();
  }
  else {
    Serial.printf("[WS] Unhandled type: %s\n", cmdType);
  }
}

void sendDeviceState() {
  StaticJsonDocument<640> doc;
  doc["type"] = "device_state";
  JsonObject angles = doc.createNestedObject("angles");
  for (int i = 0; i < 18; i++) {
    angles[String(i)] = currentServoAngles[i];
  }
  doc["backlight"] = standBacklightOn;
  String payload;
  serializeJson(doc, payload);
  webSocket.sendTXT(payload);
  Serial.println("[TX device_state] реальное состояние (сервы + подсветка) отправлено на сервер");
}

// ==================== ПОДСВЕТКА ====================
void standLedBootAnimation() {
  standLed.clear();
  standLed.setBrightness(255);
  standLed.show();

  for (int i = 0; i < STAND_LED_COUNT; i++) {
    standLed.setPixelColor(i, standLed.Color(WARM_WHITE_R, WARM_WHITE_G, WARM_WHITE_B));
    standLed.show();
    delay(1200 / STAND_LED_COUNT);
  }

  delay(400);

  for (int b = 255; b >= 0; b -= 15) {
    standLed.setBrightness(b);
    standLed.show();
    delay(20);
  }
  standLed.setBrightness(255);
  standLed.clear();
  standLed.show();
}

void setStandBacklight(bool on) {
  if (on == standBacklightOn) return;
  standBacklightOn = on;

  if (on) {
    for (int i = 0; i < STAND_LED_COUNT; i++) {
      standLed.setPixelColor(i, standLed.Color(WARM_WHITE_R, WARM_WHITE_G, WARM_WHITE_B));
    }
  } else {
    standLed.clear();
  }
  standLed.show();

  if (!on) {
    delay(2);
    standLed.clear();
    standLed.show();
  }

  Serial.printf("💡 Подсветка: %s\n", on ? "ВКЛ" : "выкл");
}

// ==================== СЕРВОПРИВОДЫ ====================
// Duty для прямого GPIO-серво (500-2500мкс импульс @ 50Гц = 20000мкс период),
// пересчитано под РЕАЛЬНОЕ разрешение SERVO_GPIO_LEDC_RES_BITS (14 бит на S3),
// а не захардкоженные под несуществующие 16 бит числа, как раньше.
int gpioServoDuty(int angle) {
  const long minDuty = 500L  * (SERVO_GPIO_LEDC_MAX_DUTY + 1) / 20000L;  // ~410
  const long maxDuty = 2500L * (SERVO_GPIO_LEDC_MAX_DUTY + 1) / 20000L;  // ~2048
  return map(angle, 0, 180, (int)minDuty, (int)maxDuty);
}

void setServoAngle(int servoId, int angle) {
  angle = constrain(angle, 0, 180);

  if (servoId < PCA_SERVO_COUNT) {
    int pulse = map(angle, 0, 180, 150, 600);
    pca.setPWM(servoId, 0, pulse);
  } else if (servoId == HEAD_ROLL_INDEX) {
    ledcWrite(SERVO_17_PIN, gpioServoDuty(angle));  // GPIO 38, не "GPIO16"
  } else if (servoId == BEAK_INDEX) {
    ledcWrite(SERVO_18_PIN, gpioServoDuty(angle));  // GPIO 39, не "GPIO17"
  }

  currentServoAngles[servoId] = angle;
}

void interpolateServos() {
  if (millis() - lastServoUpdateMs < SERVO_UPDATE_MS) return;
  lastServoUpdateMs = millis();

  bool anyMoving = false;
  for (int i = 0; i < 18; i++) {
    int diff = targetServoAngles[i] - currentServoAngles[i];
    if (abs(diff) <= SERVO_DEAD_ZONE) continue;

    // Eased-шаг: доля оставшегося расстояния, а не фиксированный шаг.
    // Работает одинаково для мелких поправок слежения и для больших скачков цели
    // (обрыв связи и реконнект, смена цели/лица, любой резкий "response" с сервера) —
    // сервопривод никогда не дёргается на всю разницу за один тик: сначала едет с
    // потолком скорости SERVO_MAX_STEP, затем плавно тормозит по мере приближения к цели.
    float rawStep = diff * SERVO_EASE_FACTOR;
    int step = (int)(rawStep + (rawStep > 0 ? 0.5f : -0.5f));  // округление к ближайшему, без доп. include
    if (step == 0) step = (diff > 0) ? 1 : -1;  // гарантируем прогресс к цели даже на "хвосте"
    step = constrain(step, -SERVO_MAX_STEP, SERVO_MAX_STEP);

    setServoAngle(i, currentServoAngles[i] + step);
    anyMoving = true;
  }

  static unsigned long lastServoLog = 0;
  if (anyMoving && (millis() - lastServoLog > 300)) {
    Serial.print("[Servo] moving:");
    for (int i = 0; i < 18; i++) {
      if (abs(targetServoAngles[i] - currentServoAngles[i]) > SERVO_DEAD_ZONE) {
        Serial.printf(" %d=%d→%d", i, currentServoAngles[i], targetServoAngles[i]);
      }
    }
    Serial.println();
    lastServoLog = millis();
  }
}

// ==================== АУДИО (I2S) ====================
void initI2SMic() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 512,
    .use_apll = true
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_MIC_BCLK,
    .ws_io_num = I2S_MIC_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_MIC_DIN
  };

  esp_err_t err1 = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  esp_err_t err2 = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err1 != ESP_OK || err2 != ESP_OK) {
    Serial.printf("❌ Микрофон init failed: install=0x%x, set_pin=0x%x\n", err1, err2);
  } else {
    Serial.println("✅ Микрофон (INMP441) инициализирован [16-bit I2S]");
  }
}

void initI2SSpeaker() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SPEAKER_SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,      // было 4 — больше буферизация, меньше рывков
    .dma_buf_len = 512,      // было 256
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_AMP_BCLK,
    .ws_io_num = I2S_AMP_LRC,
    .data_out_num = I2S_AMP_DOUT,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  esp_err_t err1 = i2s_driver_install(I2S_NUM_1, &i2s_config, 0, NULL);
  esp_err_t err2 = i2s_set_pin(I2S_NUM_1, &pin_config);
  if (err1 != ESP_OK || err2 != ESP_OK) {
    Serial.printf("❌ Динамик init failed: install=0x%x, set_pin=0x%x\n", err1, err2);
  } else {
    Serial.println("✅ Динамик (MAX98357A) инициализирован");
    int16_t silence[AUDIO_BUFFER_SIZE] = {0};
    size_t written = 0;
    i2s_write(I2S_NUM_1, silence, sizeof(silence), &written, portMAX_DELAY);
  }
}

void sendPing() {
  StaticJsonDocument<128> doc;
  doc["type"] = "ping";
  if (strlen(DEVICE_KEY) > 0) {
    doc["device_key"] = DEVICE_KEY;
  }
  String payload;
  serializeJson(doc, payload);
  webSocket.sendTXT(payload);
}

// ==================== МИКРОФОН + NOISE GATE ====================
void sendAudioChunk() {
  size_t bytesRead = 0;
  esp_err_t result = i2s_read(I2S_NUM_0, audioBuffer, sizeof(audioBuffer), &bytesRead, 0);

  static unsigned long lastMicErrLog = 0;
  if (result != ESP_OK) {
    if (millis() - lastMicErrLog > 5000) {
      Serial.printf("❌ i2s_read(mic) error: 0x%x\n", result);
      lastMicErrLog = millis();
    }
    return;
  }
  if (bytesRead == 0) return;

  size_t sampleCount = bytesRead / 2;

  int16_t maxAmp = 0;
  int16_t minAmp = 0;
  for (size_t i = 0; i < sampleCount; i++) {
    int16_t v = audioBuffer[i];
    if (v > maxAmp) maxAmp = v;
    if (v < minAmp) minAmp = v;
  }

  if (maxAmp < MIC_NOISE_GATE && abs(minAmp) < MIC_NOISE_GATE) {
    if (micAccumLen > 0 && (millis() - silenceStartMs > MIC_SILENCE_TIMEOUT)) {
      Serial.printf("[Mic] silence timeout, dropping %u bytes\n", (unsigned)micAccumLen);
      micAccumLen = 0;
    }
    if (micAccumLen == 0) {
      silenceStartMs = millis();
    }
    return;
  }

  silenceStartMs = millis();

  if (!micStreamConfirmed) {
    Serial.printf("🎤 Микрофон стримит: bytes=%u max=%d min=%d\n",
      (unsigned)bytesRead, maxAmp, minAmp);
    micStreamConfirmed = true;
  }

  static unsigned long lastAmpLog = 0;
  if (millis() - lastAmpLog > 500) {
    Serial.printf("[Mic] bytes=%u max=%5d min=%5d accum=%u\n",
      (unsigned)bytesRead, maxAmp, minAmp, (unsigned)micAccumLen);
    lastAmpLog = millis();
  }

  if (micAccumLen + bytesRead > sizeof(micAccum)) {
    Serial.printf("⚠️ micAccum overflow (%u+%u), reset\n",
      (unsigned)micAccumLen, (unsigned)bytesRead);
    micAccumLen = 0;
  }
  memcpy(micAccum + micAccumLen, audioBuffer, bytesRead);
  micAccumLen += bytesRead;

  while (micAccumLen >= AUDIO_CHUNK_BYTES) {
    uint8_t packet[4 + AUDIO_CHUNK_BYTES];
    memcpy(packet, "AUDI", 4);
    memcpy(packet + 4, micAccum, AUDIO_CHUNK_BYTES);
    bool ok = webSocket.sendBIN(packet, sizeof(packet));
    Serial.printf("[AUDI TX] %s accumBefore=%u\n",
      ok ? "OK" : "FAIL", (unsigned)micAccumLen);

    memmove(micAccum, micAccum + AUDIO_CHUNK_BYTES, micAccumLen - AUDIO_CHUNK_BYTES);
    micAccumLen -= AUDIO_CHUNK_BYTES;
  }
}

// ==================== ДИНАМИК (ring buffer) ====================
void pushToSpeakerRing(uint8_t* data, size_t len) {
  if (!spkRingBuffer || len == 0) return;

  lastSpkPushMs = millis();

  if (spkEnded) {
    // Началась новая фраза: плавный fade-in первых сэмплов и новый цикл
    // джиттер-буфера (см. feedSpeakerFromRingBuffer).
    spkEnded = false;
    spkPrebuffering = true;
    size_t fadeSamples = min(len / 2, (size_t)240);
    int16_t* samples = (int16_t*)data;
    for (size_t i = 0; i < fadeSamples; i++) {
      samples[i] = (int16_t)(samples[i] * ((float)i / fadeSamples));
    }
  }

  size_t space = SPK_BUFFER_SIZE - spkDataLen;
  if (len > space) {
    Serial.printf("[Audio] ⚠️ ring overflow: need %u, have %u — dropping %u bytes\n",
                  (unsigned)len, (unsigned)space, (unsigned)(len - space));
    len = space;
    if (len == 0) return;
  }

  for (size_t i = 0; i < len; i++) {
    spkRingBuffer[spkWriteIdx] = data[i];
    spkWriteIdx = (spkWriteIdx + 1) % SPK_BUFFER_SIZE;
  }
  spkDataLen += len;
}

void feedSpeakerFromRingBuffer() {
  if (!spkRingBuffer) return;

  if (spkPrebuffering) {
    if (spkDataLen < SPK_PREBUFFER_BYTES &&
        spkDataLen > 0 && millis() - lastSpkPushMs < SPK_PREBUFFER_STALL_MS) {
      return;
    }
    if (spkDataLen == 0) return;
    spkPrebuffering = false;
  }

  if (spkDataLen == 0) {
    if (!spkEnded && millis() - lastSpkPushMs > SPK_END_SILENCE_MS) {
      i2s_zero_dma_buffer(I2S_NUM_1);
      spkEnded = true;
    }
    return;
  }

  const size_t CHUNK = 2048;
  uint8_t temp[CHUNK];
  size_t toWrite = min((size_t)spkDataLen, CHUNK);

  for (size_t i = 0; i < toWrite; i++) {
    temp[i] = spkRingBuffer[spkReadIdx];
    spkReadIdx = (spkReadIdx + 1) % SPK_BUFFER_SIZE;
  }

  size_t bytesWritten = 0;
  esp_err_t err = i2s_write(I2S_NUM_1, temp, toWrite, &bytesWritten, pdMS_TO_TICKS(100));

  if (err == ESP_OK && bytesWritten > 0) {
    spkDataLen -= bytesWritten;
    size_t unwritten = toWrite - bytesWritten;
    if (unwritten > 0) {
      // Откатываем readIdx только на незаписанный хвост
      if (spkReadIdx >= unwritten) {
        spkReadIdx -= unwritten;
      } else {
        spkReadIdx = SPK_BUFFER_SIZE - (unwritten - spkReadIdx);
      }
    }
  } else {
    // Полный провал — откатываем всё
    if (spkReadIdx >= toWrite) {
      spkReadIdx -= toWrite;
    } else {
      spkReadIdx = SPK_BUFFER_SIZE - (toWrite - spkReadIdx);
    }
    static unsigned long lastSpkErr = 0;
    if (millis() - lastSpkErr > 2000) {
      Serial.printf("[Audio] I2S write err=%d, rollback %u bytes\n", err, (unsigned)toWrite);
      lastSpkErr = millis();
    }
  }
}

void playAudioDirect(uint8_t* data, size_t len) {
  Serial.printf("[Audio] direct play: %u bytes\n", (unsigned)len);
  size_t bytesWritten = 0;
  esp_err_t err = i2s_write(I2S_NUM_1, data, len, &bytesWritten, pdMS_TO_TICKS(200));
  if (err != ESP_OK) {
    Serial.printf("[Audio] ❌ I2S write error: %d (written=%u)\n", err, (unsigned)bytesWritten);
  } else {
    Serial.printf("[Audio] ✅ Played %u/%u bytes\n", (unsigned)bytesWritten, (unsigned)len);
  }
}

// ==================== КАМЕРА ====================
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 11;
  config.pin_d1 = 9;
  config.pin_d2 = 8;
  config.pin_d3 = 10;
  config.pin_d4 = 12;
  config.pin_d5 = 18;
  config.pin_d6 = 17;
  config.pin_d7 = 16;
  config.pin_xclk = -1;
  config.pin_pclk = 13;
  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.sccb_i2c_port = 1;
  config.pin_pwdn = -1;
  config.pin_reset = 15;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = DETECT_FRAME_SIZE;
  config.jpeg_quality = 30;
  config.fb_count = 2;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
  } else {
    Serial.println("✅ Camera initialized (QVGA adaptive FPS)");
  }
}

void sendVideoFrame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[VIDE] esp_camera_fb_get() returned NULL");
    return;
  }

  if (fb->len > sizeof(videPacket) - 4) {
    Serial.printf("[VIDE] Frame too big: %u bytes, skipping\n", (unsigned)fb->len);
    esp_camera_fb_return(fb);
    return;
  }

  memcpy(videPacket, "VIDE", 4);
  memcpy(videPacket + 4, fb->buf, fb->len);
  bool ok = webSocket.sendBIN(videPacket, 4 + fb->len);

  static unsigned long lastVideLog = 0;
  if (ok && (millis() - lastVideLog > 3000)) {
    Serial.printf("[VIDE] Sent %u bytes (adaptive FPS=%lu)\n",
      (unsigned)fb->len, 1000UL / videoFrameIntervalMs);
    lastVideLog = millis();
  }

  esp_camera_fb_return(fb);
}