/*
 * 🦉 Robot ESP32-S3 Firmware - Soren v2.1
 * + LED-глаза (WS2812)
 * + Переключение аудио режима (робот/локально)
 */

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <driver/i2s.h>
#include <esp_camera.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>

// ==================== КОНФИГУРАЦИЯ ====================
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_HOST = "192.168.1.100";  // IP ноутбука

// Общий секрет, подтверждающий серверу, что это доверенное устройство
// (см. DEVICE_KEY в .env сервера — оба значения ДОЛЖНЫ совпадать).
// Пусто ("") — сервер не проверяет ключ (совместимость со старым поведением,
// НО тогда любой WS-клиент сможет притвориться устройством, если на сервере
// включена авторизация панели PANEL_PASSWORD — рекомендуется задать оба).
const char* DEVICE_KEY = "";
const int SERVER_PORT = 8765;

// PCA9685
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);
const int PCA_FREQ = 50;

// Доп. серво на GPIO
const int SERVO_17_PIN = 17;
const int SERVO_18_PIN = 18;

// LED-глаза (WS2812)
const int LED_PIN = 8;
const int LED_COUNT = 2;
Adafruit_NeoPixel eyes(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// I2S
#define I2S_BCLK    4
#define I2S_WS      5
#define I2S_DIN     6   // INMP441
#define I2S_DOUT    7   // MAX98357A
#define SAMPLE_RATE 16000

// Камера OV2640
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    15
#define SIOD_GPIO_NUM    4
#define SIOC_GPIO_NUM    5

// Два потока видео: маленький частый (детекция/трекинг лица на сервере) и
// крупный редкий (только чтобы человеку было приятно смотреть в панели
// мониторинга). Экономит Wi-Fi и время декодирования на сервере — детектору
// лиц всё равно достаточно небольшой картинки.
#define DETECT_FRAME_SIZE      FRAMESIZE_QQVGA   // 160x120 — часто, каждый цикл (тег VIDE)
#define PANEL_FRAME_SIZE       FRAMESIZE_QVGA    // 320x240 — редко, для панели (тег VIDP)
#define PANEL_FRAME_INTERVAL_MS 700
unsigned long lastPanelFrameMs = 0;

// ==================== ГЛОБАЛЬНЫЕ ====================
WebSocketsClient webSocket;
bool isConnected = false;

int currentServoAngles[18] = {90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90};
int targetServoAngles[18] = {90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90,90};

String currentEyeLed = "soft_white_low";
String audioMode = "robot";  // "robot" или "local"

#define AUDIO_BUFFER_SIZE 1024
int16_t audioBuffer[AUDIO_BUFFER_SIZE];

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n🦉 Soren ESP32-S3 v2.1 Starting...");

  // LED-глаза
  eyes.begin();
  eyes.setBrightness(50);
  setEyeColor("soft_white_low");

  // PCA9685
  Wire.begin(21, 22);
  pca.begin();
  pca.setPWMFreq(PCA_FREQ);

  // Доп. серво
  ledcSetup(0, 50, 16);
  ledcSetup(1, 50, 16);
  ledcAttachPin(SERVO_17_PIN, 0);
  ledcAttachPin(SERVO_18_PIN, 1);

  // Сброс серво
  for (int i = 0; i < 16; i++) setServoAngle(i, 90);
  setServoAngle(16, 90);
  setServoAngle(17, 90);

  // I2S микрофон
  initI2SMic();

  // WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! IP: ");
  Serial.println(WiFi.localIP());

  // WebSocket
  webSocket.begin(SERVER_HOST, SERVER_PORT, "/ws");
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);

  // Камера
  initCamera();

  Serial.println("✅ Setup complete!");
  Serial.println("🔊 Audio mode: ROBOT (ESP32 speaker)");
}

// ==================== LOOP ====================
void loop() {
  webSocket.loop();

  if (isConnected) {
    sendAudioChunk();
    sendVideoFrame();

    unsigned long now = millis();
    if (now - lastPanelFrameMs >= PANEL_FRAME_INTERVAL_MS) {
      lastPanelFrameMs = now;
      sendPanelFrame();
    }

    interpolateServos();
  }

  delay(10);
}

// ==================== WEBSOCKET ====================
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Disconnected");
      isConnected = false;
      break;

    case WStype_CONNECTED:
      Serial.println("[WS] Connected to server");
      isConnected = true;
      sendPing();
      // Запрашиваем текущий аудио режим
      webSocket.sendTXT("{\"type\":\"audio_mode\"}");
      break;

    case WStype_TEXT: {
      String text = String((char*)payload);
      handleServerCommand(text);
      break;
    }

    case WStype_BIN: {
      if (length > 4 && memcmp(payload, "AUDI", 4) == 0) {
        // Только если режим "robot" — проигрываем аудио на динамике ESP32
        if (audioMode == "robot") {
          playAudio(payload + 4, length - 4);
        } else {
          Serial.println("[Audio] Received but ignored (LOCAL mode)");
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

  if (strcmp(cmdType, "servo_update") == 0) {
    // ВАЖНО: сервер теперь шлёт только ИЗМЕНИВШИЕСЯ углы (объект {"id": angle, ...}),
    // а не полный массив из 18 значений на каждый кадр — экономит трафик.
    // Требует новую версию сервера (websocket_server.py); со старым сервером,
    // присылающим массив, это поле будет пустым объектом и ничего не обновится.
    JsonObject angles = doc["angles"];
    for (JsonPair kv : angles) {
      int servoId = atoi(kv.key().c_str());
      if (servoId >= 0 && servoId < 18) {
        targetServoAngles[servoId] = kv.value().as<int>();
      }
    }
  }
  else if (strcmp(cmdType, "response") == 0) {
    const char* robotText = doc["robot_text"];
    const char* action = doc["action"];
    const char* emotion = doc["emotion"];
    const char* eyeLed = doc["eye_led"];

    Serial.print("🦉 Soren: ");
    Serial.println(robotText);

    if (emotion) {
      Serial.print("   Emotion: ");
      Serial.println(emotion);
    }

    if (eyeLed) {
      setEyeColor(eyeLed);
    }

    if (action && strlen(action) > 0) {
      Serial.print("🎬 Action: ");
      Serial.println(action);
    }
  }
  else if (strcmp(cmdType, "audio_mode") == 0) {
    const char* mode = doc["mode"];
    if (mode) {
      audioMode = String(mode);
      Serial.print("🔊 Audio mode changed to: ");
      Serial.println(audioMode);
      if (audioMode == "local") {
        Serial.println("   → Audio will play on PC headphones");
      } else {
        Serial.println("   → Audio will play on ESP32 speaker");
      }
    }
  }
}

// ==================== LED ГЛАЗА ====================
void setEyeColor(const char* mode) {
  if (strcmp(mode, "soft_white_low") == 0) {
    eyes.setPixelColor(0, eyes.Color(50, 50, 60));
    eyes.setPixelColor(1, eyes.Color(50, 50, 60));
  }
  else if (strcmp(mode, "dim_blue_pulse") == 0) {
    eyes.setPixelColor(0, eyes.Color(20, 30, 80));
    eyes.setPixelColor(1, eyes.Color(20, 30, 80));
  }
  else if (strcmp(mode, "bright_orange_flicker") == 0) {
    eyes.setPixelColor(0, eyes.Color(255, 100, 0));
    eyes.setPixelColor(1, eyes.Color(255, 100, 0));
  }
  else if (strcmp(mode, "warm_yellow_glow") == 0) {
    eyes.setPixelColor(0, eyes.Color(255, 200, 50));
    eyes.setPixelColor(1, eyes.Color(255, 200, 50));
  }
  else if (strcmp(mode, "steady_white_bright") == 0) {
    eyes.setPixelColor(0, eyes.Color(255, 255, 255));
    eyes.setPixelColor(1, eyes.Color(255, 255, 255));
  }
  else if (strcmp(mode, "bright_white_flash") == 0) {
    eyes.setPixelColor(0, eyes.Color(255, 255, 255));
    eyes.setPixelColor(1, eyes.Color(255, 255, 255));
    eyes.setBrightness(255);
    eyes.show();
    delay(100);
    eyes.setBrightness(50);
    return;
  }
  else if (strcmp(mode, "dim_amber_slow") == 0) {
    eyes.setPixelColor(0, eyes.Color(80, 50, 10));
    eyes.setPixelColor(1, eyes.Color(80, 50, 10));
  }
  else {
    eyes.setPixelColor(0, eyes.Color(50, 50, 60));
    eyes.setPixelColor(1, eyes.Color(50, 50, 60));
  }
  eyes.show();
}

// ==================== СЕРВОПРИВОДЫ ====================
void setServoAngle(int servoId, int angle) {
  angle = constrain(angle, 0, 180);

  if (servoId < 16) {
    int pulse = map(angle, 0, 180, 150, 600);
    pca.setPWM(servoId, 0, pulse);
  } else if (servoId == 16) {
    ledcWrite(0, map(angle, 0, 180, 1638, 8192));
  } else if (servoId == 17) {
    ledcWrite(1, map(angle, 0, 180, 1638, 8192));
  }

  currentServoAngles[servoId] = angle;
}

void interpolateServos() {
  for (int i = 0; i < 18; i++) {
    int diff = targetServoAngles[i] - currentServoAngles[i];
    if (abs(diff) > 1) {
      int step = diff / 4;
      if (step == 0) step = (diff > 0) ? 1 : -1;
      setServoAngle(i, currentServoAngles[i] + step);
    }
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
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_DIN
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

// Шлёт ping с device_key (если задан) — сервер использует это как подтверждение,
// что соединение установлено доверенным устройством, а не случайным WS-клиентом
// (см. modules/auth.py::is_valid_device_ping на стороне сервера).
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

void sendAudioChunk() {
  size_t bytesRead = 0;
  esp_err_t result = i2s_read(I2S_NUM_0, audioBuffer, sizeof(audioBuffer), &bytesRead, 0);

  if (result == ESP_OK && bytesRead > 0) {
    uint8_t packet[4 + bytesRead];
    memcpy(packet, "AUDI", 4);
    memcpy(packet + 4, audioBuffer, bytesRead);
    webSocket.sendBIN(packet, 4 + bytesRead);
  }
}

void playAudio(uint8_t* data, size_t len) {
  Serial.printf("[Audio] Playing %d bytes on ESP32 speaker\n", len);
  // TODO: I2S write to MAX98357A
}

// ==================== КАМЕРА ====================
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 11;
  config.pin_d1 = 9;
  config.pin_d2 = 10;
  config.pin_d3 = 12;
  config.pin_d4 = 18;
  config.pin_d5 = 17;
  config.pin_d6 = 16;
  config.pin_d7 = 15;
  config.pin_xclk = 8;
  config.pin_pclk = 13;
  config.pin_vsync = 6;
  config.pin_href = 7;
  config.pin_sscb_sda = 4;
  config.pin_sscb_scl = 5;
  config.pin_pwdn = -1;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = DETECT_FRAME_SIZE;  // маленькое разрешение по умолчанию — для VIDE (детекция)
  config.jpeg_quality = 15;
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
  } else {
    Serial.println("✅ Camera initialized");
  }
}

void sendVideoFrame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  uint8_t* packet = (uint8_t*)malloc(4 + fb->len);
  if (packet) {
    memcpy(packet, "VIDE", 4);
    memcpy(packet + 4, fb->buf, fb->len);
    webSocket.sendBIN(packet, 4 + fb->len);
    free(packet);
  }

  esp_camera_fb_return(fb);
}

// Крупный кадр повышенного качества — только для панели мониторинга (тег VIDP),
// шлётся редко (см. PANEL_FRAME_INTERVAL_MS). Детекцию на сервере НЕ запускает.
// ПРИМЕЧАНИЕ: смена разрешения "на лету" — стандартный для esp32-camera приём,
// но сразу после s->set_framesize() первый кадр может быть смазан/повреждён,
// пока сенсор перестраивается — поэтому один кадр после каждого переключения
// намеренно отбрасывается ("settle"-кадр).
void sendPanelFrame() {
  sensor_t * s = esp_camera_sensor_get();
  if (!s) return;

  s->set_framesize(s, PANEL_FRAME_SIZE);

  camera_fb_t *settle = esp_camera_fb_get();
  if (settle) esp_camera_fb_return(settle);

  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) {
    uint8_t* packet = (uint8_t*)malloc(4 + fb->len);
    if (packet) {
      memcpy(packet, "VIDP", 4);
      memcpy(packet + 4, fb->buf, fb->len);
      webSocket.sendBIN(packet, 4 + fb->len);
      free(packet);
    }
    esp_camera_fb_return(fb);
  }

  // Возвращаемся к маленькому разрешению для детекции (VIDE)
  s->set_framesize(s, DETECT_FRAME_SIZE);
  camera_fb_t *settle2 = esp_camera_fb_get();
  if (settle2) esp_camera_fb_return(settle2);
}