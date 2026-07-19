// =======================================================
//  OWL TRANSMITTER — ФИНАЛЬНАЯ ВЕРСИЯ ПОД ТВОЁ ТЗ (2025)
// =======================================================
#include <SPI.h>
#include <RF24.h>
#include <Encoder.h>

RF24 radio(9, 10);                             // CE=9, CSN=10
const uint64_t pipe = 0xB00B1E50LL;             // тот же адрес, что в теле

// === ДЖОЙСТИКИ KY-023 ===
#define JOY1_X    A0
#define JOY1_Y    A1
#define JOY1_SW   2      // кнопка первого джойстика
#define JOY2_X    A3
#define JOY2_Y    A4
#define JOY2_SW   3      // кнопка второго джойстика

// === КНОПКИ РЕЖИМОВ ===
#define BTN_WALK  4
#define BTN_FLY   5
#define BTN_FREE  6     // третья кнопка — разблокировка FREE-режима

// === СВЕТОДИОДЫ ===
#define LED_WALK  7      // зелёный
#define LED_FLY   8      // зелёный
#define LED_FREE  12     // зелёный (или любой свободный пин)
#define LED_LOCK  1      // красный — горит, пока FREE заблокирован (D1 на Nano)

// === ЭНКОДЕРЫ KY-040 ===
Encoder encWing(14, 15);   // A0 и A1 уже заняты джойстиками → используем A8/A9 как цифровые
Encoder encFold(A2, A6);   // любые свободные пины, я взял A2 и A6 (или 16/17)

// Если A8/A9 не подходят — просто поменяй на любые свободные цифровые пины

struct Packet {
  int16_t j1x, j1y, j2x, j2y;
  bool    j1btn, j2btn;
  uint8_t mode;        : 3;   // 0=WALK, 1=FLY, 2=FREE
  uint8_t padding        : 5;
  int8_t  enc_wing;             // изменение первого энкодера
  int8_t  enc_fold;             // изменение второго
} pkt;

uint8_t currentMode = 0;        // 0=WALK (по умолчанию)
bool lastWalk = true, lastFly = true, lastFreeBtn = true;

long oldWing = 0, oldFold = 0;

void setup() {
  // кнопки с подтяжкой
  pinMode(JOY1_SW, INPUT_PULLUP);
  pinMode(JOY2_SW, INPUT_PULLUP);
  pinMode(BTN_WALK,  INPUT_PULLUP);
  pinMode(BTN_FLY,   INPUT_PULLUP);
  pinMode(BTN_FREE,  INPUT_PULLUP);

  // светодиоды
  pinMode(LED_WALK, OUTPUT);  digitalWrite(LED_WALK, HIGH);
  pinMode(LED_FLY,  OUTPUT);
  pinMode(LED_FREE, OUTPUT);
  pinMode(LED_LOCK, OUTPUT);  digitalWrite(LED_LOCK, HIGH); // красный горит — FREE заблокирован

  radio.begin();
  radio.setPALevel(RF24_PA_MAX);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(90);
  radio.setCRCLength(RF24_CRC_16);
  radio.setRetries(15, 15);
  radio.openWritingPipe(pipe);
  radio.stopListening();

  delay(100);
}

void loop() {
  // === Читаем джойстики ===
  pkt.j1x  = analogRead(JOY1_X) - 512;
  pkt.j1y  = analogRead(JOY1_Y) - 512;
  pkt.j2x  = analogRead(JOY2_X) - 512;
  pkt.j2y  = analogRead(JOY2_Y) - 512;
  pkt.j1btn = !digitalRead(JOY1_SW);
  pkt.j2btn = !digitalRead(JOY2_SW);

  // === Переключение режимов по отпусканию кнопки ===
  bool walkBtn  = !digitalRead(BTN_WALK);
  bool flyBtn   = !digitalRead(BTN_FLY);
  bool freeBtn  = !digitalRead(BTN_FREE);  // третья кнопка — разблокировка FREE

  if (walkBtn && !lastWalk)  currentMode = 0;   // отпустили → WALK
  if (flyBtn  && !lastFly)   currentMode = 1;   // отпустили → FLY
  if (freeBtn && !lastFreeBtn) {
    if (currentMode != 2) currentMode = 2;       // переходим в FREE только по нажатию
    else currentMode = 0;                        // повторное нажатие — выход из FREE в WALK
  }

  lastWalk = walkBtn;
  lastFly  = flyBtn;
  lastFreeBtn = freeBtn;

  pkt.mode = currentMode;

  // === Светодиоды ===
  digitalWrite(LED_WALK, currentMode == 0 ? HIGH : LOW);
  digitalWrite(LED_FLY,  currentMode == 1 ? HIGH : LOW);
  digitalWrite(LED_FREE, currentMode == 2 ? HIGH : LOW);
  digitalWrite(LED_LOCK, currentMode == 2 ? LOW : HIGH); // красный гаснет только в FREE

  // === Энкодеры только в режиме FREE ===
  pkt.enc_wing = 0;
  pkt.enc_fold = 0;
  if (currentMode == 2) {
    long wing = encWing.read();
    long fold = encFold.read();

    if (wing != oldWing) {
      pkt.enc_wing = (wing > oldWing) ? 1 : -1;
      oldWing = wing;
    }
    if (fold != oldFold) {
      pkt.enc_fold = (fold > oldFold) ? 1 : -1;
      oldFold = fold;
    }
  }

  // === Отправка пакета ===
  radio.write(&pkt, sizeof(pkt));

  // индикация успешной отправки (встроенный LED)
  digitalWrite(LED_BUILTIN, radio.write(&pkt, sizeof(pkt)) ? HIGH : LOW);
  delay(2);
  digitalWrite(LED_BUILTIN, LOW);

  delay(18);   // ≈50–55 пакетов в секунду — идеально
}