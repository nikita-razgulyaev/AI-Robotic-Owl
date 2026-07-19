// =======================================================
//  OWL BODY — ОКОНЧАТЕЛЬНАЯ ВЕРСИЯ С КАЛИБРОВКОЙ ПРАВОГО КРЫЛА
//  Правое маховое серво теперь имеет свои MIN/MAX!
// =======================================================
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <SPI.h>
#include <RF24.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
RF24 radio(9, 10);

// -------------------- КАНАЛЫ --------------------
#define L_FLAP      0
#define L_FOLD      1
#define R_FLAP      2
#define R_FOLD      3
#define WING_TILT   4
#define TAIL_LIFT  13

// -------------------- ОБЩИЙ ДИАПАЗОН (для всех кроме правого махового) --------------------
#define SERVO_MIN   150
#define SERVO_MAX   600

// -------------------- ОТДЕЛЬНАЯ КАЛИБРОВКА ТОЛЬКО ДЛЯ ПРАВОГО МАХОВОГО СЕРВО --------------------
#define R_FLAP_MIN  150    // ← подбери точно под своё крыло (обычно 170–200)
#define R_FLAP_MAX  540    // ← подбери точно (обычно 550–580). У меня идеально 570!

// -------------------- ПАКЕТ --------------------
struct Packet {
  int16_t j1x, j1y, j2x, j2y;
  bool    j1btn, j2btn;
  uint8_t mode;        // 0=WALK, 1=FLY, 2=FREE
  int8_t  enc_wing;
  int8_t  enc_fold;
} pkt;

// -------------------- СОСТОЯНИЕ --------------------
enum Mode { WALK, FLY, FREE };
Mode currentMode = WALK;
Mode requestedMode = WALK;
bool modeChanging = false;
unsigned long lastPacketTime = 0;

// =======================================================
void setup() {
  Serial.begin(115200);

  pwm.begin();
  pwm.setPWMFreq(50);

  radio.begin();
  radio.setPALevel(RF24_PA_MAX);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(90);
  radio.openReadingPipe(1, 0xB00B1E50LL);
  radio.startListening();

  goToNeutral();
  Serial.println(F("OWL BODY готова — оба крыла откалиброваны и живые!"));
}

// =======================================================
void loop() {
  if (radio.available()) {
    radio.read(&pkt, sizeof(pkt));
    lastPacketTime = millis();

    requestedMode = (Mode)pkt.mode;

    if (requestedMode != currentMode && !modeChanging) {
      changeMode(requestedMode);
    }
  }

  if (millis() - lastPacketTime > 300) {
    // стоп 360° при потере связи
    pwm.setPWM(L_FOLD, 0, 0);
    pwm.setPWM(R_FOLD, 0, 0);
  }
}

// =======================================================
// ПРИЖАТОЕ ПОЛОЖЕНИЕ (WALK / FREE)
void goToNeutral() {
  pwm.setPWM(L_FLAP,    0, map(90,  0, 180, SERVO_MIN, SERVO_MAX));           // левое 90°
  pwm.setPWM(R_FLAP,    0, map(90,  0, 180, R_FLAP_MIN, R_FLAP_MAX));         // правое 90° (своя калибровка!)
  pwm.setPWM(WING_TILT, 0, map(90,  0, 180, SERVO_MIN, SERVO_MAX));
  pwm.setPWM(TAIL_LIFT, 0, map(90,  0, 180, SERVO_MIN, SERVO_MAX));
}

// =======================================================
// РАЗЛОЖЕННОЕ ПОЛОЖЕНИЕ (FLY) — крылья полностью вверх
void goToFlyPose() {
  pwm.setPWM(L_FLAP,    0, map(0,   0, 180, SERVO_MIN, SERVO_MAX));           // левое → 0°
  pwm.setPWM(R_FLAP,    0, map(180, 0, 180, R_FLAP_MIN, R_FLAP_MAX));         // правое → 180° (своя калибровка!)
  pwm.setPWM(WING_TILT, 0, map(90,  0, 180, SERVO_MIN, SERVO_MAX));
  pwm.setPWM(TAIL_LIFT, 0, map(90,  0, 180, SERVO_MIN, SERVO_MAX));
}

// =======================================================
void changeMode(Mode newMode) {
  if (newMode == currentMode) return;
  modeChanging = true;

  // стопим 360° сервы складывания
  pwm.setPWM(L_FOLD, 0, 0);
  pwm.setPWM(R_FOLD, 0, 0);

  if (newMode == WALK || newMode == FREE) {
    goToNeutral();
    Serial.println(F("Режим: WALK / FREE — крылья прижаты"));
  }
  else if (newMode == FLY) {
    goToFlyPose();
    Serial.println(F("Режим: FLY — крылья разложены полностью!"));
  }

  currentMode = newMode;
  modeChanging = false;
}