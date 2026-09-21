"""Контроллер сервоприводов с загрузкой поз из Soren_emotions.json"""
import logging
import asyncio
from typing import List, Dict, Optional, Callable, Awaitable
from pathlib import Path
from config.settings import SERVO_CONFIG, CHARACTER_DIR
from modules.animation_loader import animation_book

logger = logging.getLogger(__name__)


class ServoController:
    """Контроллер 18 сервоприводов (16 PCA9685 + 2 GPIO)"""

    def __init__(self):
        self.config = SERVO_CONFIG
        self.current_angles = [90] * 18
        self.target_angles = [90] * 18
        self.is_animating = False
        self.hardware_available = False
        self.emotion_poses: Dict[str, List[int]] = {}
        # Объекты реального железа — создаются в enable_hardware(), используются
        # в _send_to_hardware(). None пока железо не включено.
        self._pca_servos = None      # список из 16 adafruit_motor.servo.Servo (каналы PCA9685)
        self._gpio_pwms = None       # список PWM-объектов RPi.GPIO для 2 доп. серв
        # Коллбэк для отправки кадров на ESP32 по сети (WebSocket)
        self.on_servo_frame: Optional[Callable[[List[int]], Awaitable[None]]] = None

        # Загружаем позы из Soren_emotions.json
        self._load_emotion_poses()

        logger.info("ServoController инициализирован (режим эмуляции)")

    def _load_emotion_poses(self):
        """Загружает позы эмоций из character/Soren_emotions.json"""
        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        if not emotions_path.exists():
            logger.warning(f"Файл эмоций не найден: {emotions_path}")
            return

        try:
            import json
            with open(emotions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for emotion_name, emotion_data in data.get("emotions", {}).items():
                pose = emotion_data.get("servo_pose", {})
                angles = []
                for i in range(18):
                    key = f"S{i}"
                    angles.append(pose.get(key, 90))
                self.emotion_poses[emotion_name] = angles

            logger.info(f"Позы эмоций загружены: {list(self.emotion_poses.keys())}")
        except Exception as e:
            logger.error(f"Ошибка загрузки поз эмоций: {e}")

    def _notify_servo_frame(self):
        """Асинхронно уведомляет подписчиков (WebSocket → ESP32) о текущих углах."""
        if self.on_servo_frame is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.on_servo_frame(self.current_angles.copy()))
        except RuntimeError:
            # Нет запущенного event loop (тесты, импорт)
            pass

    def set_emotion_pose(self, emotion: str):
        """Устанавливает позу по эмоции"""
        if emotion in self.emotion_poses:
            self.set_all_servos(self.emotion_poses[emotion])
            logger.info(f"Поза эмоции '{emotion}' установлена")
        else:
            logger.warning(f"Поза для эмоции '{emotion}' не найдена")

    def set_servo(self, servo_id: int, angle: int, notify: bool = True):
        """Устанавливает угол одного сервопривода"""
        if not 0 <= servo_id < 18:
            logger.warning(f"Неверный ID серво: {servo_id}")
            return

        angle = max(self.config["min_angle"], min(self.config["max_angle"], angle))
        self.target_angles[servo_id] = angle
        self.current_angles[servo_id] = angle

        if self.hardware_available:
            self._send_to_hardware(servo_id, angle)

        if notify:
            self._notify_servo_frame()

        logger.info(f"Серво {servo_id} → {angle}°")

    def set_all_servos(self, angles: List[int], notify: bool = True):
        """Устанавливает углы всех сервоприводов"""
        if len(angles) != 18:
            logger.warning(f"Неверное количество углов: {len(angles)} != 18")
            return

        for i, angle in enumerate(angles):
            angle = max(self.config["min_angle"], min(self.config["max_angle"], angle))
            self.target_angles[i] = angle
            self.current_angles[i] = angle
            if self.hardware_available:
                self._send_to_hardware(i, angle)

        if notify:
            self._notify_servo_frame()

        logger.info(f"Все сервы установлены: {angles}")

    def get_current_angles(self) -> List[int]:
        """Возвращает текущие углы"""
        return self.current_angles.copy()

    def apply_device_state(self, angles: List[int]):
        """Принимает РЕАЛЬНЫЕ углы серв, присланные платой (сообщение
        {"type":"device_state", ...} от ESP32 — см. websocket_server.py).
        Обновляет current/target_angles БЕЗ повторной отправки на железо и
        БЕЗ notify — эти данные и так пришли от платы, эхом слать их обратно
        не нужно (получится бесконечная петля туда-обратно).

        Это единственный источник, которому сервер доверяет как "истине" о
        физическом положении серв: сам сервер их не двигает (hardware_available
        тут всегда False), реальное движение — на ESP32."""
        if len(angles) != 18:
            logger.warning(f"device_state: неверное количество углов: {len(angles)} != 18, игнорирую")
            return

        clamped = [max(self.config["min_angle"], min(self.config["max_angle"], int(a))) for a in angles]
        self.current_angles = clamped
        self.target_angles = clamped.copy()
        logger.info(f"📥 Состояние серв синхронизировано с платой: {clamped}")

    async def smooth_return_to_default(
        self,
        on_frame: Optional[Callable[[List[int]], Awaitable[None]]] = None,
        duration_ms: int = 1500,
        fps: int = 30,
    ):
        """Плавно возвращает ВСЕ сервы из текущего состояния (self.current_angles —
        в идеале только что синхронизированного с платой через apply_device_state)
        в позу по умолчанию (90° на каждой).

        В отличие от interpolate_to_target() — та пересчитывает шаги мгновенно,
        без реальной задержки между ними, и уведомляет только один раз в конце,
        то есть физически это "телепорт", а не плавное движение — эта корутина
        реально ждёт между кадрами (asyncio.sleep) и на каждом шаге вызывает
        on_frame, чтобы кадры реально долетали до ESP32 (и панели) по сети,
        а не оставались только в Python-состоянии."""
        default_angles = [90] * 18
        start_angles = self.current_angles.copy()
        if start_angles == default_angles:
            logger.info("Сервы уже в позе по умолчанию — плавный возврат не требуется")
            return

        steps = max(1, int(duration_ms / 1000 * fps))
        step_delay = (duration_ms / 1000) / steps

        for step in range(1, steps + 1):
            t = step / steps
            frame = [
                int(start_angles[i] + (default_angles[i] - start_angles[i]) * t)
                for i in range(18)
            ]
            self.set_all_servos(frame, notify=False)
            if on_frame:
                await on_frame(frame)
            await asyncio.sleep(step_delay)

        logger.info("Сервы плавно возвращены в позу по умолчанию (90°)")

    async def play_animation(self, animation_name: str, on_frame: Optional[Callable[[List[int]], Awaitable[None]]] = None):
        """Воспроизводит анимацию.

        on_frame — опциональный async-коллбэк, вызывается после каждого кадра
        с текущими углами (список из 18 int). Без него анимация — чистая
        Python-симуляция (self.hardware_available всегда False, реального
        железа тут нет): кадры нигде не появляются, кроме current_angles.
        Коллбэк — единственный способ доставить кадры на физический ESP32 по
        сети (см. RobotBrain.on_servo_frame, подключается в websocket_server.py).

        Анимация ищется среди встроенных (config.settings.ANIMATIONS) И
        пользовательских (character/animations/*.json, см. modules/animation_loader.py) —
        обе живут в общем реестре animation_book."""
        animation = animation_book.get_frames(animation_name)
        if animation is None:
            logger.warning(f"Анимация не найдена: {animation_name}")
            return

        if self.is_animating:
            logger.warning("Анимация уже воспроизводится")
            return

        self.is_animating = True
        # Fallback: если внешний on_frame не передан, используем внутренний
        effective_on_frame = on_frame or self.on_servo_frame

        try:
            for i, keyframe in enumerate(animation):
                self.set_all_servos(keyframe["servos"], notify=False)
                if effective_on_frame:
                    await effective_on_frame(keyframe["servos"])
                if i < len(animation) - 1:
                    next_time = animation[i + 1]["time"]
                    current_time = keyframe["time"]
                    await asyncio.sleep((next_time - current_time) / 1000)
        finally:
            self.is_animating = False

    def interpolate_to_target(self, target: List[int], steps: int = 10, step_delay_ms: int = 50):
        """Плавно интерполирует текущие углы к целевым"""
        if len(target) != 18:
            return

        for step in range(1, steps + 1):
            t = step / steps
            new_angles = [
                int(self.current_angles[i] + (target[i] - self.current_angles[i]) * t)
                for i in range(18)
            ]
            self.set_all_servos(new_angles, notify=False)

        self._notify_servo_frame()

    def _angle_to_gpio_duty(self, angle: int) -> float:
        """Угол 0-180° → duty cycle (%) для программного PWM на 50 Гц.
        Стандартный диапазон импульса серво: 1мс (0°) .. 2мс (180°) из 20мс периода
        → 5% .. 10% duty cycle. Если твои серво требуют другой диапазон импульсов
        (некоторые ходят от 0.5мс до 2.5мс, т.е. 2.5%..12.5%) — поменяй DUTY_MIN/DUTY_MAX ниже.
        """
        DUTY_MIN, DUTY_MAX = 5.0, 10.0
        angle = max(0, min(180, angle))
        return DUTY_MIN + (angle / 180.0) * (DUTY_MAX - DUTY_MIN)

    def _send_to_hardware(self, servo_id: int, angle: int):
        """Отправляет команду на реальное железо (PCA9685 для каналов 0-15,
        программный PWM на GPIO для 16-17)"""
        if servo_id < 16:
            if not self._pca_servos:
                return
            try:
                self._pca_servos[servo_id].angle = angle
            except Exception as e:
                logger.error(f"Ошибка записи в PCA9685 (servo {servo_id}): {e}")
        else:
            gpio_idx = servo_id - 16
            if not self._gpio_pwms or gpio_idx >= len(self._gpio_pwms):
                return
            try:
                duty = self._angle_to_gpio_duty(angle)
                self._gpio_pwms[gpio_idx].ChangeDutyCycle(duty)
            except Exception as e:
                logger.error(f"Ошибка записи в GPIO PWM (servo {servo_id}): {e}")

    def enable_hardware(self):
        """Включает управление реальным железом"""
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
            from adafruit_motor import servo as adafruit_servo

            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=self.config["pca9685_address"])
            self.pca.frequency = self.config["pca9685_freq"]

            # ВАЖНО: min_pulse/max_pulse (мкс) должны совпадать с твоими серво.
            # Если в SERVO_CONFIG таких ключей нет — используются безопасные
            # дефолты 500/2500 мкс (полный диапазон большинства аналоговых серво).
            min_pulse = self.config.get("min_pulse", 500)
            max_pulse = self.config.get("max_pulse", 2500)
            self._pca_servos = [
                adafruit_servo.Servo(self.pca.channels[i], min_pulse=min_pulse, max_pulse=max_pulse)
                for i in range(16)
            ]

            # 2 доп. серво напрямую на GPIO (программный PWM 50 Гц).
            # Ожидается SERVO_CONFIG["gpio_pins"] = [pin_для_servo16, pin_для_servo17].
            # Если ключа нет — GPIO-серво просто не будут работать (PCA9685-каналы
            # заработают в любом случае).
            gpio_pins = self.config.get("gpio_pins", [])
            if gpio_pins:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                self._gpio_pwms = []
                for pin in gpio_pins:
                    GPIO.setup(pin, GPIO.OUT)
                    pwm = GPIO.PWM(pin, 50)
                    pwm.start(self._angle_to_gpio_duty(90))
                    self._gpio_pwms.append(pwm)
            else:
                logger.warning("SERVO_CONFIG['gpio_pins'] не задан — серво 16/17 (GPIO) работать не будут")

            self.hardware_available = True
            logger.info("Аппаратное управление сервами активировано")

            # Сразу выставляем серво в текущие сохранённые углы — иначе до первой
            # команды set_servo/set_all_servos физическое положение не определено.
            for i, angle in enumerate(self.current_angles):
                self._send_to_hardware(i, angle)
        except Exception as e:
            logger.error(f"Не удалось инициализировать железо: {e}")
            self.hardware_available = False