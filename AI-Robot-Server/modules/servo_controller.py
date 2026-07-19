"""Контроллер сервоприводов с загрузкой поз из Soren_emotions.json"""
import logging
import asyncio
from typing import List, Dict
from pathlib import Path
from config.settings import SERVO_CONFIG, ANIMATIONS, CHARACTER_DIR

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

    def set_emotion_pose(self, emotion: str):
        """Устанавливает позу по эмоции"""
        if emotion in self.emotion_poses:
            self.set_all_servos(self.emotion_poses[emotion])
            logger.info(f"Поза эмоции '{emotion}' установлена")
        else:
            logger.warning(f"Поза для эмоции '{emotion}' не найдена")

    def set_servo(self, servo_id: int, angle: int):
        """Устанавливает угол одного сервопривода"""
        if not 0 <= servo_id < 18:
            logger.warning(f"Неверный ID серво: {servo_id}")
            return

        angle = max(self.config["min_angle"], min(self.config["max_angle"], angle))
        self.target_angles[servo_id] = angle
        self.current_angles[servo_id] = angle

        if self.hardware_available:
            self._send_to_hardware(servo_id, angle)
        else:
            logger.debug(f"Серво {servo_id} → {angle}°")

    def set_all_servos(self, angles: List[int]):
        """Устанавливает углы всех сервоприводов"""
        if len(angles) != 18:
            logger.warning(f"Неверное количество углов: {len(angles)} != 18")
            return

        for i, angle in enumerate(angles):
            self.set_servo(i, angle)

    def get_current_angles(self) -> List[int]:
        """Возвращает текущие углы"""
        return self.current_angles.copy()

    async def play_animation(self, animation_name: str):
        """Воспроизводит анимацию"""
        if animation_name not in ANIMATIONS:
            logger.warning(f"Анимация не найдена: {animation_name}")
            return

        if self.is_animating:
            logger.warning("Анимация уже воспроизводится")
            return

        self.is_animating = True
        animation = ANIMATIONS[animation_name]

        try:
            for i, keyframe in enumerate(animation):
                self.set_all_servos(keyframe["servos"])
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
            self.set_all_servos(new_angles)

    def _send_to_hardware(self, servo_id: int, angle: int):
        """Отправляет команду на реальное железо (PCA9685 или GPIO)"""
        pass

    def enable_hardware(self):
        """Включает управление реальным железом"""
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685

            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=self.config["pca9685_address"])
            self.pca.frequency = self.config["pca9685_freq"]

            self.hardware_available = True
            logger.info("Аппаратное управление сервами активировано")
        except Exception as e:
            logger.error(f"Не удалось инициализировать железо: {e}")
            self.hardware_available = False