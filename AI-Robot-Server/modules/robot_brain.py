"""Главный мозг робота - оркестратор всех модулей с эмоциональным движком Сорена"""
import asyncio
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List
from modules.stt import STTEngine
from modules.tts import TTSEngine
from modules.llm import LLMEngine
from modules.vision import VisionEngine
from modules.audio_buffer import AudioBuffer
from modules.servo_controller import ServoController
from config.settings import CHARACTER_DIR

logger = logging.getLogger(__name__)


class EmotionEngine:
    """Эмоциональный движок Сорена - определяет позы и LED из JSON"""

    def __init__(self, emotions_path: Path):
        self.emotions: Dict = {}
        self.current_emotion = "calm"
        self._load(emotions_path)

    def _load(self, path: Path):
        if not path.exists():
            logger.warning(f"Файл эмоций не найден: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.emotions = data.get("emotions", {})
            logger.info(f"Эмоции загружены: {list(self.emotions.keys())}")
        except Exception as e:
            logger.error(f"Ошибка загрузки эмоций: {e}")

    def get_pose(self, emotion: str) -> Optional[Dict[str, int]]:
        if emotion in self.emotions:
            return self.emotions[emotion].get("servo_pose")
        return None

    def get_eye_led(self, emotion: str) -> str:
        if emotion in self.emotions:
            return self.emotions[emotion].get("eye_led", "soft_white_low")
        return "soft_white_low"

    def get_servo_angles(self, emotion: str) -> List[int]:
        pose = self.get_pose(emotion)
        if not pose:
            return [90] * 18

        angles = []
        for i in range(18):
            key = f"S{i}"
            angles.append(pose.get(key, 90))
        return angles


class RobotBrain:
    """Основной контроллер - соединяет все модули"""

    def __init__(self):
        logger.info("=== Инициализация RobotBrain (Сорен) ===")

        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.llm = LLMEngine()
        self.vision = VisionEngine()
        self.audio_buffer = AudioBuffer()
        self.servos = ServoController()

        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        self.emotion_engine = EmotionEngine(emotions_path)

        self.vision_context = ""
        self.is_processing = False
        self.current_emotion = "calm"

        logger.info("=== RobotBrain (Сорен) готов ===")

    def set_stt_mode(self, mode: str):
        """Переключает режим STT"""
        self.stt.set_mode(mode)

    def set_tts_mode(self, mode: str):
        """Переключает режим TTS"""
        self.tts.set_mode(mode)

    def set_llm_mode(self, mode: str):
        """Переключает режим LLM"""
        self.llm.set_mode(mode)

    def get_modes(self) -> Dict[str, str]:
        """Возвращает текущие режимы всех модулей"""
        return {
            "stt": self.stt.get_mode(),
            "tts": self.tts.get_mode(),
            "llm": self.llm.get_mode()
        }

    async def process_audio_chunk(self, pcm_bytes: bytes) -> Optional[dict]:
        status, audio = self.audio_buffer.process_chunk(pcm_bytes)
        if status == "complete" and audio:
            return await self._process_speech(audio)
        return None

    async def _process_speech(self, audio_bytes: bytes) -> dict:
        self.is_processing = True
        try:
            logger.info("Распознавание речи...")
            stt_result = self.stt.transcribe(audio_bytes)

            if not stt_result["success"]:
                logger.warning("Речь не распознана")
                return self._build_empty_response()

            raw_text = stt_result["text"]
            user_text = stt_result.get("corrected_text", raw_text) or raw_text

            if user_text != raw_text:
                logger.info(f"🎯 Fuzzy: используем исправленный текст: '{user_text}'")

            logger.info(f"Пользователь: {user_text}")

            logger.info("Генерация ответа Сорена...")
            llm_result = self.llm.generate(user_text, self.vision_context)
            response_text = llm_result["text"]
            action = llm_result.get("action")
            emotion = llm_result.get("emotion", "calm")
            self.current_emotion = emotion

            logger.info(f"Сорен: {response_text}")
            logger.info(f"Эмоция: {emotion}")
            if action:
                logger.info(f"Действие: {action}")

            servo_angles = self.emotion_engine.get_servo_angles(emotion)
            eye_led = self.emotion_engine.get_eye_led(emotion)

            logger.info("Синтез речи...")
            tts_audio = self.tts.synthesize(response_text)

            if action:
                asyncio.create_task(self.servos.play_animation(action))
            else:
                self.servos.set_all_servos(servo_angles)

            return {
                "text": user_text,
                "raw_text": raw_text,
                "response": response_text,
                "audio": tts_audio,
                "action": action,
                "emotion": emotion,
                "servo_angles": servo_angles,
                "eye_led": eye_led
            }
        finally:
            self.is_processing = False

    def _build_empty_response(self) -> dict:
        return {
            "text": "",
            "raw_text": "",
            "response": "",
            "audio": b"",
            "action": None,
            "emotion": "calm",
            "servo_angles": [90] * 18,
            "eye_led": "soft_white_low"
        }

    async def process_video_frame(self, frame_bytes: bytes) -> dict:
        vision_result = self.vision.process_frame(frame_bytes)
        self.vision_context = vision_result.get("description", "")
        servo_angles = self.vision.get_servo_angles_from_pose()
        face_offset = self.vision.get_face_offset(640, 480)

        if vision_result.get("face_detected"):
            servo_angles[16] = int(90 - face_offset[0] * 45)
            servo_angles[17] = int(90 + face_offset[1] * 30)

        return {
            "servo_angles": servo_angles,
            "face_offset": face_offset,
            "description": self.vision_context,
            "objects": vision_result.get("objects", []),
            "face_detected": vision_result.get("face_detected", False)
        }

    async def handle_command(self, command: dict) -> dict:
        cmd_type = command.get("type")

        if cmd_type == "servo":
            self.servos.set_servo(command["id"], command["angle"])
            return {"status": "ok", "servo": command["id"], "angle": command["angle"]}

        elif cmd_type == "servo_multi":
            self.servos.set_all_servos(command["angles"])
            return {"status": "ok", "angles": command["angles"]}

        elif cmd_type == "animation":
            asyncio.create_task(self.servos.play_animation(command["name"]))
            return {"status": "ok", "animation": command["name"]}

        elif cmd_type == "text":
            try:
                from modules.fuzzy_matcher import correct_speech_text
                raw_text = command["text"]
                corrected_text = correct_speech_text(raw_text)
                if corrected_text != raw_text:
                    logger.info(f"🎯 Fuzzy (text): '{raw_text}' → '{corrected_text}'")
                user_text = corrected_text
            except ImportError:
                user_text = command["text"]

            llm_result = self.llm.generate(user_text, self.vision_context)
            emotion = llm_result.get("emotion", "calm")
            servo_angles = self.emotion_engine.get_servo_angles(emotion)
            eye_led = self.emotion_engine.get_eye_led(emotion)
            tts_audio = self.tts.synthesize(llm_result["text"])

            if llm_result.get("action"):
                asyncio.create_task(self.servos.play_animation(llm_result["action"]))
            else:
                self.servos.set_all_servos(servo_angles)

            return {
                "status": "ok",
                "text": user_text,
                "raw_text": command.get("text", ""),
                "response": llm_result["text"],
                "audio": tts_audio.hex() if tts_audio else "",
                "action": llm_result.get("action"),
                "emotion": emotion,
                "servo_angles": servo_angles,
                "eye_led": eye_led
            }

        elif cmd_type == "get_status":
            return {
                "status": "ok",
                "servo_angles": self.servos.get_current_angles(),
                "processing": self.is_processing,
                "vision_context": self.vision_context,
                "current_emotion": self.current_emotion,
                "modes": self.get_modes()
            }

        elif cmd_type == "clear_history":
            self.llm.clear_history()
            self.current_emotion = "calm"
            return {"status": "ok", "message": "История очищена"}

        elif cmd_type == "set_mode":
            module = command.get("module")
            mode = command.get("mode")
            if module == "stt":
                self.set_stt_mode(mode)
            elif module == "tts":
                self.set_tts_mode(mode)
            elif module == "llm":
                self.set_llm_mode(mode)
            return {"status": "ok", "module": module, "mode": mode, "modes": self.get_modes()}

        else:
            return {"status": "error", "message": f"Неизвестная команда: {cmd_type}"}

    def shutdown(self):
        logger.info("Завершение работы RobotBrain...")
        self.vision.release()
