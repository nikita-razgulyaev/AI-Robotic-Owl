"""Главный мозг робота — оркестратор с эмоциональным движком и памятью"""
import asyncio
import logging
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Callable, Awaitable
from modules.stt import STTEngine
from modules.tts import TTSEngine
from modules.llm import LLMEngine
from modules.vision import VisionEngine
from modules.audio_buffer import AudioBuffer
from modules.servo_controller import ServoController
from modules.memory import MemoryManager
from modules import memory_flags
from modules.quick_answers import quick_answers
from modules.wake_word import strip_wake_word
from config.settings import (
    CHARACTER_DIR,
    FAST_MODE,
    LLM_LOCAL_MODELS,
    LLM_CLOUD_MODELS,
    WHISPER_MODELS,
    QUICK_ANSWERS_ENABLED,
    FACE_TRACKING_ENABLED,
    DIALOG_ACTIVE_TIMEOUT_SEC,
    PROVISIONAL_DIALOG_TIMEOUT_SEC,
    FACE_PAN_SERVO,
    FACE_TILT_SERVO,
    FACE_PAN_GAIN,
    FACE_TILT_GAIN,
    HEAD_SMOOTHING_ALPHA,
    FACE_DEADZONE_X,
    FACE_DEADZONE_Y,
    EXIT_EASE_ALPHA,
    EXIT_EASE_EPSILON,
)

logger = logging.getLogger(__name__)


class EmotionEngine:
    """Эмоциональный движок Сорена — позы и LED из JSON"""

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
    """Основной контроллер — соединяет все модули с памятью"""

    def __init__(self):
        logger.info("=== Инициализация RobotBrain (Сорен + Vector RAG) ===")

        # Память ДО остальных модулей
        self.memory = MemoryManager()

        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.llm = LLMEngine(memory_manager=self.memory)
        self.vision = VisionEngine()
        self.audio_buffer = AudioBuffer()
        self.servos = ServoController()

        emotions_path = CHARACTER_DIR / "Soren_emotions.json"
        self.emotion_engine = EmotionEngine(emotions_path)

        self.vision_context = ""
        self.is_processing = False
        self.current_emotion = "calm"

        # Кэш последней озвученной фразы — на будущее, для восстановления
        # передачи после обрыва Wi-Fi (см. audio_resume_request от ESP32).
        # utterance_id инкрементируется на каждый новый синтез, чтобы файл,
        # который реально шлёт AUDI-чанки, мог отличить "актуальную" фразу
        # от устаревшей и повторно отправить именно её, а не всю историю.
        self.last_tts_audio: bytes = b""
        self.last_tts_utterance_id: int = 0

        # Коллбэк для доставки кадров анимации на физическое устройство — без
        # него ServoController.play_animation() работает как чистая Python-
        # симуляция (hardware_available всегда False), и жесты/анимации
        # реально не двигают серво. Подключается снаружи в websocket_server.py
        # (см. startup()), потому что только там есть доступ к device_connections.
        self.on_servo_frame: Optional[Callable[[List[int]], Awaitable[None]]] = None

        # === Фоновые потоки для тяжёлых синхронных операций ===
        # Не даём CV (YOLO/DNN/Pose) и STT/LLM/TTS блокировать asyncio event loop —
        # иначе на время одного тяжёлого вызова встают ВСЕ подключения (панель, /speak, /voice,
        # другие устройства), а не только текущее. Два ОТДЕЛЬНЫХ пула (а не общий executor
        # asyncio), чтобы долгая генерация ответа LLM не задерживала обработку видео —
        # слежение за лицом должно продолжаться, даже пока робот "думает" над ответом.
        # max_workers=1 в каждом — этого достаточно (устройство одно), и это же служит
        # бесплатной защитой от случайного параллельного запуска двух тяжёлых кадров разом.
        self._cv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cv-worker")
        self._speech_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-worker")

        # === Состояние "активного диалога" (для слежения за лицом) ===
        # Робот следит за лицом только пока идёт общение: сразу после
        # распознанной речи/текстовой команды и ещё DIALOG_ACTIVE_TIMEOUT_SEC
        # секунд после последнего обмена репликами (чтобы не дёргать голову
        # между отдельными фразами диалога).
        # Два независимых таймера с разным уровнем доверия к сигналу:
        #  - last_interaction_ts ("подтверждённый"): продлевается, когда речь РЕАЛЬНО
        #    распознана / пришла текстовая команда — держит окно DIALOG_ACTIVE_TIMEOUT_SEC,
        #    а после ответа ещё и на время самого озвучивания (см. _process_speech).
        #  - provisional_interaction_ts ("предварительный"): продлевается уже на VAD
        #    speech_start/speech, ДО распознавания — короткое окно PROVISIONAL_DIALOG_TIMEOUT_SEC,
        #    чтобы голова начинала реагировать сразу, но быстро отпускала, если это
        #    оказался просто шум/кашель, а не обращение к роботу.
        # Диалог считается активным, если жив хотя бы один из двух таймеров.
        self.last_interaction_ts = 0.0
        self.provisional_interaction_ts = 0.0

        # === Сглаживание слежения за лицом + "мёртвая зона" ===
        self._smoothed_face_offset = [0.0, 0.0]   # (x, y), между кадрами
        self._last_target_track_id = None         # для детекции смены цели (снап в центр)
        self._was_dialog_active = False            # для детекции старта диалога (снап в центр)

        # Последнее известное состояние лица — используется при отрисовке кадров ПАНЕЛИ
        # (VIDP), которые приходят отдельно от кадров детекции (VIDE) и сами по себе
        # детекцию не запускают (см. update_panel_frame / render_panel_jpeg).
        self._last_face_detected = False
        self._last_face_bbox = None
        self._last_faces_count = 0

        logger.info("=== RobotBrain (Сорен + Vector RAG) готов ===")

    def mark_dialog_active(self):
        """Подтверждённое взаимодействие (распознанная речь / текстовая команда) —
        продлевает основное окно слежения за лицом"""
        self.last_interaction_ts = time.time()

    def mark_dialog_provisional(self):
        """Предварительный сигнал (VAD засёк голос, ещё не распознано) —
        продлевает короткое окно, чтобы голова среагировала раньше"""
        self.provisional_interaction_ts = time.time()

    def is_dialog_active(self) -> bool:
        """Активен ли сейчас диалог (для решения — следить за лицом или нет)"""
        if self.is_processing:
            return True

        now = time.time()

        confirmed_active = (
            self.last_interaction_ts > 0 and
            (now - self.last_interaction_ts) < DIALOG_ACTIVE_TIMEOUT_SEC
        )
        provisional_active = (
            self.provisional_interaction_ts > 0 and
            (now - self.provisional_interaction_ts) < PROVISIONAL_DIALOG_TIMEOUT_SEC
        )
        return confirmed_active or provisional_active

    def is_wake_session_active(self) -> bool:
        """Можно ли обрабатывать команду БЕЗ повторного "Сорен" — та же
        логика окна, что и у is_dialog_active(), но нарочно только по
        подтверждённому таймеру (last_interaction_ts), без provisional:
        одного VAD-сигнала "кто-то говорит" недостаточно, чтобы считать
        будильник уже произнесённым — сессию открывает только реально
        распознанная команда (см. mark_dialog_active())."""
        return (
            self.last_interaction_ts > 0 and
            (time.time() - self.last_interaction_ts) < DIALOG_ACTIVE_TIMEOUT_SEC
        )

    def set_stt_mode(self, mode: str):
        self.stt.set_mode(mode)

    def set_tts_mode(self, mode: str):
        self.tts.set_mode(mode)

    def set_llm_mode(self, mode: str):
        self.llm.set_mode(mode)

    def get_modes(self) -> Dict[str, str]:
        return {
            "stt": self.stt.get_mode(),
            "tts": self.tts.get_mode(),
            "llm": self.llm.get_mode()
        }

    def get_memory_flags(self) -> Dict[str, bool]:
        """Текущее состояние всех уровней памяти (stm/ltm/profile/rag)"""
        return memory_flags.get_flags()

    def set_memory_flag(self, level: str, enabled: bool) -> Dict[str, bool]:
        """Включает/выключает уровень памяти из веб-панели, сохраняет на диск
        и при необходимости лениво поднимает тяжёлые ресурсы (Qdrant/RAG),
        если они не были загружены при старте (например, из-за fast_mode)"""
        flags = memory_flags.set_flag(level, enabled)
        if enabled and level == "ltm" and self.memory:
            self.memory.long_term.ensure_ready()
        if enabled and level == "rag":
            self.llm.ensure_rag_loaded()
        return flags

    def get_model_config(self) -> Dict:
        """Текущий выбор моделей + куратированные списки для выпадающих
        списков в /panel (зависят от текущего режима local/cloud)"""
        return {
            "llm": {
                "mode": self.llm.get_mode(),
                "current": self.llm.get_current_model_id(),
                "local_models": LLM_LOCAL_MODELS,
                "cloud_models": LLM_CLOUD_MODELS,
            },
            "stt": {
                "current": self.stt.get_current_model_id(),
                "models": WHISPER_MODELS,
            },
            "tts": {
                "mode": self.tts.get_mode(),
                "current": self.tts.get_current_speaker(),
                "speakers": self.tts.get_available_speakers(),
            },
        }

    def set_llm_model(self, model_id: str) -> bool:
        return self.llm.set_model(model_id)

    def set_stt_model(self, model_id: str):
        self.stt.set_model(model_id)

    def set_tts_speaker(self, speaker: str):
        self.tts.set_speaker(speaker)

    def get_quick_answers_status(self) -> Dict:
        return {"enabled": QUICK_ANSWERS_ENABLED, "count": quick_answers.count()}

    def reload_quick_answers(self) -> Dict:
        """Перечитывает character/quick_answers.json с диска — без рестарта сервера"""
        count = quick_answers.reload()
        return {"enabled": QUICK_ANSWERS_ENABLED, "count": count}

    async def process_audio_chunk(self, pcm_bytes: bytes) -> Optional[dict]:
        status, audio = self.audio_buffer.process_chunk(pcm_bytes)

        if status == "speech_start":
            # Начало новой фразы — сразу продлеваем предварительное окно
            # диалога, чтобы голова начала реагировать ДО того, как фраза
            # будет распознана (выбор цели слежения идёт по размеру лица
            # в кадре — см. VisionEngine._select_target).
            self.mark_dialog_provisional()
        elif status == "speech":
            # Человек продолжает говорить — держим предварительное окно свежим,
            # чтобы оно не истекло на середине длинной фразы.
            self.mark_dialog_provisional()
        elif status == "complete" and audio:
            return await self._process_speech(audio)

        return None

    async def _process_speech(self, audio_bytes: bytes) -> dict:
        self.is_processing = True
        try:
            loop = asyncio.get_event_loop()
            # STT + LLM + TTS — тяжёлые синхронные вызовы (LLM может думать секундами) —
            # выполняем в отдельном потоке, чтобы не блокировать event loop целиком.
            result = await loop.run_in_executor(self._speech_executor, self._process_speech_sync, audio_bytes)

            # asyncio.create_task требует запущенный event loop в текущем потоке,
            # поэтому анимацию запускаем здесь, а не внутри воркер-потока.
            action = result.get("action")
            if action:
                asyncio.create_task(self.servos.play_animation(action, on_frame=self.on_servo_frame))
            else:
                # Явно доставляем эмоциональную позу на ESP32 (из потока коллбэк не ушёл)
                await self._notify_servo_frame()

            return result
        finally:
            self.is_processing = False

    def _process_speech_sync(self, audio_bytes: bytes) -> dict:
        """Синхронное тело обработки речи (STT+LLM+TTS) — выполняется в фоновом потоке"""
        logger.info("Распознавание речи...")
        stt_result = self.stt.transcribe(audio_bytes)

        if not stt_result["success"]:
            logger.warning("Речь не распознана")
            return self._build_empty_response()

        raw_text = stt_result["text"]
        user_text = stt_result.get("corrected_text", raw_text) or raw_text

        if user_text != raw_text:
            logger.info(f"🎯 Fuzzy: используем исправленный текст: '{user_text}'")

        # Будильник "Сорен" — команды принимаются только сразу после него,
        # либо пока открыта сессия диалога (см. is_wake_session_active).
        # Иначе случайная реплика в комнате ("а он вообще слушает?") не
        # должна улетать в LLM.
        command_text = strip_wake_word(user_text)
        if command_text is None:
            if self.is_wake_session_active():
                command_text = user_text
            else:
                logger.info(f"🔇 Будильник не найден, фраза проигнорирована: '{user_text}'")
                return self._build_empty_response()
        else:
            logger.info(f"👂 Будильник распознан: '{user_text}' → '{command_text or '(пусто)'}'")

        logger.info(f"Пользователь: {command_text or '(только будильник, без команды)'}")
        self.mark_dialog_active()

        # Словарь быстрых ответов → память → LLM — вся логика в одном месте
        # (см. generate_reply), чтобы не размножать её по каждой точке входа.
        llm_result = self.generate_reply(command_text)

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

        if tts_audio:
            self.last_tts_audio = tts_audio
            self.last_tts_utterance_id += 1

            # Продлеваем подтверждённое окно диалога на время самого озвучивания ответа —
            # оба TTS-движка отдают PCM 16-bit mono 48kHz (см. modules/tts.py), поэтому
            # длительность оцениваем как байты / (48000 * 2). Так голова не "отключится"
            # на середине длинного ответа Сорена.
            estimated_playback_sec = len(tts_audio) / (48000 * 2)
            self.last_interaction_ts = time.time() + estimated_playback_sec

        if not action:
            # Анимацию (если есть) запустит async-обёртка _process_speech — ей нужен event loop.
            # notify=False, т.к. мы в фоновом потоке без event loop — уведомим вручную после return.
            self.servos.set_all_servos(servo_angles, notify=False)

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

    def _try_quick_answer(self, user_text: str) -> Optional[dict]:
        """Проверяет словарь быстрых ответов (character/quick_answers.json) ДО
        похода в память и LLM. Если находится совпадение — Сорен отвечает
        мгновенно, без единого обращения к llama.cpp/облаку. Возвращает dict
        в том же формате, что и LLMEngine.generate(), либо None (тогда
        вызывающий код идёт обычным путём через LLM)."""
        if not QUICK_ANSWERS_ENABLED:
            return None

        qa = quick_answers.find(user_text)
        if qa is None:
            return None

        logger.info(f"⚡ Быстрый ответ '{qa.id}' (без LLM): {user_text!r} → {qa.response!r}")
        return {"text": qa.response, "action": qa.action, "emotion": qa.emotion or "calm"}

    def generate_reply(self, user_text: str) -> dict:
        """Единая точка получения ответа Сорена на текст пользователя — сначала
        словарь быстрых ответов, затем (если не сработало) память + LLM.

        ВАЖНО: это единственное место, где должен вызываться self.llm.generate().
        Все остальные точки входа (голос, текстовый чат по WS, HTTP /speak,
        HTTP /voice) обязаны идти через этот метод, а не звать
        self.llm.generate() напрямую — иначе словарь быстрых ответов будет
        молча пропущен именно для этого пути.

        Возвращает dict {"text", "action", "emotion"} — тот же формат,
        что и LLMEngine.generate().

        user_text == "" — особый случай: будильник "Сорен" был произнесён/
        написан без команды следом (позвали по имени). Отвечаем коротким
        "Да?" без похода в quick_answers/LLM — там всё равно нечего искать.
        """
        if not user_text.strip():
            return {"text": "Да?", "action": None, "emotion": "calm"}

        quick = self._try_quick_answer(user_text)
        if quick is not None:
            if self.memory:
                self.memory.record_interaction(user_text, quick["text"], quick["emotion"])
            return quick

        memory_context = self._build_memory_context(user_text)
        return self.llm.generate(user_text, self.vision_context, memory_context)

    def _build_memory_context(self, user_text: str) -> str:
        """Собирает контекст для LLM из памяти — каждый уровень (stm/ltm/profile)
        независимо включается/выключается через memory_flags (панель /panel)"""
        flags = memory_flags.get_flags()

        # Краткосрочная память — последние реплики текущей сессии (RAM, не хранилище)
        stm = self.memory.short_term.get_context() if flags.get("stm", True) else ""

        parts = []
        if stm:
            parts.append(f"Последние реплики:\n{stm}")

        # Долгосрочная память — релевантные воспоминания (Qdrant)
        if flags.get("ltm", True):
            ltm = self.memory.get_relevant_memories(user_text)
            if ltm:
                parts.append(f"Релевантные воспоминания:\n{ltm}")

        # Эмоциональный профиль
        if flags.get("profile", True):
            dom = max(self.memory.profile.dominant_emotions, key=self.memory.profile.dominant_emotions.get)
            parts.append(f"Доминантная эмоция пользователя: {dom}")

            # Приветствие по времени
            greeting = self.memory.get_day_greeting()
            if greeting and not stm:
                parts.append(f"Приветствие: {greeting}")

        return "\n\n".join(parts)

    def _handle_text_command_sync(self, user_text: str) -> dict:
        """Синхронное тело обработки текстовой команды (LLM+TTS) — выполняется в фоновом потоке"""
        llm_result = self.generate_reply(user_text)

        emotion = llm_result.get("emotion", "calm")
        servo_angles = self.emotion_engine.get_servo_angles(emotion)
        eye_led = self.emotion_engine.get_eye_led(emotion)
        tts_audio = self.tts.synthesize(llm_result["text"])

        if tts_audio:
            self.last_tts_audio = tts_audio
            self.last_tts_utterance_id += 1

        if not llm_result.get("action"):
            # Анимацию (если есть) запустит handle_command — ей нужен event loop.
            # notify=False, т.к. мы в фоновом потоке — уведомим вручную после return.
            self.servos.set_all_servos(servo_angles, notify=False)

        return {
            "response": llm_result["text"],
            "audio": tts_audio,
            "action": llm_result.get("action"),
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led
        }

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
        loop = asyncio.get_event_loop()
        # CV-обработка (YOLO/DNN-детектор лица/Pose) — тяжёлая и синхронная,
        # выполняем в отдельном потоке, чтобы не блокировать event loop на время кадра.
        return await loop.run_in_executor(self._cv_executor, self._process_video_frame_sync, frame_bytes)

    def _process_video_frame_sync(self, frame_bytes: bytes) -> dict:
        """Синхронное тело обработки видеокадра — выполняется в фоновом потоке"""
        vision_result = self.vision.process_frame(frame_bytes)
        self.vision_context = vision_result.get("description", "")

        face_detected = vision_result.get("face_detected", False)
        target_track_id = vision_result.get("target_track_id")
        raw_offset = self.vision.get_face_offset(640, 480)

        # Слежение за лицом включается ТОЛЬКО во время активного диалога
        dialog_active = FACE_TRACKING_ENABLED and self.is_dialog_active()

        # Единственные сервы, реагирующие на зрение, — голова. Крылья на позу
        # тела не реагируют: get_servo_angles_from_pose() в vision.py доступна
        # на случай явной позы/жеста по команде, но из видео-пайплайна не вызывается.
        servo_angles = self.servos.get_current_angles()

        if dialog_active and face_detected:
            # "Снап" в центр нового лица — без сглаживания и без мёртвой зоны —
            # если диалог только что начался ИЛИ переключились на другого говорящего.
            # Физически голова всё равно доедет плавно за счёт интерполяции на ESP32.
            is_reorient = (not self._was_dialog_active) or (target_track_id != self._last_target_track_id)

            if is_reorient:
                self._smoothed_face_offset = [raw_offset[0], raw_offset[1]]
            else:
                in_deadzone = (
                    abs(raw_offset[0]) < FACE_DEADZONE_X and
                    abs(raw_offset[1]) < FACE_DEADZONE_Y
                )
                if not in_deadzone:
                    # Экспоненциальное сглаживание — гасит дрожание из-за шума детектора
                    self._smoothed_face_offset[0] += HEAD_SMOOTHING_ALPHA * (raw_offset[0] - self._smoothed_face_offset[0])
                    self._smoothed_face_offset[1] += HEAD_SMOOTHING_ALPHA * (raw_offset[1] - self._smoothed_face_offset[1])
                # если в мёртвой зоне — сглаженное состояние не трогаем, голова держит текущее положение

            servo_angles[FACE_PAN_SERVO] = int(90 - self._smoothed_face_offset[0] * FACE_PAN_GAIN)
            servo_angles[FACE_TILT_SERVO] = int(90 + self._smoothed_face_offset[1] * FACE_TILT_GAIN)

        elif dialog_active and not face_detected:
            # Диалог всё ещё идёт, лицо на мгновение потерялось (пара кадров) —
            # просто держим текущее положение головы, ждём, пока лицо снова найдётся.
            # Сглаженное состояние не трогаем, чтобы не начинать "выход" раньше времени.
            pass

        else:
            # Диалог закончился (или ещё не начинался) — плавно "отпускаем" голову
            # обратно в центр, а не оставляем её резко висеть в последней позиции.
            still_offset = (
                abs(self._smoothed_face_offset[0]) > EXIT_EASE_EPSILON or
                abs(self._smoothed_face_offset[1]) > EXIT_EASE_EPSILON
            )
            if still_offset:
                self._smoothed_face_offset[0] += EXIT_EASE_ALPHA * (0.0 - self._smoothed_face_offset[0])
                self._smoothed_face_offset[1] += EXIT_EASE_ALPHA * (0.0 - self._smoothed_face_offset[1])
                servo_angles[FACE_PAN_SERVO] = int(90 - self._smoothed_face_offset[0] * FACE_PAN_GAIN)
                servo_angles[FACE_TILT_SERVO] = int(90 + self._smoothed_face_offset[1] * FACE_TILT_GAIN)
            # иначе голова уже практически по центру — больше ничего не шлём,
            # чтобы не досылать бесконечные микро-поправки к идеальному нулю
        # (следующий раз, когда диалог начнётся заново, is_reorient сработает
        # и голова сразу прицелится в центр лица, прервав "оседание" при необходимости)

        self._was_dialog_active = dialog_active
        self._last_target_track_id = target_track_id
        self._last_face_detected = face_detected
        self._last_face_bbox = vision_result.get("face_bbox")
        self._last_faces_count = vision_result.get("faces_count", 0)

        return {
            "servo_angles": servo_angles,
            "face_offset": raw_offset,
            "description": self.vision_context,
            "objects": vision_result.get("objects", []),
            "face_detected": face_detected,
            "face_bbox": vision_result.get("face_bbox"),
            "faces_count": vision_result.get("faces_count", 0),
            "target_track_id": target_track_id,
            "dialog_active": dialog_active,
        }

    def get_panel_annotation_status(self) -> dict:
        """Текущее состояние лица/диалога для отрисовки в панели мониторинга —
        обновляется на каждом обработанном кадре детекции (VIDE)"""
        return {
            "face_detected": self._last_face_detected,
            "face_bbox": self._last_face_bbox,
            "faces_count": self._last_faces_count,
            "dialog_active": self._was_dialog_active,
        }

    async def render_panel_jpeg(self, quality: int) -> Optional[bytes]:
        """Рендерит текущий кадр для панели (с оверлеем лица) в фоновом CV-потоке"""
        loop = asyncio.get_event_loop()
        status = self.get_panel_annotation_status()
        tracking_active = bool(status["dialog_active"] and status["face_detected"])
        return await loop.run_in_executor(
            self._cv_executor, self.vision.get_annotated_jpeg, tracking_active, quality
        )

    async def update_panel_frame(self, frame_bytes: bytes) -> bool:
        """Принимает кадр повышенного качества ТОЛЬКО для панели (тег VIDP от прошивки,
        см. пункт про двойное разрешение) — выполняется в фоновом CV-потоке"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._cv_executor, self.vision.update_panel_frame, frame_bytes)

    async def handle_command(self, command: dict) -> dict:
        cmd_type = command.get("type")

        if cmd_type == "servo":
            self.servos.set_servo(command["id"], command["angle"])
            return {"status": "ok", "servo": command["id"], "angle": command["angle"]}

        elif cmd_type == "servo_multi":
            self.servos.set_all_servos(command["angles"])
            return {"status": "ok", "angles": command["angles"]}

        elif cmd_type == "animation":
            asyncio.create_task(self.servos.play_animation(command["name"], on_frame=self.on_servo_frame))
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

            # Тот же будильник "Сорен", что и в голосовом пути — для
            # единообразия он нужен и в текстовых командах с панели.
            command_text = strip_wake_word(user_text)
            if command_text is None:
                if self.is_wake_session_active():
                    command_text = user_text
                else:
                    return {
                        "status": "ignored",
                        "message": "Команда проигнорирована: нет будильника 'Сорен'",
                        "text": user_text,
                    }

            self.mark_dialog_active()
            self.is_processing = True
            try:
                loop = asyncio.get_event_loop()
                # LLM.generate + TTS.synthesize — те же тяжёлые блокирующие вызовы, что и в
                # голосовом пути (_process_speech), тем же приёмом уводим их в фоновый поток.
                result = await loop.run_in_executor(self._speech_executor, self._handle_text_command_sync, command_text)

                if result.get("action"):
                    asyncio.create_task(self.servos.play_animation(result["action"], on_frame=self.on_servo_frame))
                else:
                    # Явно доставляем эмоциональную позу на ESP32 (из потока коллбэк не ушёл)
                    await self._notify_servo_frame()

                return {
                    "status": "ok",
                    "text": command_text,
                    "raw_text": command.get("text", ""),
                    "response": result["response"],
                    "audio": result["audio"].hex() if result["audio"] else "",
                    "action": result.get("action"),
                    "emotion": result["emotion"],
                    "servo_angles": result["servo_angles"],
                    "eye_led": result["eye_led"]
                }
            finally:
                self.is_processing = False

        elif cmd_type == "get_status":
            return {
                "status": "ok",
                "servo_angles": self.servos.get_current_angles(),
                "processing": self.is_processing,
                "vision_context": self.vision_context,
                "current_emotion": self.current_emotion,
                "modes": self.get_modes(),
                "memory": self.memory.short_term.get_summary() if self.memory else {},
                "ltm_enabled": self.memory.long_term.enabled if self.memory else False,
                "memory_flags": self.get_memory_flags(),
                "model_config": self.get_model_config(),
                "quick_answers": self.get_quick_answers_status()
            }

        elif cmd_type == "clear_history":
            self.llm.clear_history()
            if self.memory:
                self.memory.clear()
            self.current_emotion = "calm"
            return {"status": "ok", "message": "История и память очищены"}

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

        elif cmd_type == "get_memory_config":
            return {"status": "ok", "memory_flags": self.get_memory_flags()}

        elif cmd_type == "set_memory_config":
            level = command.get("level")
            enabled = command.get("enabled")
            if level not in ("stm", "ltm", "profile", "rag"):
                return {"status": "error", "message": f"Неизвестный уровень памяти: {level}"}
            flags = self.set_memory_flag(level, bool(enabled))
            return {"status": "ok", "level": level, "enabled": bool(enabled), "memory_flags": flags}

        elif cmd_type == "get_model_config":
            return {"status": "ok", "model_config": self.get_model_config()}

        elif cmd_type == "set_llm_model":
            model_id = command.get("model_id")
            ok = self.set_llm_model(model_id)
            if not ok:
                return {"status": "error", "message": f"Неизвестная модель LLM: {model_id}"}
            return {"status": "ok", "model_config": self.get_model_config()}

        elif cmd_type == "set_stt_model":
            model_id = command.get("model_id")
            if not model_id:
                return {"status": "error", "message": "Не указан model_id"}
            self.set_stt_model(model_id)
            return {"status": "ok", "model_config": self.get_model_config()}

        elif cmd_type == "set_tts_speaker":
            speaker = command.get("speaker")
            if not speaker:
                return {"status": "error", "message": "Не указан speaker"}
            self.set_tts_speaker(speaker)
            return {"status": "ok", "model_config": self.get_model_config()}

        elif cmd_type == "reload_quick_answers":
            return {"status": "ok", "quick_answers": self.reload_quick_answers()}

        else:
            return {"status": "error", "message": f"Неизвестная команда: {cmd_type}"}

    async def _notify_servo_frame(self):
        """Явно рассылает текущие углы серво на физическое устройство.
        Используется после возврата из фонового потока _speech_executor,
        где asyncio event loop недоступен и _notify_servo_frame внутри
        ServoController молча проглатывается."""
        if self.on_servo_frame:
            await self.on_servo_frame(self.servos.get_current_angles())

    def get_last_tts_audio(self) -> tuple:
        """Возвращает (utterance_id, audio_bytes) последней озвученной фразы —
        для повторной отправки по запросу ESP32 после обрыва Wi-Fi
        (audio_resume_request). utterance_id=0 значит, что ещё ничего не
        озвучивалось."""
        return self.last_tts_utterance_id, self.last_tts_audio

    def shutdown(self):
        logger.info("Завершение работы RobotBrain...")
        if self.memory:
            self.memory.save_profile()
        self.vision.release()
        self._cv_executor.shutdown(wait=False)
        self._speech_executor.shutdown(wait=False)