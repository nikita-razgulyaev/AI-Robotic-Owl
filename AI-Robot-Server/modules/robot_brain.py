"""Главный мозг робота — оркестратор с эмоциональным движком и памятью"""
import asyncio
import logging
import json
import time
import wave
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
from modules.semantic_quick_match import semantic_quick_match
from modules.wake_word import strip_wake_word
from modules.animation_loader import animation_book
from modules.arbiter import arbiter, Priority
from modules import motion_primitives
from modules.locomotion_mode import locomotion_mode
from modules.wing_flap import WingFlapController
from modules.behavior import behavior_engine
from modules.chat_log import chat_log
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
    IDLE_RETURN_TIMEOUT_SEC,
)

# --- safe imports для новых флагов инверсии (обратная совместимость) ---
try:
    from config.settings import FACE_PAN_INVERT
except ImportError:
    FACE_PAN_INVERT = False
try:
    from config.settings import FACE_TILT_INVERT
except ImportError:
    FACE_TILT_INVERT = False

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
            return [90] * 22

        angles = []
        for i in range(22):
            key = f"S{i}"
            angles.append(pose.get(key, 90))
        return angles


class RobotBrain:
    """Основной контроллер — соединяет все модули с памятью"""

    def __init__(self):
        logger.info("=== Инициализация RobotBrain (Сорен + Vector RAG) ===")

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

        self.last_tts_audio: bytes = b""
        self.last_tts_utterance_id: int = 0

        self.on_servo_frame: Optional[Callable[[List[int]], Awaitable[None]]] = None
        self.wing_flap = WingFlapController(self.servos, lambda: self.on_servo_frame)

        # === Фоновые потоки для тяжёлых синхронных операций ===
        self._cv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cv-worker")
        self._speech_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech-worker")

        # === Состояние "активного диалога" (для слежения за лицом) ===
        self.last_interaction_ts = 0.0
        self.provisional_interaction_ts = 0.0

        # === Сглаживание слежения за лицом + "мёртвая зона" ===
        self._smoothed_face_offset = [0.0, 0.0]
        self._last_target_track_id = None
        self._was_dialog_active = False

        self._last_face_detected = False
        self._last_face_bbox = None
        self._last_faces_count = 0

        self._idle_pose_applied = True
        self._dialog_inactive_since: Optional[float] = None
        self._watchdog_task: Optional[asyncio.Task] = None

        logger.info("=== RobotBrain (Сорен + Vector RAG) готов ===")

    def start_background_tasks(self):
        """Запускает фоновые asyncio-задачи, которым нужен работающий event
        loop — вызывается из startup() в websocket_server.py, а не из __init__
        (на момент создания RobotBrain loop ещё может быть не запущен)."""
        if self._watchdog_task is None:
            self._watchdog_task = asyncio.create_task(self._idle_watchdog_loop())

    async def _idle_watchdog_loop(self):
        """Раз в секунду проверяет: если диалог неактивен дольше
        IDLE_RETURN_TIMEOUT_SEC, ничего не анимируется и не обрабатывается,
        а сервы всё ещё не в позе "calm" — плавно возвращает их туда и
        рассылает на устройство. Срабатывает один раз за переход в простой
        (флаг _idle_pose_applied), а не на каждом тике."""
        while True:
            try:
                await asyncio.sleep(1.0)

                if self.is_dialog_active() or self.is_processing or self.servos.is_animating:
                    self._dialog_inactive_since = None
                    continue

                now = time.time()
                if self._dialog_inactive_since is None:
                    self._dialog_inactive_since = now

                if self._idle_pose_applied:
                    continue

                if (now - self._dialog_inactive_since) < IDLE_RETURN_TIMEOUT_SEC:
                    continue

                calm_angles = self.emotion_engine.get_servo_angles("calm")
                if self.servos.get_current_angles() == calm_angles:
                    self._idle_pose_applied = True
                    continue

                logger.info(
                    f"😴 Простой {IDLE_RETURN_TIMEOUT_SEC:.0f}с+ — возвращаю сервы в позу покоя"
                )
                self.current_emotion = "calm"
                self.servos.set_all_servos(calm_angles, notify=False)
                await self._notify_servo_frame()
                self._idle_pose_applied = True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Ошибка в idle watchdog: {e}")

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

    async def emergency_stop(self, reason: str = "unknown") -> Dict:
        """Пункт 3: реальный путь SAFETY. Бронирует наивысший приоритет
        арбитра (перебивает вообще всё, включая панель/геймпад) и переводит
        сервы в нейтральную позу, отменяя всё, что сейчас играет.
        reason — для логов/метрик (arbiter.stats() покажет, что и как часто
        реально триггерило safety-остановку)."""
        await arbiter.request(Priority.SAFETY, duration_s=3.0, label=f"safety:{reason}")
        await self.servos.emergency_stop(on_frame=self.on_servo_frame)
        logger.warning(f"🛑 Emergency stop: {reason}")
        return {"status": "ok", "reason": reason}

    async def return_to_default(self, reason: str = "panel") -> Dict:
        """Плавно возвращает все сервы в 90° — тот же путь, что при подключении
        платы (_glide_all_servos_to_default) и при safety-остановке, но без
        мгновенного «телепорта»: кадры реально уходят на ESP32 и в панель."""
        await arbiter.request(Priority.PANEL, duration_s=2.0, label=f"return_to_default:{reason}")
        # Если идёт анимация — сначала отменяем её, иначе smooth_return
        # интерполирует от «логического» current_angles, а не от реальной позы.
        if self.servos.is_animating:
            await self.servos.cancel_current()
        await self.servos.smooth_return_to_default(on_frame=self.on_servo_frame)
        logger.info(f"↩️ Сервы плавно возвращены в 90° ({reason})")
        return {"status": "ok", "angles": self.servos.get_current_angles()}

    def get_quick_answers_status(self) -> Dict:
        return {"enabled": QUICK_ANSWERS_ENABLED, "count": quick_answers.count()}

    def reload_quick_answers(self) -> Dict:
        """Перечитывает character/quick_answers.json с диска — без рестарта сервера"""
        count = quick_answers.reload()
        semantic_quick_match.build_index()
        return {"enabled": QUICK_ANSWERS_ENABLED, "count": count}

    async def process_audio_chunk(self, pcm_bytes: bytes) -> Optional[dict]:
        status, audio = self.audio_buffer.process_chunk(pcm_bytes)

        if status == "speech_start":
            self.mark_dialog_provisional()
        elif status == "speech":
            self.mark_dialog_provisional()
        elif status == "complete" and audio:
            return await self._process_speech(audio)

        return None

    async def _process_speech(self, audio_bytes: bytes) -> dict:
        self.is_processing = True
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._speech_executor, self._process_speech_sync, audio_bytes)

            if getattr(self, "_wake_word_heard", False):
                self._wake_word_heard = False
                await arbiter.request(Priority.ENVIRONMENT, duration_s=1.0, label="wake_word")

            action = result.get("action")
            if action:
                asyncio.create_task(self.servos.play_animation(action, on_frame=self.on_servo_frame))
            elif result.get("servo_changed"):
                # Явно доставляем эмоциональную позу на ESP32 (из потока коллбэк не ушёл).
                # Пропускается, когда распознавание ничего не дало (servo_changed=False) —
                # иначе на каждый ложный VAD-триггер сервер заново рассылал бы одну и ту же
                # (устаревшую) позу устройству, создавая впечатление "зависания".
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
            logger.info(f"👂 Будильник распознан: '{user_text}' -> '{command_text or '(пусто)'}'")
            self._wake_word_heard = True

        logger.info(f"Пользователь: {command_text or '(только будильник, без команды)'}")
        self.mark_dialog_active()
        chat_log.emit("user", command_text or user_text, source="robot")

        llm_result = self.generate_reply(command_text)

        response_text = llm_result["text"]
        action = llm_result.get("action")
        emotion = llm_result.get("emotion", "calm")
        self.current_emotion = emotion

        logger.info(f"Сорен: {response_text}")
        logger.info(f"Эмоция: {emotion}")
        if action:
            logger.info(f"Действие: {action}")
        chat_log.emit("bot", response_text, emotion=emotion, source="robot")

        servo_angles = self.emotion_engine.get_servo_angles(emotion)
        eye_led = self.emotion_engine.get_eye_led(emotion)

        logger.info("Синтез речи...")
        tts_audio = self._synthesize_or_cached(response_text, llm_result.get("audio"))

        if tts_audio:
            self.last_tts_audio = tts_audio
            self.last_tts_utterance_id += 1

            estimated_playback_sec = len(tts_audio) / (48000 * 2)
            self.last_interaction_ts = time.time() + estimated_playback_sec

        servo_changed = False
        if not action:
            self.servos.set_all_servos(servo_angles, notify=False)
            servo_changed = True

        self._idle_pose_applied = (emotion == "calm")
        self._dialog_inactive_since = None

        return {
            "text": user_text,
            "raw_text": raw_text,
            "response": response_text,
            "audio": tts_audio,
            "action": action,
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led,
            "servo_changed": servo_changed,
        }

    def _load_cached_audio(self, rel_path: Optional[str]) -> Optional[bytes]:
        """Читает заранее сгенерированный WAV (character/<rel_path>, см.
        scripts/build_audio_cache.py) и возвращает сырой PCM 16-bit mono
        48kHz — тот же формат, что отдаёт self.tts.synthesize(). Модуль
        wave сам отбрасывает заголовок WAV при чтении, поэтому конвертация
        не нужна — файл просто должен быть в этом формате (ровно то, что
        пишет TTSEngine.synthesize_to_wav())."""
        if not rel_path:
            return None
        if not isinstance(rel_path, str) or len(rel_path) > 260:
            return None
        path = CHARACTER_DIR / rel_path
        if not path.exists():
            logger.warning(f"[audio_cache] Файл не найден: {path} — использую live TTS")
            return None
        try:
            with wave.open(str(path), "rb") as wf:
                if wf.getframerate() != 48000 or wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                    logger.warning(f"[audio_cache] {path} не в формате 16-bit/mono/48kHz — использую live TTS")
                    return None
                return wf.readframes(wf.getnframes())
        except Exception as e:
            logger.error(f"[audio_cache] Ошибка чтения {path}: {e} — использую live TTS")
            return None

    def _synthesize_or_cached(self, text: str, cached_rel_path: Optional[str]) -> Optional[bytes]:
        """Единая точка получения аудио ответа: сначала пробуем офлайн-кэш
        (быстро, TTS не трогается), при любой неудаче — падаем на живой TTS,
        чтобы отсутствие/поломка файла кэша никогда не оставляла Сорена немым."""
        cached = self._load_cached_audio(cached_rel_path)
        if cached is not None:
            return cached
        audio = self.tts.synthesize(text)
        if not audio:
            logger.warning(f"[TTS] Пустое аудио для фразы {text[:60]!r} — на динамик ничего не уйдёт (см. ошибки TTS выше)")
        return audio

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
            qa = semantic_quick_match.find(user_text)
        if qa is None:
            return None

        variant = quick_answers.pick_response(qa)
        logger.info(f"⚡ Быстрый ответ '{qa.id}' (без LLM): {user_text!r} -> {variant.text!r}")

        if qa.id == "control_mode_manual":
            from modules.control_mode import control_mode
            control_mode.set("manual", reason="voice")
        elif qa.id == "control_mode_autonomous":
            from modules.control_mode import control_mode
            control_mode.set("autonomous", reason="voice")

        return {
            "text": variant.text,
            "action": variant.action,
            "emotion": variant.emotion or "calm",
            "audio": variant.audio,
            "sfx": variant.sfx,
        }

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

        stm = self.memory.short_term.get_context() if flags.get("stm", True) else ""

        parts = []
        if stm:
            parts.append(f"Последние реплики:\n{stm}")

        if flags.get("ltm", True):
            ltm = self.memory.get_relevant_memories(user_text)
            if ltm:
                parts.append(f"Релевантные воспоминания:\n{ltm}")

        if flags.get("profile", True):
            dom = max(self.memory.profile.dominant_emotions, key=self.memory.profile.dominant_emotions.get)
            parts.append(f"Доминантная эмоция пользователя: {dom}")

            greeting = self.memory.get_day_greeting()
            if greeting and not stm:
                parts.append(f"Приветствие: {greeting}")

        return "\n\n".join(parts)

    def _handle_text_command_sync(self, user_text: str) -> dict:
        """Синхронное тело обработки текстовой команды (LLM+TTS) — выполняется в фоновом потоке"""
        llm_result = self.generate_reply(user_text)

        emotion = llm_result.get("emotion", "calm")
        bot_chat_id = chat_log.emit("bot", llm_result["text"], emotion=emotion, source="panel")
        self._last_bot_chat_id = bot_chat_id
        servo_angles = self.emotion_engine.get_servo_angles(emotion)
        eye_led = self.emotion_engine.get_eye_led(emotion)
        tts_audio = self._synthesize_or_cached(llm_result["text"], llm_result.get("audio"))

        if tts_audio:
            self.last_tts_audio = tts_audio
            self.last_tts_utterance_id += 1

        servo_changed = False
        if not llm_result.get("action"):
            self.servos.set_all_servos(servo_angles, notify=False)
            servo_changed = True

        self._idle_pose_applied = (emotion == "calm")
        self._dialog_inactive_since = None

        return {
            "response": llm_result["text"],
            "audio": tts_audio,
            "action": llm_result.get("action"),
            "emotion": emotion,
            "servo_angles": servo_angles,
            "eye_led": eye_led,
            "servo_changed": servo_changed,
        }

    def _build_empty_response(self) -> dict:
        return {
            "text": "",
            "raw_text": "",
            "response": "",
            "audio": b"",
            "action": None,
            "emotion": "calm",
            "servo_angles": [90] * 22,
            "eye_led": "soft_white_low",
            "servo_changed": False,
        }

    async def process_video_frame(self, frame_bytes: bytes) -> dict:
        if self.is_dialog_active():
            now = time.monotonic()
            if now - getattr(self, "_last_env_reserve", 0.0) > 0.8:
                self._last_env_reserve = now
                await arbiter.request(Priority.ENVIRONMENT, duration_s=1.5, label="face_tracking")

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._cv_executor, self._process_video_frame_sync, frame_bytes)

    def _process_video_frame_sync(self, frame_bytes: bytes) -> dict:
        """Синхронное тело обработки видеокадра — выполняется в фоновом потоке"""
        vision_result = self.vision.process_frame(frame_bytes)
        self.vision_context = vision_result.get("description", "")

        face_detected = vision_result.get("face_detected", False)
        target_track_id = vision_result.get("target_track_id")
        raw_offset = self.vision.get_face_offset()

        dialog_active = FACE_TRACKING_ENABLED and self.is_dialog_active()

        servo_angles = self.servos.get_current_angles()

        if dialog_active and face_detected:
            is_reorient = (not self._was_dialog_active) or (target_track_id != self._last_target_track_id)

            if is_reorient:
                self._smoothed_face_offset = [raw_offset[0], raw_offset[1]]
            else:
                in_deadzone = (
                    abs(raw_offset[0]) < FACE_DEADZONE_X and
                    abs(raw_offset[1]) < FACE_DEADZONE_Y
                )
                if not in_deadzone:
                    self._smoothed_face_offset[0] += HEAD_SMOOTHING_ALPHA * (raw_offset[0] - self._smoothed_face_offset[0])
                    self._smoothed_face_offset[1] += HEAD_SMOOTHING_ALPHA * (raw_offset[1] - self._smoothed_face_offset[1])

            pan_sign = 1 if FACE_PAN_INVERT else -1
            tilt_sign = 1 if FACE_TILT_INVERT else -1

            servo_angles[FACE_PAN_SERVO] = max(0, min(180, int(90 + pan_sign * self._smoothed_face_offset[0] * FACE_PAN_GAIN)))
            servo_angles[FACE_TILT_SERVO] = max(0, min(180, int(90 + tilt_sign * self._smoothed_face_offset[1] * FACE_TILT_GAIN)))

        elif dialog_active and not face_detected:
            pass

        else:
            still_offset = (
                abs(self._smoothed_face_offset[0]) > EXIT_EASE_EPSILON or
                abs(self._smoothed_face_offset[1]) > EXIT_EASE_EPSILON
            )
            if still_offset:
                pan_sign = 1 if FACE_PAN_INVERT else -1
                tilt_sign = 1 if FACE_TILT_INVERT else -1

                self._smoothed_face_offset[0] += EXIT_EASE_ALPHA * (0.0 - self._smoothed_face_offset[0])
                self._smoothed_face_offset[1] += EXIT_EASE_ALPHA * (0.0 - self._smoothed_face_offset[1])
                servo_angles[FACE_PAN_SERVO] = max(0, min(180, int(90 + pan_sign * self._smoothed_face_offset[0] * FACE_PAN_GAIN)))
                servo_angles[FACE_TILT_SERVO] = max(0, min(180, int(90 + tilt_sign * self._smoothed_face_offset[1] * FACE_TILT_GAIN)))
            # иначе голова уже практически по центру — больше ничего не шлём,
            # чтобы не досылать бесконечные микро-поправки к идеальному нулю
        # (следующий раз, когда диалог начнётся заново, is_reorient сработает
        # и голова сразу прицелится в центр лица, прервав "оседание" при необходимости)

        # ==================== КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ====================
        for i, angle in enumerate(servo_angles):
            self.servos.current_angles[i] = angle
            self.servos.target_angles[i] = angle

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

    async def _ensure_wing_baseline(self, mode: str, priority: Priority = Priority.GAMEPAD) -> Optional[bool]:
        """Плавно выводит сервы взмаха в базовую позу режима (walk → 0°, flight → 90°).

        Возвращает: None — крылья уже на месте, ничего не делали; True — переход
        выполнен; False — сервы заняты, переход не начат."""
        deg = motion_primitives.wing_baseline_deg(mode)
        if motion_primitives.wings_at(self.servos.get_current_angles(), deg):
            return None
        await self.wing_flap.stop()
        if self.servos.is_animating:
            return False
        frames = motion_primitives.build_wing_transition(self.servos.get_current_angles(), deg)
        await arbiter.request(priority, frames[-1]["time"] / 1000.0, label=f"wings:{mode}")
        return await self.servos.play_motion(frames, on_frame=self.on_servo_frame)

    @staticmethod
    def _gamepad_blocked(command: dict) -> bool:
        from modules.control_mode import control_mode
        return control_mode.is_autonomous() and str(command.get("source", "gamepad")).lower() == "gamepad"

    async def handle_command(self, command: dict) -> dict:
        cmd_type = command.get("type")

        if cmd_type == "emergency_stop":
            return await self.emergency_stop(reason=command.get("reason", "manual"))

        if cmd_type == "return_to_default":
            return await self.return_to_default(reason=command.get("reason", "panel"))

        if cmd_type == "set_control_mode":
            from modules.control_mode import control_mode
            return control_mode.set(
                command.get("mode", "manual"),
                reason=command.get("reason", "command"),
            )

        if cmd_type == "get_control_mode":
            from modules.control_mode import control_mode
            return {"status": "ok", **control_mode.to_dict()}

        if cmd_type == "get_locomotion_mode":
            return {"status": "ok", **locomotion_mode.to_dict()}

        if cmd_type == "set_locomotion_mode":
            if self._gamepad_blocked(command):
                return {"status": "rejected", "message": "Режим autonomous — геймпад отключён."}
            result = locomotion_mode.set(command.get("mode", ""), reason=command.get("reason", command.get("source", "command")))
            if result["status"] != "ok":
                return result
            priority = Priority[str(command.get("source", "GAMEPAD")).upper()]
            moved = await self._ensure_wing_baseline(result["mode"], priority)
            result["wings"] = "already" if moved is None else ("ok" if moved else "busy")
            return result

        if cmd_type == "flap":
            # Непрерывный взмах: {type:"flap", throttle: 0..1} шлют каждый тик, пока стик отклонён.
            if self._gamepad_blocked(command):
                return {"status": "rejected", "message": "Режим autonomous — геймпад отключён."}
            if not locomotion_mode.is_flight():
                return {"status": "rejected", "message": "Взмах доступен только в режиме полёта"}
            throttle = float(command.get("throttle", 0.0))
            if throttle > 0.0 and not self.wing_flap.running:
                if self.servos.is_animating:
                    return {"status": "rejected", "message": "Сервы заняты другой анимацией/примитивом"}
                moved = await self._ensure_wing_baseline("flight")
                if moved is False:
                    return {"status": "rejected", "message": "Сервы заняты другой анимацией/примитивом"}
                if moved:
                    return {"status": "ok", "message": "Крылья выведены в нейтраль 90°, взмах начнётся со следующего тика"}
            accepted = self.wing_flap.set_throttle(throttle)
            return {"status": "ok" if accepted else "rejected", "flap_hz": round(self.wing_flap.frequency_hz, 2)}

        if cmd_type in ("servo", "servo_multi", "animation"):
            await arbiter.request(Priority.PANEL, duration_s=2.0, label=f"panel:{cmd_type}")

        if cmd_type == "servo":
            self.servos.set_servo(command["id"], command["angle"])
            return {"status": "ok", "servo": command["id"], "angle": command["angle"]}

        elif cmd_type == "servo_multi":
            self.servos.set_all_servos(command["angles"])
            return {"status": "ok", "angles": command["angles"]}

        elif cmd_type == "animation":
            if command["name"] not in animation_book:
                return {"status": "error", "message": f"Анимация не найдена: {command['name']}"}
            asyncio.create_task(self.servos.play_animation(command["name"], on_frame=self.on_servo_frame))
            return {"status": "ok", "animation": command["name"]}

        elif cmd_type == "list_animations":
            return {"status": "ok", "animations": animation_book.list_info()}

        elif cmd_type == "reload_animations":
            count = animation_book.reload()
            return {"status": "ok", "animations": animation_book.list_info(), "count": count}

        elif cmd_type in ("walk", "fly", "turn", "takeoff", "land"):
            from modules.control_mode import control_mode
            src_name = str(command.get("source", "gamepad")).lower()
            if control_mode.is_autonomous() and src_name == "gamepad":
                return {
                    "status": "rejected",
                    "message": "Режим autonomous — геймпад отключён. Переключите на «Ручное» или управляйте с панели.",
                }
            source = Priority[command.get("source", "GAMEPAD").upper()]

            if cmd_type == "takeoff":
                takeoff_ok, takeoff_reason = behavior_engine.can_takeoff()
                if not takeoff_ok:
                    return {"status": "rejected", "message": takeoff_reason}

            if cmd_type in ("walk", "turn"):
                # Режим локомоции задаёт позу крыльев: ходьба — 0°, полёт — 90°.
                # Сначала плавно выводим крылья, потом играем шаги с зафиксированными крыльями.
                if self.wing_flap.running:
                    return {"status": "rejected", "message": "Идёт взмах — отпустите стик вперёд"}
                wing_deg = motion_primitives.wing_baseline_deg(locomotion_mode.mode)
                if await self._ensure_wing_baseline(locomotion_mode.mode, source) is False:
                    return {"status": "rejected", "message": "Сервы заняты другой анимацией/примитивом"}

            if cmd_type == "walk":
                frames = motion_primitives.apply_wing_baseline(
                    motion_primitives.build_walk(command.get("distance_cm", 0.0), steps=command.get("steps")), wing_deg)
            elif cmd_type == "fly":
                frames = motion_primitives.build_flight_forward(command["distance_cm"])
            elif cmd_type == "turn":
                frames = motion_primitives.apply_wing_baseline(
                    motion_primitives.build_turn(command["angle_deg"]), wing_deg)
            elif cmd_type == "takeoff":
                frames = motion_primitives.build_takeoff()
            else:
                frames = motion_primitives.build_landing()

            if not frames:
                return {"status": "ok", "message": "Нулевая дистанция/угол — команда проигнорирована"}

            duration_s = frames[-1]["time"] / 1000.0
            accepted = await arbiter.request(source, duration_s, label=f"motion:{cmd_type}")
            if not accepted:
                return {"status": "rejected", "message": f"Арбитр занят приоритетнее {source.name}"}

            if self.servos.is_animating:
                return {"status": "rejected", "message": "Сервы заняты другой анимацией/примитивом"}
            asyncio.create_task(self.servos.play_motion(frames, on_frame=self.on_servo_frame))
            if cmd_type == "takeoff":
                behavior_engine.begin_self_flight()

            result = {"status": "ok", "motion": cmd_type}
            qa = quick_answers.get_by_id(f"motion_{cmd_type}")
            if qa:
                variant = quick_answers.pick_response(qa)
                result["sfx"] = variant.sfx
                result["audio"] = variant.audio
            return result

        elif cmd_type == "text":
            try:
                from modules.fuzzy_matcher import correct_speech_text
                raw_text = command["text"]
                corrected_text = correct_speech_text(raw_text)
                if corrected_text != raw_text:
                    logger.info(f"🎯 Fuzzy (text): '{raw_text}' -> '{corrected_text}'")
                user_text = corrected_text
            except ImportError:
                user_text = command["text"]

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
            user_chat_id = await chat_log.emit_async("user", command_text or user_text, source="panel",
                                                    chat_id=command.get("client_id"))
            self.is_processing = True
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self._speech_executor, self._handle_text_command_sync, command_text)

                if result.get("action"):
                    asyncio.create_task(self.servos.play_animation(result["action"], on_frame=self.on_servo_frame))
                else:
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
                    "eye_led": result["eye_led"],
                    "user_chat_id": user_chat_id,
                    "bot_chat_id": getattr(self, "_last_bot_chat_id", None),
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
        if self._watchdog_task:
            self._watchdog_task.cancel()
        if self.memory:
            self.memory.save_profile()
        self.vision.release()
        self._cv_executor.shutdown(wait=False)
        self._speech_executor.shutdown(wait=False)