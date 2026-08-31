"""Speech-to-Text модуль - Whisper (local) + Fuzzy matching"""
import logging
import tempfile
import os
from pathlib import Path
from config.settings import (
    STT_MODE, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_MODELS
)

logger = logging.getLogger(__name__)

# Порог "вероятности отсутствия речи" от самого Whisper (усреднённый по сегментам
# no_speech_prob). Выше этого — модель сама не уверена, что там вообще была речь,
# такое почти всегда шум/тишина, даже если что-то "распозналось".
NO_SPEECH_PROB_THRESHOLD = 0.6

# Известные галлюцинации faster-whisper на тишине/шуме для русского языка —
# модель обучена в том числе на субтитрах с YouTube и на пустом/шумном входе
# вместо пустой строки часто выдаёт титровые фразы вроде этих. Точное или
# частичное совпадение → результат отбрасывается как "речь не распознана".
HALLUCINATION_PATTERNS = [
    "редактор субтитров",
    "корректор",
    "субтитры сделал",
    "субтитры создавал",
    "продолжение следует",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "ставьте лайк",
]


def _looks_like_hallucination(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in HALLUCINATION_PATTERNS)


class LocalSTTEngine:
    """Локальное распознавание речи через faster-whisper"""

    def __init__(self, model_size: str = None):
        self.model_size = model_size or WHISPER_MODEL_SIZE
        self.model = None
        self._load_model()
        # Lazy import fuzzy matcher
        try:
            from modules.fuzzy_matcher import correct_speech_text
            self.correct_text = correct_speech_text
            self.fuzzy_available = True
            logger.info("Fuzzy matching загружен")
        except ImportError:
            self.correct_text = lambda x: x
            self.fuzzy_available = False
            logger.warning("Fuzzy matcher не найден, работаем без адаптивной коррекции")

    def _load_model(self):
        """Загружает Whisper модель"""
        logger.info(f"Загрузка Whisper: {self.model_size} ({WHISPER_DEVICE})")
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_size,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE
            )
            logger.info("Whisper загружен")
        except Exception as e:
            logger.error(f"Ошибка загрузки Whisper: {e}")
            self.model = None

    def transcribe(self, audio_bytes: bytes) -> dict:
        if self.model is None:
            return {"success": False, "text": "", "corrected_text": "", "error": "Модель не загружена"}

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import wave
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_bytes)
                tmp_path = tmp.name

            result = self.transcribe_from_file(tmp_path)
            os.unlink(tmp_path)
            return result

        except Exception as e:
            logger.error(f"Ошибка STT: {e}")
            return {"success": False, "text": "", "corrected_text": "", "error": str(e)}

    def transcribe_from_file(self, audio_path: str) -> dict:
        if self.model is None:
            return {"success": False, "text": "", "corrected_text": "", "error": "Модель не загружена"}

        try:
            segments, info = self.model.transcribe(audio_path, language="ru", beam_size=5)
            segments = list(segments)  # generator -> список, нужен для двух проходов
            text = " ".join(segment.text for segment in segments).strip()

            if segments:
                avg_no_speech_prob = sum(s.no_speech_prob for s in segments) / len(segments)
                if avg_no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
                    logger.info(
                        f"🔇 STT отклонён — no_speech_prob={avg_no_speech_prob:.2f} "
                        f"(похоже на шум/тишину): '{text}'"
                    )
                    return {"success": False, "text": "", "corrected_text": "", "error": "Речь не распознана"}

            if text and _looks_like_hallucination(text):
                logger.info(f"🔇 STT отклонён — похоже на галлюцинацию Whisper: '{text}'")
                return {"success": False, "text": "", "corrected_text": "", "error": "Речь не распознана"}

            if text:
                corrected_text = self.correct_text(text) if self.fuzzy_available else text

                if corrected_text != text:
                    logger.info(f"🎤 STT: '{text}' → '{corrected_text}'")
                else:
                    logger.info(f"🎤 STT: '{text}'")

                return {
                    "success": True,
                    "text": text,
                    "corrected_text": corrected_text,
                    "error": None
                }
            else:
                return {"success": False, "text": "", "corrected_text": "", "error": "Речь не распознана"}

        except Exception as e:
            logger.error(f"Ошибка STT из файла: {e}")
            return {"success": False, "text": "", "corrected_text": "", "error": str(e)}


class STTEngine:
    """STT движок — только локальный (faster-whisper)"""

    def __init__(self):
        self.mode = STT_MODE
        self.local_engine = LocalSTTEngine()
        self.model_id = self.local_engine.model_size
        logger.info("💻 STT режим: ЛОКАЛЬНЫЙ (faster-whisper)")

    def set_mode(self, mode: str):
        """STT только локальный — облачные режимы недоступны без VPN"""
        logger.warning("STT доступен только в локальном режиме (faster-whisper). Облачные STT API недоступны без VPN.")
        self.mode = "local"

    def get_mode(self) -> str:
        return "local"

    def get_current_model_id(self) -> str:
        return self.model_id

    def set_model(self, model_size: str):
        """Перезагружает faster-whisper с другим размером модели — сразу, без рестарта сервера"""
        logger.info(f"🎤 Перезагружаю Whisper: {self.model_id} → {model_size}…")
        self.local_engine = LocalSTTEngine(model_size)
        self.model_id = model_size

    def transcribe(self, audio_bytes: bytes) -> dict:
        return self.local_engine.transcribe(audio_bytes)

    def transcribe_from_file(self, audio_path: str) -> dict:
        return self.local_engine.transcribe_from_file(audio_path)