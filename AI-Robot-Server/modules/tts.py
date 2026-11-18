"""TTS модуль - синтез речи (Silero local / FreeTTS cloud via edge-tts)"""
import io
import wave
import logging
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from config.settings import (
    TTS_MODE, SAMPLE_RATE,
    FREETTS_VOICE, FREETTS_SPEED, SILERO_SPEAKER
)

logger = logging.getLogger(__name__)

_tts_async_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-async")


def _run_async(coro):
    """Синхронно выполняет coroutine, безопасно независимо от того, есть ли
    уже активный event loop в текущем потоке."""
    future = _tts_async_executor.submit(asyncio.run, coro)
    return future.result()


class FreeTTSEngine:
    """Бесплатный облачный синтез речи через edge-tts (Microsoft Edge TTS).
    Вызывает edge_tts напрямую из Python (без subprocess/CLI и без временных
    файлов на диске) — это убирает ~0.5-1 сек накладных расходов на каждый
    запуск отдельного процесса, что особенно заметно на слабом CPU."""

    def __init__(self):
        self.voice = FREETTS_VOICE
        self.speed = FREETTS_SPEED
        self._check_edge_tts()
        logger.info(f"☁️ FreeTTS (edge-tts) инициализирован: voice={self.voice}, speed={self.speed}")

    def _check_edge_tts(self):
        """Проверяет, установлен ли пакет edge-tts"""
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            logger.warning("Пакет edge-tts не найден. Установи: pip install edge-tts")

    def synthesize(self, text: str, speaker: str = None) -> bytes:
        """Синтезирует речь через edge_tts (Python API), возвращает PCM 16-bit 48kHz"""
        try:
            import edge_tts
            import soundfile as sf
            import librosa
            import numpy as np

            voice = self.voice
            speed_pct = int((self.speed - 1.0) * 100)
            rate_arg = f"+{speed_pct}%" if speed_pct >= 0 else f"{speed_pct}%"

            async def _stream_mp3() -> bytes:
                communicate = edge_tts.Communicate(text, voice, rate=rate_arg)
                buf = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buf.extend(chunk["data"])
                return bytes(buf)

            mp3_bytes = _run_async(_stream_mp3())
            if not mp3_bytes:
                logger.error("edge-tts вернул пустой ответ")
                return b""

            audio, sr = sf.read(io.BytesIO(mp3_bytes))

            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            if audio.size > 0:
                max_val = max(abs(audio).max(), 1e-9)
                audio = audio / max_val * 32767
            audio_int16 = audio.astype('int16')
            return audio_int16.tobytes()

        except ImportError as e:
            logger.error(f"Библиотека не установлена: {e}. Установи: pip install edge-tts soundfile librosa numpy")
            return b""
        except Exception as e:
            status = getattr(e, "status", None)
            if status == 403:
                logger.error("Ошибка FreeTTS (edge-tts): Microsoft отклонил запрос (HTTP 403). "
                             "Обычно лечится обновлением пакета: pip install -U edge-tts")
            elif status:
                logger.error(f"Ошибка FreeTTS (edge-tts): HTTP {status}")
            else:
                logger.error(f"Ошибка FreeTTS (edge-tts): {type(e).__name__}: {str(e)[:200]}")
            return b""

    def synthesize_to_wav(self, text: str, output_path: Path, speaker: str = None) -> Path:
        """Синтезирует в WAV файл (путь обязателен: папка кэша по умолчанию больше не создаётся)"""

        pcm_data = self.synthesize(text)
        if not pcm_data:
            return None

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm_data)

        return output_path

    def get_available_speakers(self) -> list:
        """Возвращает список доступных русских голосов edge-tts"""
        return [
            "ru-RU-SvetlanaNeural",
            "ru-RU-DmitryNeural",
            "ru-RU-EkaterinaNeural",
        ]


class LocalTTSEngine:
    """Локальный синтез речи через Silero (PyTorch)"""

    def __init__(self):
        self.device = None
        self.model = None
        self.sample_rate = 48000
        self.default_speaker = SILERO_SPEAKER
        self._load_model()

    def _load_model(self):
        """Загружает Silero TTS модель"""
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Silero TTS device: {self.device}")

            model_url = 'https://models.silero.ai/models/tts/ru/v4_ru.pt'
            model_path = Path("./models/silero_v4_ru.pt")
            model_path.parent.mkdir(parents=True, exist_ok=True)

            if not model_path.exists():
                logger.info(f"Скачивание Silero TTS модели...")
                torch.hub.download_url_to_file(model_url, model_path)
                logger.info("Silero модель скачана")

            self.model = torch.package.PackageImporter(model_path).load_pickle(
                "tts_models", "model"
            )
            self.model.to(self.device)

            if hasattr(self.model, 'sample_rate'):
                self.sample_rate = self.model.sample_rate

            logger.info(f"Silero TTS загружен, sample_rate={self.sample_rate}")

        except Exception as e:
            logger.error(f"Ошибка загрузки Silero: {e}")
            self.model = None

    def synthesize(self, text: str, speaker: str = None) -> bytes:
        try:
            if self.model is None:
                logger.error("Silero модель не загружена")
                return b""

            speaker = speaker or self.default_speaker

            import torch
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate
            )

            audio = audio.unsqueeze(0)
            audio_np = audio.squeeze().cpu().numpy()
            audio_np = audio_np / max(abs(audio_np)) * 32767
            audio_int16 = audio_np.astype('int16')
            return audio_int16.tobytes()

        except Exception as e:
            logger.error(f"Ошибка TTS: {e}")
            return b""

    def synthesize_to_wav(self, text: str, output_path: Path, speaker: str = None) -> Path:

        pcm_data = self.synthesize(text, speaker or self.default_speaker)
        if not pcm_data:
            return None

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)

        return output_path

    def get_available_speakers(self) -> list:
        return ['xenia', 'baya', 'kseniya', 'aidar']


class TTSEngine:
    """Универсальный TTS движок с переключением local/cloud"""

    def __init__(self):
        self.mode = TTS_MODE
        self.local_engine = None
        self.cloud_engine = None
        self.local_speaker = SILERO_SPEAKER
        self.cloud_speaker = FREETTS_VOICE
        self._cloud_down_until = 0.0

        if self.mode == "cloud":
            self.cloud_engine = FreeTTSEngine()
            logger.info("☁️ TTS режим: ОБЛАЧНЫЙ (FreeTTS / edge-tts)")
        else:
            self.local_engine = LocalTTSEngine()
            logger.info("💻 TTS режим: ЛОКАЛЬНЫЙ (Silero)")

    def set_mode(self, mode: str):
        """Переключает режим TTS"""
        if mode not in ["local", "cloud"]:
            logger.warning(f"Неверный режим TTS: {mode}. Используем 'local'")
            mode = "local"

        self.mode = mode
        if mode == "cloud":
            if self.cloud_engine is None:
                self.cloud_engine = FreeTTSEngine()
                self.cloud_engine.voice = self.cloud_speaker
            self.local_engine = None
            logger.info("☁️ TTS переключён на ОБЛАЧНЫЙ (FreeTTS)")
        else:
            if self.local_engine is None:
                self.local_engine = LocalTTSEngine()
                self.local_engine.default_speaker = self.local_speaker
            self.cloud_engine = None
            logger.info("💻 TTS переключён на ЛОКАЛЬНЫЙ")

    def get_mode(self) -> str:
        return self.mode

    def get_current_speaker(self) -> str:
        return self.cloud_speaker if self.mode == "cloud" else self.local_speaker

    def set_speaker(self, speaker: str):
        """Меняет голос для текущего режима — применяется сразу, без пересоздания движка"""
        if self.mode == "cloud":
            self.cloud_speaker = speaker
            if self.cloud_engine:
                self.cloud_engine.voice = speaker
            logger.info(f"☁️ Голос TTS (облако) → {speaker}")
        else:
            self.local_speaker = speaker
            if self.local_engine:
                self.local_engine.default_speaker = speaker
            logger.info(f"💻 Голос TTS (локально) → {speaker}")

    CLOUD_RETRY_PAUSE_SEC = 300  # сколько не трогать облако после неудачи

    def _cloud_available(self) -> bool:
        return time.monotonic() >= self._cloud_down_until

    def _mark_cloud_down(self):
        was_up = self._cloud_available()
        self._cloud_down_until = time.monotonic() + self.CLOUD_RETRY_PAUSE_SEC
        if was_up:
            logger.warning(f"Облачный TTS недоступен — следующие {self.CLOUD_RETRY_PAUSE_SEC // 60} мин озвучиваю "
                           f"локальным Silero (потом попробую облако снова)")

    def _get_local_fallback(self) -> "LocalTTSEngine":
        """Ленивая инициализация локального Silero — грузится только когда реально
        понадобился запасной вариант (облако недоступно/упало), не при старте,
        чтобы не терять время на загрузку модели, пока облако работает штатно."""
        if self.local_engine is None:
            logger.info("💻 Инициализирую локальный Silero как запасной TTS...")
            self.local_engine = LocalTTSEngine()
            self.local_engine.default_speaker = self.local_speaker
        return self.local_engine

    def synthesize(self, text: str, speaker: str = None) -> bytes:
        """В cloud-режиме игнорируем speaker из Silero, используем FREETTS_VOICE.
        В local-режиме speaker=None -> берётся SILERO_SPEAKER из config.yaml."""
        if self.mode == "cloud" and self.cloud_engine:
            if self._cloud_available():
                audio = self.cloud_engine.synthesize(text)
                if audio:
                    return audio
                self._mark_cloud_down()
            return self._get_local_fallback().synthesize(text, speaker)
        elif self.local_engine:
            return self.local_engine.synthesize(text, speaker)
        else:
            logger.error("TTS движок не инициализирован")
            return b""

    def synthesize_to_wav(self, text: str, output_path: Path, speaker: str = None) -> Path:
        if self.mode == "cloud" and self.cloud_engine:
            if self._cloud_available():
                wav_path = self.cloud_engine.synthesize_to_wav(text, output_path)
                if wav_path:
                    return wav_path
                self._mark_cloud_down()
            return self._get_local_fallback().synthesize_to_wav(text, output_path, speaker)
        elif self.local_engine:
            return self.local_engine.synthesize_to_wav(text, output_path, speaker)
        else:
            return None

    def get_available_speakers(self) -> list:
        if self.mode == "cloud" and self.cloud_engine:
            return self.cloud_engine.get_available_speakers()
        elif self.local_engine:
            return self.local_engine.get_available_speakers()
        return []