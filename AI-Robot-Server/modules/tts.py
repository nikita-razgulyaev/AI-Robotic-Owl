"""TTS модуль - синтез речи (Silero local / FreeTTS cloud via edge-tts)"""
import io
import wave
import logging
import tempfile
import os
import subprocess
import shutil
from pathlib import Path
from config.settings import (
    TTS_MODE, AUDIO_CACHE_DIR, SAMPLE_RATE,
    FREETTS_VOICE, FREETTS_SPEED
)

logger = logging.getLogger(__name__)


class FreeTTSEngine:
    """Бесплатный облачный синтез речи через edge-tts (Microsoft Edge TTS)"""

    def __init__(self):
        self.voice = FREETTS_VOICE
        self.speed = FREETTS_SPEED
        self._check_edge_tts()
        logger.info(f"☁️ FreeTTS (edge-tts) инициализирован: voice={self.voice}, speed={self.speed}")

    def _check_edge_tts(self):
        """Проверяет, установлен ли edge-tts"""
        if not shutil.which("edge-tts"):
            logger.warning("edge-tts CLI не найден. Установи: pip install edge-tts")

    def synthesize(self, text: str, speaker: str = None) -> bytes:
        """Синтезирует речь через edge-tts CLI, возвращает PCM 16-bit 48kHz"""
        try:
            import soundfile as sf
            import librosa
            import numpy as np

            # Игнорируем speaker из Silero, используем FREETTS_VOICE
            voice = self.voice
            speed_pct = int((self.speed - 1.0) * 100)
            rate_arg = f"+{speed_pct}%" if speed_pct >= 0 else f"{speed_pct}%"

            # Временные файлы
            tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tmp_mp3.close()

            # Запускаем edge-tts через subprocess (самый надёжный способ)
            cmd = [
                "edge-tts",
                "--voice", voice,
                "--rate", rate_arg,
                "--text", text,
                "--write-media", tmp_mp3.name,
            ]

            logger.debug(f"FreeTTS cmd: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"edge-tts ошибка: {result.stderr}")
                os.unlink(tmp_mp3.name)
                return b""

            # Читаем MP3 и конвертируем в PCM 16-bit mono 48kHz
            audio, sr = sf.read(tmp_mp3.name)
            os.unlink(tmp_mp3.name)

            # Ресемплинг в 48kHz если нужно
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

            # В mono если stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Нормализация и конвертация в int16
            if audio.size > 0:
                max_val = max(abs(audio).max(), 1e-9)
                audio = audio / max_val * 32767
            audio_int16 = audio.astype('int16')
            return audio_int16.tobytes()

        except ImportError as e:
            logger.error(f"Библиотека не установлена: {e}. Установи: pip install edge-tts soundfile librosa numpy")
            return b""
        except subprocess.TimeoutExpired:
            logger.error("FreeTTS: таймаут синтеза (60 сек)")
            return b""
        except Exception as e:
            logger.error(f"Ошибка FreeTTS (edge-tts): {e}")
            import traceback
            traceback.print_exc()
            return b""

    def synthesize_to_wav(self, text: str, output_path: Path = None, speaker: str = None) -> Path:
        """Синтезирует в WAV файл"""
        if output_path is None:
            output_path = AUDIO_CACHE_DIR / f"tts_freetts_{hash(text)}.wav"

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
            "ru-RU-SvetlanaNeural",   # Женский, мягкий
            "ru-RU-DmitryNeural",     # Мужской, уверенный
            "ru-RU-EkaterinaNeural",  # Женский, дружелюбный
        ]


class LocalTTSEngine:
    """Локальный синтез речи через Silero (PyTorch)"""

    def __init__(self):
        self.device = None
        self.model = None
        self.sample_rate = 48000
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

    def synthesize(self, text: str, speaker: str = 'xenia') -> bytes:
        try:
            if self.model is None:
                logger.error("Silero модель не загружена")
                return b""

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

    def synthesize_to_wav(self, text: str, output_path: Path = None, speaker: str = 'xenia') -> Path:
        if output_path is None:
            output_path = AUDIO_CACHE_DIR / f"tts_local_{hash(text)}.wav"

        pcm_data = self.synthesize(text, speaker)
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
            self.local_engine = None
            logger.info("☁️ TTS переключён на ОБЛАЧНЫЙ (FreeTTS)")
        else:
            if self.local_engine is None:
                self.local_engine = LocalTTSEngine()
            self.cloud_engine = None
            logger.info("💻 TTS переключён на ЛОКАЛЬНЫЙ")

    def get_mode(self) -> str:
        return self.mode

    def synthesize(self, text: str, speaker: str = 'xenia') -> bytes:
        """В cloud-режиме игнорируем speaker из Silero, используем FREETTS_VOICE"""
        if self.mode == "cloud" and self.cloud_engine:
            # Не передаём speaker — FreeTTSEngine использует self.voice (FREETTS_VOICE)
            return self.cloud_engine.synthesize(text)
        elif self.local_engine:
            return self.local_engine.synthesize(text, speaker)
        else:
            logger.error("TTS движок не инициализирован")
            return b""

    def synthesize_to_wav(self, text: str, output_path: Path = None, speaker: str = 'xenia') -> Path:
        if self.mode == "cloud" and self.cloud_engine:
            return self.cloud_engine.synthesize_to_wav(text, output_path)
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