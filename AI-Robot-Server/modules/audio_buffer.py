"""Аудио буфер с Voice Activity Detection"""
import collections
import logging
import webrtcvad
from config.settings import (
    SAMPLE_RATE, CHUNK_DURATION_MS, VAD_AGGRESSIVENESS, SILENCE_TIMEOUT_MS
)

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Буфер аудио с VAD для определения начала/конца речи"""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.sample_rate = SAMPLE_RATE
        self.chunk_duration_ms = CHUNK_DURATION_MS
        self.chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

        self.ring_buffer = collections.deque(maxlen=int(SILENCE_TIMEOUT_MS / CHUNK_DURATION_MS))
        self.triggered = False
        self.voiced_frames = []
        self.silence_frames = 0
        self.max_silence_frames = int(SILENCE_TIMEOUT_MS / CHUNK_DURATION_MS)

        logger.info(f"AudioBuffer инициализирован: {SAMPLE_RATE}Hz, chunk={self.chunk_size} samples")

    def process_chunk(self, pcm_bytes: bytes) -> tuple:
        """
        Обрабатывает чанк аудио

        Args:
            pcm_bytes: 16-bit PCM mono, длина = chunk_size * 2 байт

        Returns:
            (status, audio_bytes)
            status: "silence", "speech", "complete"
            audio_bytes: накопленное аудио (только при "complete")
        """
        if len(pcm_bytes) != self.chunk_size * 2:
            logger.warning(f"Неверный размер чанка: {len(pcm_bytes)} != {self.chunk_size * 2}")
            return "silence", b""

        is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)

        if not self.triggered:
            # Ждём начала речи
            self.ring_buffer.append(pcm_bytes)
            num_voiced = sum(
                self.vad.is_speech(f, self.sample_rate) 
                for f in self.ring_buffer
            )

            if num_voiced > 0.9 * len(self.ring_buffer):
                self.triggered = True
                self.voiced_frames = list(self.ring_buffer)
                self.ring_buffer.clear()
                return "speech", b""

            return "silence", b""

        else:
            # Речь идёт
            self.voiced_frames.append(pcm_bytes)

            if not is_speech:
                self.silence_frames += 1
            else:
                self.silence_frames = 0

            if self.silence_frames > self.max_silence_frames:
                # Речь закончилась
                audio = b"".join(self.voiced_frames)
                self.reset()
                return "complete", audio

            return "speech", b""

    def reset(self):
        """Сбрасывает состояние буфера"""
        self.triggered = False
        self.voiced_frames = []
        self.silence_frames = 0
        self.ring_buffer.clear()
