"""Аудио буфер с Voice Activity Detection"""
import collections
import logging
import webrtcvad
from config.settings import (
    SAMPLE_RATE, CHUNK_DURATION_MS, VAD_AGGRESSIVENESS, SILENCE_TIMEOUT_MS
)

logger = logging.getLogger(__name__)

# Минимальная длительность накопленной фразы, чтобы вообще отправлять её в STT.
MIN_SPEECH_MS = 300

# Окно предзаписи (pre-roll) для триггера speech_start — отдельная, короткая
# величина, НЕ совпадающая с SILENCE_TIMEOUT_MS (который про таймаут окончания
# фразы). Раньше ring_buffer ошибочно использовал SILENCE_TIMEOUT_MS и для
# предзаписи тоже — из-за этого буфер должен был набрать ~800мс подряд ДО
# триггера, и короткие команды не успевали его заполнить и никогда не
# распознавались.
TRIGGER_WINDOW_MS = 300


class AudioBuffer:
    """Буфер аудио с VAD для определения начала/конца речи"""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.sample_rate = SAMPLE_RATE
        self.chunk_duration_ms = CHUNK_DURATION_MS
        self.chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

        self.ring_buffer = collections.deque(maxlen=int(TRIGGER_WINDOW_MS / CHUNK_DURATION_MS))
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
            status: "silence", "speech_start", "speech", "complete"
            audio_bytes: накопленное аудио (только при "complete")
        """
        if len(pcm_bytes) != self.chunk_size * 2:
            logger.warning(f"Неверный размер чанка: {len(pcm_bytes)} != {self.chunk_size * 2}")
            return "silence", b""

        is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)

        if not self.triggered:
            # Ждём начала речи
            self.ring_buffer.append(pcm_bytes)

            # Оцениваем только когда буфер набрал полное окно — иначе один
            # ложно распознанный VAD-кадр в начале (щелчок, гул) даёт
            # 100%-й voiced ratio на выборке из 1-2 чанков и мгновенно
            # триггерит speech_start. Сравнивать нужно с maxlen (полным
            # окном), а не с текущей длиной буфера.
            if len(self.ring_buffer) < self.ring_buffer.maxlen:
                return "silence", b""

            num_voiced = sum(
                self.vad.is_speech(f, self.sample_rate) 
                for f in self.ring_buffer
            )

            if num_voiced > 0.9 * self.ring_buffer.maxlen:
                self.triggered = True
                self.voiced_frames = list(self.ring_buffer)
                self.ring_buffer.clear()
                # Отдельный статус именно для МОМЕНТА начала новой фразы —
                # используется, например, для выбора цели слежения по губам.
                return "speech_start", b""

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
                speech_ms = len(self.voiced_frames) * self.chunk_duration_ms
                self.reset()

                if speech_ms < MIN_SPEECH_MS:
                    logger.debug(f"Отброшена короткая фраза: {speech_ms}ms < {MIN_SPEECH_MS}ms")
                    return "silence", b""

                return "complete", audio

            return "speech", b""

    def reset(self):
        """Сбрасывает состояние буфера"""
        self.triggered = False
        self.voiced_frames = []
        self.silence_frames = 0
        self.ring_buffer.clear()