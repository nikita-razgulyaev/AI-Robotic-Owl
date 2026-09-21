"""Пользовательские серво-анимации — по одному JSON-файлу на анимацию в
character/animations/
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import ANIMATIONS_DIR, ANIMATIONS as BUILTIN_ANIMATIONS, SERVO_CONFIG

logger = logging.getLogger(__name__)

MIN_ANGLE = SERVO_CONFIG.get("min_angle", 0)
MAX_ANGLE = SERVO_CONFIG.get("max_angle", 180)


@dataclass
class AnimationInfo:
    name: str
    frames: List[Dict]
    source: str  # "builtin" | "custom"
    description: str = ""
    file: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        return self.frames[-1]["time"] if self.frames else 0

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def _validate_frames(frames, context: str) -> Optional[List[Dict]]:
    """Проверяет список кадров, возвращает нормализованную копию (int-время,
    int-углы) либо None, если файл некорректен (тогда он просто пропускается,
    а не роняет весь сервер)."""
    if not isinstance(frames, list) or not frames:
        logger.error(f"Анимация «{context}»: поле 'frames' пустое или отсутствует — файл пропущен")
        return None

    normalized = []
    prev_time = -1
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict) or "time" not in frame or "servos" not in frame:
            logger.error(f"Анимация «{context}»: кадр #{idx} без 'time'/'servos' — файл пропущен")
            return None

        try:
            time_ms = float(frame["time"])
        except (TypeError, ValueError):
            logger.error(f"Анимация «{context}»: кадр #{idx} — 'time' не число — файл пропущен")
            return None

        if time_ms < 0:
            logger.error(f"Анимация «{context}»: кадр #{idx} — отрицательное 'time' — файл пропущен")
            return None
        if time_ms < prev_time:
            logger.error(
                f"Анимация «{context}»: кадр #{idx} — 'time' идёт назад "
                f"({time_ms} после {prev_time}), кадры должны быть по возрастанию времени — файл пропущен"
            )
            return None
        prev_time = time_ms

        servos = frame["servos"]
        if not isinstance(servos, list) or len(servos) != 18:
            logger.error(
                f"Анимация «{context}»: кадр #{idx} — 'servos' должен быть списком "
                f"ровно из 18 углов, получено {len(servos) if isinstance(servos, list) else type(servos).__name__} — файл пропущен"
            )
            return None

        clamped_servos = []
        for s_idx, angle in enumerate(servos):
            try:
                angle = int(round(float(angle)))
            except (TypeError, ValueError):
                logger.error(
                    f"Анимация «{context}»: кадр #{idx}, серво #{s_idx} — угол не число — файл пропущен"
                )
                return None
            if angle < MIN_ANGLE or angle > MAX_ANGLE:
                logger.warning(
                    f"Анимация «{context}»: кадр #{idx}, серво #{s_idx} — угол {angle} "
                    f"вне диапазона [{MIN_ANGLE}, {MAX_ANGLE}], обрезан"
                )
                angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
            clamped_servos.append(angle)

        normalized.append({"time": time_ms, "servos": clamped_servos})

    return normalized


class AnimationBook:
    """Загружает и хранит анимации — встроенные (config.settings.ANIMATIONS)
    + пользовательские (character/animations/*.json). Потокобезопасен,
    поддерживает hot-reload без рестарта сервера."""

    def __init__(self, directory: Path = ANIMATIONS_DIR):
        self.directory = directory
        self._lock = threading.Lock()
        self._animations: Dict[str, AnimationInfo] = {}
        self.reload()

    def reload(self) -> int:
        """Перечитывает встроенные анимации + character/animations/*.json с диска.
        Возвращает итоговое число анимаций. Можно вызывать во время работы сервера."""
        animations: Dict[str, AnimationInfo] = {}

        for name, frames in BUILTIN_ANIMATIONS.items():
            animations[name] = AnimationInfo(name=name, frames=frames, source="builtin")

        self.directory.mkdir(parents=True, exist_ok=True)
        custom_count = 0
        for path in sorted(self.directory.glob("*.json")):
            name = path.stem
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception as e:
                logger.error(f"Анимация «{name}» ({path.name}): ошибка чтения JSON — файл пропущен: {e}")
                continue

            if not isinstance(raw, dict):
                logger.error(f"Анимация «{name}» ({path.name}): корневой элемент должен быть объектом — файл пропущен")
                continue

            frames = _validate_frames(raw.get("frames"), name)
            if frames is None:
                continue

            if name in animations:
                logger.warning(f"Пользовательская анимация «{name}» переопределяет встроенную")

            animations[name] = AnimationInfo(
                name=name,
                frames=frames,
                source="custom",
                description=str(raw.get("description", "")),
                file=path.name,
            )
            custom_count += 1

        with self._lock:
            self._animations = animations

        logger.info(f"🎬 Анимации загружены: {len(animations)} всего ({custom_count} пользовательских из {self.directory})")
        return len(animations)

    def get_frames(self, name: str) -> Optional[List[Dict]]:
        with self._lock:
            info = self._animations.get(name)
        return info.frames if info else None

    def get_all(self) -> Dict[str, List[Dict]]:
        """Совместимо по форме с config.settings.ANIMATIONS — {имя: [кадры]}"""
        with self._lock:
            return {name: info.frames for name, info in self._animations.items()}

    def list_info(self) -> List[Dict]:
        """Для панели — список анимаций с метаданными (без самих кадров)."""
        with self._lock:
            items = list(self._animations.values())
        items.sort(key=lambda info: (info.source != "custom", info.name))
        return [
            {
                "name": info.name,
                "source": info.source,
                "description": info.description,
                "frame_count": info.frame_count,
                "duration_ms": info.duration_ms,
            }
            for info in items
        ]

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._animations


# Глобальный экземпляр — грузится один раз при импорте модуля
animation_book = AnimationBook()
