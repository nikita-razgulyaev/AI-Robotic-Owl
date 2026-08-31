"""
Лёгкий трекер лиц между кадрами (без ML).

Задача: детектор (Haar/DNN) каждый кадр находит лица заново и НЕ гарантирует
одинаковый порядок/индексацию между кадрами. Чтобы понять "какое из уже
виденных лиц — это тот же человек", нужно сопоставлять новые детекции со
старыми треками — это и делает FaceTracker (сопоставление по расстоянию
между центрами bbox, простая и дешёвая эвристика, без переобучаемых моделей).
"""
import logging
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]  # x, y, w, h


class FaceTrack:
    """Один отслеживаемый человек в кадре"""

    def __init__(self, track_id: int, bbox: BBox):
        self.id = track_id
        self.bbox = bbox
        self.missed_frames = 0

    def update_bbox(self, bbox: BBox):
        self.bbox = bbox
        self.missed_frames = 0

    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)


class FaceTracker:
    """Сопоставляет детекции лиц между кадрами по ближайшему центру bbox"""

    def __init__(self, max_distance: float, max_missed_frames: int):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.tracks: Dict[int, FaceTrack] = {}
        self._next_id = 1

    def update(self, detections: List[BBox]) -> List[FaceTrack]:
        """
        Обновляет треки по новым детекциям текущего кадра.
        Возвращает список актуальных (видимых в этом кадре) треков.
        """
        unmatched_detections = list(range(len(detections)))
        matched_track_ids = set()

        # Жадное сопоставление: для каждого существующего трека ищем ближайшую
        # детекцию в пределах max_distance (простая, но достаточная эвристика
        # для 1-3 лиц в кадре, без тяжёлого венгерского алгоритма).
        for track_id, track in self.tracks.items():
            if not unmatched_detections:
                break
            tcx, tcy = track.center()
            best_idx = None
            best_dist = self.max_distance
            for idx in unmatched_detections:
                x, y, w, h = detections[idx]
                dcx, dcy = x + w / 2, y + h / 2
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is not None:
                track.update_bbox(detections[best_idx])
                matched_track_ids.add(track_id)
                unmatched_detections.remove(best_idx)

        # Детекции без пары — новые люди в кадре
        for idx in unmatched_detections:
            track = FaceTrack(self._next_id, detections[idx])
            self.tracks[self._next_id] = track
            matched_track_ids.add(self._next_id)
            self._next_id += 1

        # Треки без детекции в этом кадре — увеличиваем счётчик "потерян"
        visible_tracks = []
        dead_ids = []
        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids:
                visible_tracks.append(track)
            else:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    dead_ids.append(track_id)

        for track_id in dead_ids:
            del self.tracks[track_id]

        return visible_tracks

    def get_track(self, track_id: int) -> Optional[FaceTrack]:
        return self.tracks.get(track_id)

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
