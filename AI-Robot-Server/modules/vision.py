"""Vision модуль - обработка видео: YOLO + MediaPipe"""
import cv2
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config.settings import YOLO_MODEL, ENABLE_POSE_TRACKING

logger = logging.getLogger(__name__)

# Пробуем импортировать YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics не установлен. YOLO недоступен.")

# Пробуем импортировать MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe не установлен.")


class VisionEngine:
    """Движок компьютерного зрения"""

    def __init__(self):
        self.yolo = None
        self.pose = None
        self.face_detection = None

        # Загрузка YOLO
        if YOLO_AVAILABLE:
            logger.info("Загрузка YOLO модели...")
            if YOLO_MODEL.exists():
                try:
                    self.yolo = YOLO(str(YOLO_MODEL))
                    logger.info(f"YOLO загружен: {YOLO_MODEL}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки YOLO: {e}")
            else:
                logger.warning(f"YOLO модель не найдена: {YOLO_MODEL}")
                logger.info("Скачай: python download_yolo.py")

        # Пробуем инициализировать MediaPipe — но НЕ падаем при ошибке
        if MEDIAPIPE_AVAILABLE and ENABLE_POSE_TRACKING:
            self._try_init_mediapipe()

        self.last_pose_landmarks = None
        self.person_detected = False
        self.face_position = None

    def _try_init_mediapipe(self):
        """Пробует инициализировать MediaPipe, но не падает при ошибке"""
        # Пробуем старый API (MediaPipe < 0.10)
        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose инициализирован (old API)")
        except AttributeError:
            logger.warning("MediaPipe старый API недоступен (mp.solutions)")
            self.pose = None
        except Exception as e:
            logger.warning(f"MediaPipe Pose не удалось инициализировать: {e}")
            self.pose = None

        try:
            self.face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe Face Detection инициализирован")
        except AttributeError:
            logger.warning("MediaPipe Face Detection недоступен")
            self.face_detection = None
        except Exception as e:
            logger.warning(f"MediaPipe Face Detection не удалось: {e}")
            self.face_detection = None

    def process_frame(self, frame_bytes: bytes) -> dict:
        """Обрабатывает кадр видео"""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"error": "Не удалось декодировать кадр"}

            h, w = frame.shape[:2]
            result = {
                "objects": [],
                "pose_landmarks": None,
                "face_detected": False,
                "face_position": None,
                "description": ""
            }

            # 1. YOLO Detection
            if self.yolo is not None:
                try:
                    yolo_results = self.yolo(frame, verbose=False)
                    for r in yolo_results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = self.yolo.names[cls_id]
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            result["objects"].append({
                                "class": cls_name,
                                "confidence": round(conf, 3),
                                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            })

                            if cls_name == "person":
                                self.person_detected = True
                except Exception as e:
                    logger.error(f"Ошибка YOLO: {e}")

            # 2. Face Detection (если доступно)
            if self.face_detection is not None:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = self.face_detection.process(rgb_frame)
                    if face_results and face_results.detections:
                        result["face_detected"] = True
                        detection = face_results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        cx = int((bbox.xmin + bbox.width/2) * w)
                        cy = int((bbox.ymin + bbox.height/2) * h)
                        result["face_position"] = (cx, cy)
                        self.face_position = (cx, cy)
                except Exception as e:
                    logger.debug(f"Face Detection ошибка: {e}")

            # 3. Pose Tracking (если доступно)
            if self.pose is not None:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(rgb_frame)
                    if pose_results and pose_results.pose_landmarks:
                        landmarks = []
                        for lm in pose_results.pose_landmarks.landmark:
                            landmarks.append({
                                "x": lm.x,
                                "y": lm.y,
                                "z": lm.z,
                                "visibility": lm.visibility
                            })
                        result["pose_landmarks"] = landmarks
                        self.last_pose_landmarks = landmarks
                except Exception as e:
                    logger.debug(f"Pose Tracking ошибка: {e}")

            # 4. Генерируем текстовое описание
            result["description"] = self._generate_description(result)

            return result

        except Exception as e:
            logger.error(f"Ошибка vision: {e}")
            return {"error": str(e)}

    def _generate_description(self, result: dict) -> str:
        """Генерирует текстовое описание для LLM"""
        parts = []

        if result["face_detected"]:
            parts.append("вижу лицо человека")

        if result["pose_landmarks"]:
            parts.append("вижу позу человека")

        objects = [o["class"] for o in result["objects"] if o["class"] != "person"]
        if objects:
            parts.append(f"рядом объекты: {', '.join(set(objects))}")

        if not parts:
            return "ничего не вижу"

        return "; ".join(parts)

    def get_face_offset(self, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """Возвращает смещение лица от центра кадра"""
        if self.face_position is None:
            return (0.0, 0.0)

        cx, cy = self.face_position
        offset_x = (cx - frame_width / 2) / (frame_width / 2)
        offset_y = (cy - frame_height / 2) / (frame_height / 2)
        return (offset_x, offset_y)

    def get_servo_angles_from_pose(self) -> List[int]:
        """Вычисляет углы сервоприводов на основе позы человека"""
        if self.last_pose_landmarks is None:
            return [90] * 18

        angles = [90] * 18

        try:
            left_shoulder = self.last_pose_landmarks[11]
            right_shoulder = self.last_pose_landmarks[12]
            angles[0] = int(left_shoulder["y"] * 180)
            angles[1] = int(right_shoulder["y"] * 180)

            left_elbow = self.last_pose_landmarks[13]
            right_elbow = self.last_pose_landmarks[14]
            angles[2] = int(left_elbow["y"] * 180)
            angles[3] = int(right_elbow["y"] * 180)

            nose = self.last_pose_landmarks[0]
            angles[16] = int(nose["x"] * 180)
            angles[17] = int(nose["y"] * 180)

        except (IndexError, KeyError):
            pass

        return angles

    def release(self):
        """Освобождает ресурсы"""
        if self.pose:
            try:
                self.pose.close()
            except:
                pass
        if self.face_detection:
            try:
                self.face_detection.close()
            except:
                pass
