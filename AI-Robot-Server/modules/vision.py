"""Vision модуль - обработка видео: YOLO + MediaPipe (поза) + OpenCV (лицо, трекинг, губы)"""
import cv2
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config.settings import (
    YOLO_MODEL,
    ENABLE_POSE_TRACKING,
    FACE_DETECTOR_PROTOTXT,
    FACE_DETECTOR_MODEL,
    FACE_DETECTOR_CONFIDENCE,
    FACE_DETECTOR_INPUT_SIZE,
    FACE_CASCADE_SCALE_FACTOR,
    FACE_CASCADE_MIN_NEIGHBORS,
    FACE_CASCADE_MIN_SIZE,
    FACE_TRACKER_MAX_DISTANCE,
    FACE_TRACKER_MAX_MISSED_FRAMES,
    LIP_ACTIVITY_WINDOW_SEC,
    MIN_LIP_SAMPLES_FOR_DECISION,
    YOLO_DETECTION_INTERVAL_SEC,
    POSE_DETECTION_INTERVAL_SEC,
)
from modules.face_tracker import FaceTracker, FaceTrack

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics не установлен. YOLO недоступен.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe не установлен.")

_COLOR_TRACKING = (46, 204, 113)
_COLOR_IDLE = (140, 140, 140)
_COLOR_OTHER = (90, 90, 90)

_LIP_TOP = 13
_LIP_BOTTOM = 14
_LIP_LEFT = 61
_LIP_RIGHT = 291


class VisionEngine:
    """Движок компьютерного зрения"""

    def __init__(self):
        self.yolo = None
        self.pose = None
        self.face_mesh = None
        self.face_detector_net = None
        self.face_cascade = None

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

        self._init_face_detector()

        if MEDIAPIPE_AVAILABLE:
            if ENABLE_POSE_TRACKING:
                self._try_init_pose()
            self._try_init_face_mesh()

        self.tracker = FaceTracker(
            max_distance=FACE_TRACKER_MAX_DISTANCE,
            max_missed_frames=FACE_TRACKER_MAX_MISSED_FRAMES,
            lip_window_sec=LIP_ACTIVITY_WINDOW_SEC,
        )
        self.target_track_id: Optional[int] = None
        self.speech_episode_active: bool = False

        self.last_pose_landmarks = None
        self.person_detected = False
        self.face_position = None
        self.face_bbox = None
        self.all_faces_bbox: List[Tuple[int, int, int, int]] = []
        self.last_frame = None

        # === Поддержка двух видеопотоков от ESP32 (если прошивка обновлена) ===
        # "VIDE" (частый, маленькое разрешение) → self.last_frame, используется для
        # детекции. "VIDP" (редкий, покрупнее, только для человека) → self._panel_frame.
        # Bbox всегда считается в системе координат last_frame на момент детекции
        # (self._detection_frame_size) и масштабируется под фактический размер кадра
        # панели при отрисовке — так что если прошивка НЕ обновлена и VIDP не приходит,
        # всё работает ровно как раньше (self._panel_frame остаётся None → fallback на last_frame).
        self._panel_frame = None
        self._detection_frame_size: Optional[Tuple[int, int]] = None  # (w, h)

        # Троттлинг тяжёлых детекторов (YOLO/Pose не гоняем на каждый кадр —
        # см. YOLO_DETECTION_INTERVAL_SEC / POSE_DETECTION_INTERVAL_SEC).
        # Лицо/трекер/губы намеренно НЕ троттлятся — от их частоты зависит
        # плавность слежения и скорость выбора говорящего.
        self._last_yolo_run_ts = 0.0
        self._last_pose_run_ts = 0.0
        self._cached_objects: List[Dict] = []  # последний результат YOLO (переиспользуется между запусками)

    def _init_face_detector(self):
        if FACE_DETECTOR_PROTOTXT.exists() and FACE_DETECTOR_MODEL.exists():
            try:
                self.face_detector_net = cv2.dnn.readNetFromCaffe(
                    str(FACE_DETECTOR_PROTOTXT), str(FACE_DETECTOR_MODEL)
                )
                logger.info("Детектор лиц: OpenCV DNN (res10_300x300_ssd) — основной, точный")
                return
            except Exception as e:
                logger.error(f"Не удалось загрузить DNN детектор лиц: {e}")
                self.face_detector_net = None
        else:
            logger.warning(
                "Модель DNN-детектора лиц не найдена. "
                "Скачай: python scripts/download_face_detector.py — используем Haar Cascade как запасной вариант."
            )
        self._init_face_cascade()

    def _init_face_cascade(self):
        try:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                logger.error(f"Не удалось загрузить Haar cascade: {cascade_path}")
                self.face_cascade = None
            else:
                self.face_cascade = cascade
                logger.info(f"OpenCV Haar Cascade (лицо) загружен: {cascade_path}")
        except Exception as e:
            logger.error(f"Ошибка инициализации OpenCV face cascade: {e}")
            self.face_cascade = None

    def _try_init_pose(self):
        try:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose инициализирован")
        except AttributeError:
            logger.warning("MediaPipe старый API недоступен (mp.solutions) — поза тела не отслеживается")
            self.pose = None
        except Exception as e:
            logger.warning(f"MediaPipe Pose не удалось инициализировать: {e}")
            self.pose = None

    def _try_init_face_mesh(self):
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe FaceMesh инициализирован (точный детектор движения губ)")
        except AttributeError:
            logger.info("MediaPipe FaceMesh недоступен — активность губ считается через OpenCV (разница пикселей ROI)")
            self.face_mesh = None
        except Exception as e:
            logger.warning(f"MediaPipe FaceMesh не удалось инициализировать: {e}")
            self.face_mesh = None

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        boxes: List[Tuple[int, int, int, int]] = []

        if self.face_detector_net is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (FACE_DETECTOR_INPUT_SIZE, FACE_DETECTOR_INPUT_SIZE)),
                    1.0,
                    (FACE_DETECTOR_INPUT_SIZE, FACE_DETECTOR_INPUT_SIZE),
                    (104.0, 177.0, 123.0)
                )
                self.face_detector_net.setInput(blob)
                det = self.face_detector_net.forward()
                for i in range(det.shape[2]):
                    conf = float(det[0, 0, i, 2])
                    if conf < FACE_DETECTOR_CONFIDENCE:
                        continue
                    box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2 - x1, y2 - y1))
            except Exception as e:
                logger.debug(f"DNN Face Detection ошибка: {e}")
        elif self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=FACE_CASCADE_SCALE_FACTOR,
                    minNeighbors=FACE_CASCADE_MIN_NEIGHBORS,
                    minSize=(FACE_CASCADE_MIN_SIZE, FACE_CASCADE_MIN_SIZE)
                )
                boxes = [(int(x), int(y), int(w_), int(h_)) for (x, y, w_, h_) in faces]
            except Exception as e:
                logger.debug(f"Haar Face Detection ошибка: {e}")

        return boxes

    def _update_lip_activity(self, frame: np.ndarray, track: FaceTrack):
        x, y, w_, h_ = track.bbox
        if w_ <= 0 or h_ <= 0:
            return

        if self.face_mesh is not None:
            try:
                pad_x, pad_y = int(w_ * 0.15), int(h_ * 0.15)
                x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
                x1, y1 = min(frame.shape[1], x + w_ + pad_x), min(frame.shape[0], y + h_ + pad_y)
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    return
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mesh_result = self.face_mesh.process(rgb)
                if mesh_result and mesh_result.multi_face_landmarks:
                    lm = mesh_result.multi_face_landmarks[0].landmark
                    ch, cw = crop.shape[:2]
                    vert = abs(lm[_LIP_BOTTOM].y - lm[_LIP_TOP].y) * ch
                    horiz = abs(lm[_LIP_RIGHT].x - lm[_LIP_LEFT].x) * cw
                    ratio = vert / horiz if horiz > 1e-3 else 0.0

                    prev_ratio = getattr(track, "_last_mouth_ratio", None)
                    track._last_mouth_ratio = ratio
                    activity = abs(ratio - prev_ratio) if prev_ratio is not None else 0.0
                    track.add_lip_sample(activity)
                return
            except Exception as e:
                logger.debug(f"FaceMesh (губы) ошибка: {e}")

        try:
            mouth_y0 = max(0, y + int(h_ * 0.62))
            mouth_y1 = min(frame.shape[0], y + int(h_ * 0.95))
            mouth_x0 = max(0, x + int(w_ * 0.22))
            mouth_x1 = min(frame.shape[1], x + int(w_ * 0.78))
            if mouth_y1 <= mouth_y0 or mouth_x1 <= mouth_x0:
                return
            roi = cv2.cvtColor(frame[mouth_y0:mouth_y1, mouth_x0:mouth_x1], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (40, 24))
            if track.prev_mouth_roi is not None:
                diff = cv2.absdiff(roi, track.prev_mouth_roi)
                track.add_lip_sample(float(np.mean(diff)))
            track.prev_mouth_roi = roi
        except Exception as e:
            logger.debug(f"Fallback разница пикселей рта ошибка: {e}")

    def update_panel_frame(self, frame_bytes: bytes) -> bool:
        """Принимает более качественный/крупный кадр ТОЛЬКО для показа в панели мониторинга
        (тег VIDP от прошивки). НЕ запускает детекцию — просто обновляет картинку,
        которую get_annotated_jpeg() использует как фон (bbox масштабируется отдельно)."""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return False
            self._panel_frame = frame
            return True
        except Exception as e:
            logger.error(f"Ошибка декодирования кадра панели (VIDP): {e}")
            return False

    def notify_speech_start(self):
        self.speech_episode_active = True

    def notify_speech_end(self):
        self.speech_episode_active = False

    def _select_target(self, tracks: List[FaceTrack]):
        if not tracks:
            self.target_track_id = None
            return

        if self.speech_episode_active:
            candidates = [t for t in tracks if t.sample_count() >= MIN_LIP_SAMPLES_FOR_DECISION]
            if candidates:
                best = max(candidates, key=lambda t: t.lip_activity_score())
                self.target_track_id = best.id
                return

        current = self.tracker.get_track(self.target_track_id) if self.target_track_id is not None else None
        if current is not None and current.missed_frames == 0:
            return

        biggest = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])
        self.target_track_id = biggest.id

    def process_frame(self, frame_bytes: bytes) -> dict:
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return {"error": "Не удалось декодировать кадр"}

            self.last_frame = frame
            self._detection_frame_size = (frame.shape[1], frame.shape[0])  # (w, h)

            result = {
                "objects": [],
                "pose_landmarks": None,
                "face_detected": False,
                "face_position": None,
                "face_bbox": None,
                "faces_count": 0,
                "target_track_id": None,
                "description": ""
            }

            # 1. YOLO Detection (троттлится — см. YOLO_DETECTION_INTERVAL_SEC,
            # результат нужен только для текстового описания сцены для LLM,
            # частая переоценка не требуется)
            now = time.time()
            if self.yolo is not None and (now - self._last_yolo_run_ts) >= YOLO_DETECTION_INTERVAL_SEC:
                self._last_yolo_run_ts = now
                try:
                    yolo_results = self.yolo(frame, verbose=False)
                    objects = []
                    for r in yolo_results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = self.yolo.names[cls_id]
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            objects.append({
                                "class": cls_name,
                                "confidence": round(conf, 3),
                                "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            })

                            if cls_name == "person":
                                self.person_detected = True
                    self._cached_objects = objects
                except Exception as e:
                    logger.error(f"Ошибка YOLO: {e}")
            result["objects"] = self._cached_objects

            # 2. Детекция лиц (DNN/Haar) + трекинг между кадрами + активность губ + выбор цели
            # (намеренно НЕ троттлится — см. комментарий в __init__)
            face_boxes = self._detect_faces(frame)
            tracks = self.tracker.update(face_boxes)
            for track in tracks:
                self._update_lip_activity(frame, track)
            self._select_target(tracks)

            self.all_faces_bbox = [t.bbox for t in tracks]
            result["faces_count"] = len(tracks)

            target = self.tracker.get_track(self.target_track_id) if self.target_track_id is not None else None
            if target is not None:
                result["face_detected"] = True
                result["face_bbox"] = list(target.bbox)
                cx, cy = target.center()
                result["face_position"] = (int(cx), int(cy))
                result["target_track_id"] = target.id
                self.face_position = (int(cx), int(cy))
                self.face_bbox = target.bbox
            else:
                self.face_position = None
                self.face_bbox = None

            # 3. Pose Tracking (троттлится — см. POSE_DETECTION_INTERVAL_SEC;
            # руки/плечи не требуют такой же реакции, как голова)
            if self.pose is not None and (now - self._last_pose_run_ts) >= POSE_DETECTION_INTERVAL_SEC:
                self._last_pose_run_ts = now
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
                        self.last_pose_landmarks = landmarks
                except Exception as e:
                    logger.debug(f"Pose Tracking ошибка: {e}")
            result["pose_landmarks"] = self.last_pose_landmarks

            result["description"] = self._generate_description(result)

            return result

        except Exception as e:
            logger.error(f"Ошибка vision: {e}")
            return {"error": str(e)}

    def _generate_description(self, result: dict) -> str:
        parts = []

        if result["face_detected"]:
            if result["faces_count"] > 1:
                parts.append(f"вижу {result['faces_count']} человек, слежу за собеседником")
            else:
                parts.append("вижу лицо человека")

        if result["pose_landmarks"]:
            parts.append("вижу позу человека")

        objects = [o["class"] for o in result["objects"] if o["class"] != "person"]
        if objects:
            parts.append(f"рядом объекты: {', '.join(set(objects))}")

        if not parts:
            return "ничего не вижу"

        return "; ".join(parts)

    def get_annotated_jpeg(self, tracking_active: bool = False, quality: int = 70) -> Optional[bytes]:
        # Показываем кадр панели (крупнее, реже), если прошивка его присылает;
        # иначе — обычный кадр детекции (полностью обратная совместимость).
        base_frame = self._panel_frame if self._panel_frame is not None else self.last_frame
        if base_frame is None:
            return None

        frame = base_frame.copy()
        target_bbox = self.face_bbox

        # Bbox посчитан в системе координат last_frame (кадра детекции) — если панель
        # показывает кадр ДРУГОГО размера (VIDP), масштабируем координаты пропорционально.
        scale_x, scale_y = 1.0, 1.0
        if self._detection_frame_size is not None:
            det_w, det_h = self._detection_frame_size
            disp_h, disp_w = frame.shape[:2]
            if det_w > 0 and det_h > 0:
                scale_x = disp_w / det_w
                scale_y = disp_h / det_h

        def _scaled(bbox):
            x, y, w_, h_ = bbox
            return (int(x * scale_x), int(y * scale_y), int(w_ * scale_x), int(h_ * scale_y))

        for bbox in self.all_faces_bbox:
            if target_bbox is not None and tuple(bbox) == tuple(target_bbox):
                continue
            x, y, w_, h_ = _scaled(bbox)
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), _COLOR_OTHER, 1)

        if target_bbox is not None:
            x, y, w_, h_ = _scaled(target_bbox)
            color = _COLOR_TRACKING if tracking_active else _COLOR_IDLE
            cv2.rectangle(frame, (x, y), (x + w_, y + h_), color, 2)
            label = "TRACKING" if tracking_active else "TARGET"
            label_y = max(y - 8, 14)
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        try:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        except Exception as e:
            logger.error(f"Ошибка кодирования JPEG для панели: {e}")
            return None

        if not ok:
            return None
        return buf.tobytes()

    def get_face_offset(self, frame_width: int, frame_height: int) -> Tuple[float, float]:
        if self.face_position is None:
            return (0.0, 0.0)

        cx, cy = self.face_position
        offset_x = (cx - frame_width / 2) / (frame_width / 2)
        offset_y = (cy - frame_height / 2) / (frame_height / 2)
        return (offset_x, offset_y)

    def get_servo_angles_from_pose(self) -> List[int]:
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
        if self.pose:
            try:
                self.pose.close()
            except Exception:
                pass
        if self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception:
                pass
