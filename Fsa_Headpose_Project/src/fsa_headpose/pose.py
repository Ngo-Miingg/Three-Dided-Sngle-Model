import cv2
import numpy as np
import mediapipe as mp
from .math_utils import rotation_matrix_to_euler_degrees

# 1: nose tip, 152: chin, 33: left eye outer, 263: right eye outer, 61: left mouth, 291: right mouth
POSE_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]

# Fixed 3D face model (mm-ish)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),     # Nose tip  (model origin)
    (0.0, -330.0,  -65.0),     # Chin
    (-225.0, 170.0, -135.0),   # Left eye outer corner
    (225.0,  170.0, -135.0),   # Right eye outer corner
    (-150.0,-150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float64)

class HeadPoseEstimator:
    def __init__(self,
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 use_extrinsic_guess: bool = True,
                 refine_pnp: bool = True):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.use_extrinsic_guess = bool(use_extrinsic_guess)
        self.refine_pnp = bool(refine_pnp)

        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None

    def reset_guess(self):
        self._prev_rvec = None
        self._prev_tvec = None

    @staticmethod
    def _landmarks_bbox(face_landmarks, w: int, h: int):
        xs = [lm.x * w for lm in face_landmarks.landmark]
        ys = [lm.y * h for lm in face_landmarks.landmark]
        x1, x2 = int(max(0, min(xs))), int(min(w - 1, max(xs)))
        y1, y2 = int(max(0, min(ys))), int(min(h - 1, max(ys)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def estimate(self, frame_bgr: np.ndarray):
        """Returns dict or None:
        {
          'rvec','tvec','pitch','yaw','roll',
          'image_points'(6x2),
          'camera_matrix','dist_coeffs',
          'bbox' (x1,y1,x2,y2) in pixels
        }
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        bbox = self._landmarks_bbox(face_landmarks, w, h)

        image_points = []
        for i in POSE_LANDMARK_IDX:
            lm = face_landmarks.landmark[i]
            image_points.append((lm.x * w, lm.y * h))
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if self.use_extrinsic_guess and self._prev_rvec is not None and self._prev_tvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                self._prev_rvec,
                self._prev_tvec,
                True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            return None

        # Optional refine if supported
        if self.refine_pnp:
            try:
                if hasattr(cv2, "solvePnPRefineLM"):
                    rvec, tvec = cv2.solvePnPRefineLM(
                        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, rvec, tvec
                    )
                elif hasattr(cv2, "solvePnPRefineVVS"):
                    rvec, tvec = cv2.solvePnPRefineVVS(
                        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, rvec, tvec
                    )
            except Exception:
                pass

        self._prev_rvec = rvec
        self._prev_tvec = tvec

        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = rotation_matrix_to_euler_degrees(R)

        return {
            "rvec": rvec,
            "tvec": tvec,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "image_points": image_points,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "bbox": bbox,
        }
