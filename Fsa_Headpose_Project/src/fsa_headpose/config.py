from dataclasses import dataclass

@dataclass
class AppConfig:
    camera_index: int = 0
    width: int | None = None
    height: int | None = None
    flip_view: bool = False

    # Mediapipe FaceMesh
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Visualization
    axis_len_px_min: int = 80
    axis_len_px_max: int = 180
    text_color_bgr: tuple[int, int, int] = (0, 255, 0)

    # Smoothing (EMA) for angles
    smooth_alpha: float = 0.75  # closer to 1 = smoother, but more latency
