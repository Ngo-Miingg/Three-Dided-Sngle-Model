import math
import numpy as np

def rotation_matrix_to_euler_degrees(R: np.ndarray) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees from a 3x3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])      # pitch
        y = math.atan2(-R[2, 0], sy)          # yaw
        z = math.atan2(R[1, 0], R[0, 0])      # roll
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    pitch, yaw, roll = np.degrees([x, y, z]).astype(float)
    return float(pitch), float(yaw), float(roll)
