import cv2
import numpy as np

def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, axis_len_px: int, origin_2d=None):
    """Draw 3D axes projected into the image.
    X = red, Y = green, Z = blue.

    origin_2d:
      - If provided (x,y) in pixels, we *anchor* the axes origin to this exact 2D point
        (e.g., MediaPipe nose landmark). This makes the axes start exactly at the nose center,
        even if the generic 3D face model doesn't perfectly match the person.
    """
    axes_3d = np.float32([
        [axis_len_px, 0, 0],   # X
        [0, axis_len_px, 0],   # Y
        [0, 0, axis_len_px],   # Z
    ])
    origin_3d = np.float32([[0, 0, 0]])

    imgpts_axes, _ = cv2.projectPoints(axes_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts_origin, _ = cv2.projectPoints(origin_3d, rvec, tvec, camera_matrix, dist_coeffs)

    o_proj = imgpts_origin[0].ravel().astype(np.float32)  # projected origin from the 3D model

    if origin_2d is not None:
        origin_2d = np.array(origin_2d, dtype=np.float32).ravel()
        delta = origin_2d - o_proj
    else:
        origin_2d = o_proj
        delta = np.zeros(2, dtype=np.float32)

    def shift_point(p):
        p2 = p.ravel().astype(np.float32) + delta
        return tuple(p2.astype(int))

    o = tuple(origin_2d.astype(int))
    xpt = shift_point(imgpts_axes[0])
    ypt = shift_point(imgpts_axes[1])
    zpt = shift_point(imgpts_axes[2])

    cv2.arrowedLine(frame, o, xpt, (0, 0, 255), 3, tipLength=0.2)   # Red
    cv2.arrowedLine(frame, o, ypt, (0, 255, 0), 3, tipLength=0.2)   # Green
    cv2.arrowedLine(frame, o, zpt, (255, 0, 0), 3, tipLength=0.2)   # Blue

def draw_pose_text(frame, yaw, pitch, roll, color=(0, 255, 0)):
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Roll: {roll:.2f}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
