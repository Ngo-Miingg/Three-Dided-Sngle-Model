import argparse
import cv2
import numpy as np
import os
import time
from datetime import datetime
from collections import deque

from .config import AppConfig
from .pose import HeadPoseEstimator
from .smoother import EMASmoother
from .draw import draw_axes, draw_pose_text


TARGET_POSES = ["LEFT", "RIGHT", "DOWN"]  # 3 góc 


# ---------------------------
# Utility: UI drawing helpers
# ---------------------------
def draw_transparent_box(img, x1, y1, x2, y2, alpha=0.55):
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(img.shape[1] - 1, x2)), int(min(img.shape[0] - 1, y2))
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_line(img, text, x, y, scale=0.65, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_progress_bar(img, x, y, w, h, progress, label=None):
    progress = float(np.clip(progress, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill_w = int(w * progress)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), (0, 255, 0), -1)
    if label:
        put_line(img, label, x, y - 8, scale=0.55, color=(0, 255, 0), thick=2)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------
# Camera helpers (30fps-friendly)
# ---------------------------
def set_camera_params(cap, width, height, fps, fourcc="MJPG"):
    # Many webcams reach 30fps more reliably with MJPG
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


# ---------------------------
# Quality gates
# ---------------------------
def expand_bbox(bbox, frame_w, frame_h, margin=0.25):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(frame_w - 1, x2 + mx)
    y2 = min(frame_h - 1, y2 + my)
    return (x1, y1, x2, y2)

def safe_crop(frame, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(frame.shape[1] - 1, x2)), int(min(frame.shape[0] - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()

def sharpness_score(bgr_crop):
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def size_gate(eye_dist_px, bbox, frame_w, frame_h, min_eye_dist=80.0, min_area_ratio=0.06):
    if bbox is None:
        return False
    if eye_dist_px < min_eye_dist:
        return False
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    return area >= (frame_w * frame_h * float(min_area_ratio))

def border_gate(bbox, frame_w, frame_h, margin_ratio=0.05):
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    mx = int(frame_w * float(margin_ratio))
    my = int(frame_h * float(margin_ratio))
    return (x1 > mx and y1 > my and x2 < frame_w - mx and y2 < frame_h - my)

def in_target_window(direction, yaw, pitch):
    # Relaxed: just require the labeled direction
    if direction in ("LEFT", "RIGHT", "DOWN"):
        return True
    return False



# ---------------------------
# Direction state with hysteresis (anti-flicker)
# ---------------------------
class DirectionState:
    def __init__(self):
        self.state = "CENTER"

    def update(self, yaw, pitch, yaw_enter=18, yaw_exit=12, pitch_enter=12, pitch_exit=8):
        s = self.state

        # Exit rules (hysteresis)
        if s == "LEFT":
            if yaw > -yaw_exit:
                s = "CENTER"
        elif s == "RIGHT":
            if yaw < yaw_exit:
                s = "CENTER"
        elif s == "DOWN":
            if pitch < pitch_exit:
                s = "CENTER"

        # Enter rules
        if s == "CENTER":
            if yaw <= -yaw_enter:
                s = "LEFT"
            elif yaw >= yaw_enter:
                s = "RIGHT"
            elif pitch >= pitch_enter:
                s = "DOWN"

        self.state = s
        return s


# ---------------------------
# Stability tracking (angles + bbox motion)
# ---------------------------
class StabilityTracker:
    def __init__(self, max_angle_jump=2.5, max_center_jump=8.0):
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_center = None
        self.max_angle_jump = float(max_angle_jump)
        self.max_center_jump = float(max_center_jump)

    def reset(self):
        self.prev_yaw = None
        self.prev_pitch = None
        self.prev_center = None

    def ok(self, yaw, pitch, bbox):
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

        if self.prev_yaw is None:
            self.prev_yaw, self.prev_pitch, self.prev_center = yaw, pitch, center
            return True

        if abs(yaw - self.prev_yaw) > self.max_angle_jump:
            return False
        if abs(pitch - self.prev_pitch) > self.max_angle_jump:
            return False
        if np.linalg.norm(center - self.prev_center) > self.max_center_jump:
            return False

        self.prev_yaw, self.prev_pitch, self.prev_center = yaw, pitch, center
        return True


# ---------------------------
# Best-frame buffer (pick sharpest)
# ---------------------------
class BestFrameBuffer:
    def __init__(self, maxlen=15):
        self.buf = deque(maxlen=int(maxlen))

    def clear(self):
        self.buf.clear()

    def push(self, crop):
        score = sharpness_score(crop)
        self.buf.append((score, crop))

    def best(self):
        if not self.buf:
            return None, 0.0
        score, crop = max(self.buf, key=lambda x: x[0])
        return crop, float(score)

    def __len__(self):
        return len(self.buf)


# ---------------------------
# Capture manager (sequential like common apps)
# ---------------------------
class CaptureFlow:
    def __init__(self, outdir, stable_frames=60, shots_per_pose=1, cooldown_sec=1.0):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        self.stable_frames = int(stable_frames)
        self._last_need_frames = int(stable_frames)
        self.shots_per_pose = int(shots_per_pose)
        self.cooldown_sec = float(cooldown_sec)

        self.pose_idx = 0  # sequential order
        self.done_counts = {p: 0 for p in TARGET_POSES}

        self.hold_count = 0
        self.last_save_t = 0.0
        self.buffer = BestFrameBuffer(maxlen=15)

    def reset(self):
        self.pose_idx = 0
        self.done_counts = {p: 0 for p in TARGET_POSES}
        self.hold_count = 0
        self.last_save_t = 0.0
        self.buffer.clear()

    def current_target(self):
        if self.pose_idx >= len(TARGET_POSES):
            return None
        return TARGET_POSES[self.pose_idx]

    def is_done(self):
        return self.pose_idx >= len(TARGET_POSES)

    def progress_text(self):
        parts = []
        for p in TARGET_POSES:
            ok = self.done_counts[p] >= self.shots_per_pose
            parts.append(f"{p}:{'✅' if ok else '⬜'} {self.done_counts[p]}/{self.shots_per_pose}")
        return "  ".join(parts)

    def update(self, direction, frame, bbox, eye_dist_px, yaw, pitch,
               gates_ok, target_ok, stability_ok, sharp_ok, crop_for_buffer):
        """
        Returns (saved_path or None, message)
        """
        now = time.perf_counter()

        if self.is_done():
            return None, "DONE"

        target = self.current_target()

        # Cooldown
        if now - self.last_save_t < self.cooldown_sec:
            self.hold_count = 0
            self.buffer.clear()
            return None, f"Cooldown..."

        # Only build hold if ALL conditions match target
        # Soft-window: nếu không vào window thì vẫn cho giữ, nhưng sẽ yêu cầu giữ lâu hơn
        need_frames = self.stable_frames if target_ok else int(self.stable_frames * 1.5)
        self._last_need_frames = max(1, need_frames)

        if (direction == target and gates_ok and stability_ok and sharp_ok):
            self.hold_count += 1
            if crop_for_buffer is not None:
                self.buffer.push(crop_for_buffer)
            else:
                # Fallback: push whole frame so we always have something to save
                self.buffer.push(frame)
        else:
            self.hold_count = 0
            self.buffer.clear()

        # Và khi check đủ:
        if self.hold_count >= need_frames and len(self.buffer) >= 1:
            best_crop, best_score = self.buffer.best()
            if best_crop is None:
                self.hold_count = 0
                self.buffer.clear()
                return None, "No best frame"

            # Save best crop
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            idx = self.done_counts[target] + 1
            pose_dir = os.path.join(self.outdir, target.lower())
            os.makedirs(pose_dir, exist_ok=True)
            filename = f"{idx:02d}_{ts}.jpg"
            path = os.path.join(pose_dir, filename)
            ok = cv2.imwrite(path, best_crop)

            self.hold_count = 0
            self.buffer.clear()

            if ok:
                self.done_counts[target] += 1
                self.last_save_t = now

                # Step next if enough shots
                if self.done_counts[target] >= self.shots_per_pose:
                    self.pose_idx += 1

                return path, f"CAPTURED {target} (sharp={best_score:.0f})"

        return None, None

    def hold_progress(self):
        return float(self.hold_count) / float(max(1, self._last_need_frames))


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Head pose (LEFT/RIGHT/DOWN) + app-like guided auto capture @30fps")

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--fourcc", type=str, default="MJPG", help="MJPG often helps webcams reach 30fps")

    p.add_argument("--flip", action="store_true", help="Mirror display (selfie view)")

    # angle thresholds (hysteresis)
    p.add_argument("--yaw-enter", type=float, default=18.0)
    p.add_argument("--yaw-exit", type=float, default=12.0)
    p.add_argument("--pitch-enter", type=float, default=12.0)
    p.add_argument("--pitch-exit", type=float, default=8.0)

    # gates
    p.add_argument("--min-eye-dist", type=float, default=40.0)
    p.add_argument("--min-area-ratio", type=float, default=0.02)
    p.add_argument("--border-margin", type=float, default=0.02)
    p.add_argument("--lr-max-pitch", type=float, default=45.0)
    p.add_argument("--down-max-abs-yaw", type=float, default=45.0)

    # stability
    p.add_argument("--stable-frames", type=int, default=60, help="At 30fps, ~2s hold (user request)")
    p.add_argument("--max-angle-jump", type=float, default=2.5)
    p.add_argument("--max-center-jump", type=float, default=8.0)

    # sharpness
    p.add_argument("--min-sharp", type=float, default=30.0)

    # saving
    p.add_argument("--outdir", type=str, default="captures")
    p.add_argument("--shots-per-pose", type=int, default=1)
    p.add_argument("--cooldown", type=float, default=1.0)

    # smoothing
    p.add_argument("--alpha", type=float, default=0.75, help="EMA alpha for angles")
    p.add_argument("--alpha-rt", type=float, default=0.85, help="EMA alpha for rvec/tvec")
    p.add_argument("--alpha-anchor", type=float, default=0.80, help="EMA alpha for nose anchor")

    # sign fixes
    p.add_argument("--invert-yaw", action="store_true", help="Flip yaw sign (use if directions look reversed)")
    p.add_argument("--invert-pitch", action="store_true")

    return p.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    cfg = AppConfig(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        flip_view=bool(args.flip),
        smooth_alpha=float(args.alpha),
    )

    # session folder
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, session)
    flow = CaptureFlow(outdir=outdir,
                       stable_frames=args.stable_frames,
                       shots_per_pose=args.shots_per_pose,
                       cooldown_sec=args.cooldown)

    cap = cv2.VideoCapture(cfg.camera_index)
    set_camera_params(cap, cfg.width, cfg.height, args.fps, args.fourcc)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cfg.camera_index}")

    estimator = HeadPoseEstimator(
        max_num_faces=cfg.max_num_faces,
        refine_landmarks=cfg.refine_landmarks,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        use_extrinsic_guess=True,
        refine_pnp=True
    )

    dir_state = DirectionState()
    stability = StabilityTracker(max_angle_jump=args.max_angle_jump, max_center_jump=args.max_center_jump)

    angle_smoother = EMASmoother(alpha=float(args.alpha))
    rt_smoother = EMASmoother(alpha=float(args.alpha_rt))
    anchor_smoother = EMASmoother(alpha=float(args.alpha_anchor))

    lost_frames = 0

    # fps limiter + display
    target_dt = 1.0 / max(1e-6, float(args.fps))
    last_fps_t = time.perf_counter()
    fps_counter = 0
    fps_now = 0.0

    last_msg = ""
    last_msg_t = 0.0

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        if cfg.flip_view:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        out = estimator.estimate(frame)

        direction = "NO_FACE"
        hold_progress = 0.0
        target_pose = flow.current_target() or "DONE"
        gates_debug = ""

        if out is None:
            lost_frames += 1
            if lost_frames > 10:
                estimator.reset_guess()
                angle_smoother.reset()
                rt_smoother.reset()
                anchor_smoother.reset()
                stability.reset()
                dir_state.state = "CENTER"
        else:
            lost_frames = 0

            # Smooth rvec/tvec for stable axes
            rvec = out["rvec"].reshape(-1).astype(np.float32)
            tvec = out["tvec"].reshape(-1).astype(np.float32)
            rt = np.concatenate([rvec, tvec], axis=0)
            rt_s = rt_smoother.update(rt)
            rvec_s = rt_s[:3].reshape(3, 1).astype(np.float64)
            tvec_s = rt_s[3:].reshape(3, 1).astype(np.float64)

            # Angles
            # Flip yaw/pitch by default so LEFT turn and look DOWN map naturally on UI
            yaw, pitch, roll = -out["yaw"], -out["pitch"], out["roll"]
            if args.invert_yaw:
                yaw = -yaw
            if args.invert_pitch:
                pitch = -pitch

            ang_s = angle_smoother.update(np.array([yaw, pitch, roll], dtype=np.float32))
            yaw_s, pitch_s, roll_s = float(ang_s[0]), float(ang_s[1]), float(ang_s[2])

            # Direction with hysteresis
            direction = dir_state.update(
                yaw=yaw_s, pitch=pitch_s,
                yaw_enter=args.yaw_enter, yaw_exit=args.yaw_exit,
                pitch_enter=args.pitch_enter, pitch_exit=args.pitch_exit
            )

            # Eye distance for size gate + axis length
            pts = out["image_points"]
            eye_dist = float(np.linalg.norm(pts[2] - pts[3]))
            axis_len = int(clamp(eye_dist * 0.9, cfg.axis_len_px_min, cfg.axis_len_px_max))

            # Anchor axes to nose (exact pixel)
            nose_xy = pts[0].astype(np.float32)
            nose_xy_s = anchor_smoother.update(nose_xy)

            draw_axes(frame, rvec_s, tvec_s, out["camera_matrix"], out["dist_coeffs"],
                      axis_len_px=axis_len, origin_2d=nose_xy_s)

            #draw_pose_text(frame, yaw_s, pitch_s, roll_s, color=cfg.text_color_bgr)

            bbox = out["bbox"]
            if bbox is not None:
                # draw bbox (like common apps)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Crop buffer (no sharpness gate)
            crop = None
            sharp_ok = True
            if bbox is not None:
                bb = expand_bbox(bbox, w, h, margin=0.25)
                crop = safe_crop(frame, bb)

            # Gates (relaxed)
            gates_ok = (bbox is not None)
            target_ok = in_target_window(direction, yaw_s, pitch_s)
            stability_ok = True

            # Keep UI clean: no detailed gate reasons
            gates_debug = ""

            # Update capture flow (sequential like apps)
            saved_path, msg = flow.update(
                direction=direction,
                frame=frame,
                bbox=bbox,
                eye_dist_px=eye_dist,
                yaw=yaw_s,
                pitch=pitch_s,
                gates_ok=gates_ok,
                target_ok=target_ok,
                stability_ok=stability_ok,
                sharp_ok=sharp_ok,
                crop_for_buffer=crop
            )

            hold_progress = flow.hold_progress()

            if saved_path:
                last_msg = f"Saved: {os.path.basename(saved_path)}"
                last_msg_t = time.perf_counter()
            elif msg and msg not in ("DONE", None):
                # keep short
                pass

        # FPS measure
        fps_counter += 1
        now = time.perf_counter()
        if now - last_fps_t >= 1.0:
            fps_now = fps_counter / (now - last_fps_t)
            fps_counter = 0
            last_fps_t = now

        # ---------------------------
        # UI Panel (like common apps)
        # ---------------------------
        panel_x1, panel_y1 = 15, 15
        panel_x2, panel_y2 = 520, 210
        draw_transparent_box(frame, panel_x1, panel_y1, panel_x2, panel_y2, alpha=0.55)

        put_line(frame, "Face Capture (LEFT / RIGHT / DOWN)", 30, 45, scale=0.75, color=(255, 255, 255), thick=2)
        put_line(frame, f"Step: {min(flow.pose_idx+1, len(TARGET_POSES))}/{len(TARGET_POSES)}   Target: {target_pose}",
                 30, 75, scale=0.65, color=(0, 255, 255), thick=2)

        put_line(frame, f"Detected: {direction}", 30, 105, scale=0.65, color=(255, 255, 255), thick=2)
        put_line(frame, f"Progress: {flow.progress_text()}", 30, 135, scale=0.60, color=(255, 255, 255), thick=2)

        # Hold bar
        bar_label = "Hold steady..."
        draw_progress_bar(frame, 30, 155, 360, 18, hold_progress, label=bar_label)

        # Small info
        put_line(frame, f"FPS target: {args.fps:.0f}   measured: {fps_now:.1f}", 30, 195, scale=0.55, color=(255, 255, 255), thick=2)
        if gates_debug:
            put_line(frame, f"Check: {gates_debug}", 300, 195, scale=0.55, color=(0, 255, 0) if gates_debug == "OK" else (0, 255, 255), thick=2)

        # Capture message (short toast)
        if last_msg and (time.perf_counter() - last_msg_t) < 1.5:
            cv2.putText(frame, last_msg, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # DONE screen
        if flow.is_done():
            draw_transparent_box(frame, int(w*0.25), int(h*0.35), int(w*0.75), int(h*0.55), alpha=0.65)
            put_line(frame, "DONE! Captures saved.", int(w*0.28), int(h*0.45), scale=0.9, color=(0, 255, 0), thick=3)
            put_line(frame, f"Folder: {outdir}", int(w*0.28), int(h*0.50), scale=0.55, color=(255, 255, 255), thick=2)
            put_line(frame, "Press R to retake, ESC to quit.", int(w*0.28), int(h*0.55), scale=0.6, color=(255, 255, 255), thick=2)

        # Draw yaw/pitch/roll LAST so it stays on top of the panel (readable)
        if out is not None:
            draw_pose_text(frame, yaw_s, pitch_s, roll_s, color=cfg.text_color_bgr)

        cv2.imshow("FSA Face Capture (ESC quit, R retake)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key in (ord('r'), ord('R')):
            flow.reset()
            estimator.reset_guess()
            angle_smoother.reset()
            rt_smoother.reset()
            anchor_smoother.reset()
            stability.reset()
            dir_state.state = "CENTER"
            last_msg = "Retake: reset"
            last_msg_t = time.perf_counter()

        # FPS limiter
        elapsed = time.perf_counter() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
