import argparse
import os
import time
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import AppConfig
from .pose import HeadPoseEstimator
from .smoother import EMASmoother
from .draw import draw_axes, draw_pose_text


TARGET_POSES = ["LEFT", "RIGHT", "DOWN"]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def set_camera_params(cap, width, height, fps, fourcc="MJPG"):
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


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


class DirectionState:
    def __init__(self):
        self.state = "CENTER"

    def update(self, yaw, pitch, yaw_enter=15, yaw_exit=9, pitch_enter=9, pitch_exit=5):
        s = self.state
        if s == "LEFT" and yaw > -yaw_exit:
            s = "CENTER"
        elif s == "RIGHT" and yaw < yaw_exit:
            s = "CENTER"
        elif s == "DOWN" and pitch < pitch_exit:
            s = "CENTER"

        if s == "CENTER":
            if yaw <= -yaw_enter:
                s = "LEFT"
            elif yaw >= yaw_enter:
                s = "RIGHT"
            elif pitch >= pitch_enter:
                s = "DOWN"

        self.state = s
        return s


def direction_gate(direction, yaw, pitch, lr_max_pitch=18.0, down_max_abs_yaw=18.0):
    if direction in ("LEFT", "RIGHT"):
        return pitch < lr_max_pitch
    if direction == "DOWN":
        return abs(yaw) < down_max_abs_yaw
    return True


def in_target_window(direction, yaw, pitch):
    # relaxed window
    if direction == "LEFT":
        return (-60 <= yaw <= -15) and (pitch < 20)
    if direction == "RIGHT":
        return (15 <= yaw <= 60) and (pitch < 20)
    if direction == "DOWN":
        return (10 <= pitch <= 45) and (abs(yaw) < 20)
    return False


class StabilityTracker:
    def __init__(self, max_angle_jump=4.0, max_center_jump=14.0):
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


class BestFrameBuffer:
    def __init__(self, maxlen=15):
        self.buf = deque(maxlen=int(maxlen))

    def clear(self):
        self.buf.clear()

    def push(self, crop):
        self.buf.append((sharpness_score(crop), crop))

    def best(self):
        if not self.buf:
            return None, 0.0
        score, crop = max(self.buf, key=lambda x: x[0])
        return crop, float(score)

    def __len__(self):
        return len(self.buf)


class CaptureFlow:
    def __init__(self, outdir, stable_frames=8, shots_per_pose=1, cooldown_sec=0.7):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        self.stable_frames = int(stable_frames)
        self.shots_per_pose = int(shots_per_pose)
        self.cooldown_sec = float(cooldown_sec)

        self.pose_idx = 0
        self.done = {p: 0 for p in TARGET_POSES}
        self.hold_count = 0
        self.last_save_t = 0.0
        self.buffer = BestFrameBuffer(maxlen=15)

    def current_target(self):
        if self.pose_idx >= len(TARGET_POSES):
            return None
        return TARGET_POSES[self.pose_idx]

    def is_done(self):
        return self.pose_idx >= len(TARGET_POSES)

    def progress_text(self):
        parts = []
        for p in TARGET_POSES:
            ok = self.done[p] >= self.shots_per_pose
            parts.append(f"{p}:{'OK' if ok else '--'} {self.done[p]}/{self.shots_per_pose}")
        return "  ".join(parts)

    def reset(self):
        self.pose_idx = 0
        self.done = {p: 0 for p in TARGET_POSES}
        self.hold_count = 0
        self.last_save_t = 0.0
        self.buffer.clear()

    def hold_progress(self, need_frames):
        return float(self.hold_count) / float(max(1, need_frames))

    def update(self, direction, frame, bbox, eye_dist, yaw, pitch,
               gates_ok, target_ok, stability_ok, sharp_ok, crop_for_buffer, min_sharp=70.0):
        now = time.perf_counter()
        if self.is_done():
            return None, "DONE"

        target = self.current_target()
        if target is None:
            return None, "DONE"

        # cooldown
        if now - self.last_save_t < self.cooldown_sec:
            self.hold_count = 0
            self.buffer.clear()
            return None, "COOLDOWN"

        # Soft-window: if not in target window, require longer hold
        need_frames = self.stable_frames if target_ok else int(self.stable_frames * 1.5)

        if (direction == target and gates_ok and stability_ok and sharp_ok):
            self.hold_count += 1
            if crop_for_buffer is not None:
                self.buffer.push(crop_for_buffer)
        else:
            self.hold_count = 0
            self.buffer.clear()

        if self.hold_count >= need_frames and len(self.buffer) >= min(5, need_frames):
            best_crop, best_score = self.buffer.best()
            self.hold_count = 0
            self.buffer.clear()

            if best_crop is None:
                return None, None

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            idx = self.done[target] + 1
            filename = f"{target.lower()}_{idx:02d}_{ts}.jpg"
            path = os.path.join(self.outdir, filename)
            ok = cv2.imwrite(path, best_crop)
            if ok:
                self.done[target] += 1
                self.last_save_t = now
                if self.done[target] >= self.shots_per_pose:
                    self.pose_idx += 1
                return path, f"CAPTURED {target} (sharp={best_score:.0f})"

        return None, None


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = {}
        self.running = True


def build_app(shared: SharedState):
    app = FastAPI(title="FSA Head Pose Capture API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def home():
        return """
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>
        <body style="font-family:Arial;background:#111;color:#eee">
          <h2>FSA Capture Live</h2>
          <img src="/mjpeg" style="width:100%;max-width:720px;border:1px solid #333;border-radius:8px"/>
          <pre id="st" style="white-space:pre-wrap;background:#1b1b1b;padding:10px;border-radius:8px"></pre>
          <button onclick="fetch('/reset',{method:'POST'})" style="padding:10px 14px">Reset</button>
          <script>
            async function tick(){
              const r = await fetch('/status'); 
              const j = await r.json();
              document.getElementById('st').textContent = JSON.stringify(j,null,2);
            }
            setInterval(tick, 400);
            tick();
          </script>
        </body></html>
        """

    @app.get("/status")
    def status():
        with shared.lock:
            return JSONResponse(shared.status or {})

    @app.post("/reset")
    def reset():
        with shared.lock:
            shared.status["cmd_reset"] = True
        return {"ok": True}

    @app.get("/snapshot.jpg")
    def snapshot():
        with shared.lock:
            data = shared.jpeg
        if data is None:
            return JSONResponse({"error": "no frame yet"}, status_code=503)
        return StreamingResponse(iter([data]), media_type="image/jpeg")

    @app.get("/mjpeg")
    def mjpeg():
        def gen():
            while shared.running:
                with shared.lock:
                    frame = shared.jpeg
                if frame is None:
                    time.sleep(0.03)
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.03)
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--flip", action="store_true")

    p.add_argument("--min-eye-dist", type=float, default=60.0)
    p.add_argument("--min-area-ratio", type=float, default=0.04)
    p.add_argument("--border-margin", type=float, default=0.03)
    p.add_argument("--min-sharp", type=float, default=70.0)

    p.add_argument("--stable-frames", type=int, default=8)
    p.add_argument("--shots-per-pose", type=int, default=1)
    p.add_argument("--outdir", default="captures_api")

    p.add_argument("--invert-yaw", action="store_true")
    p.add_argument("--invert-pitch", action="store_true")

    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--alpha-rt", type=float, default=0.85)
    p.add_argument("--alpha-anchor", type=float, default=0.80)
    return p.parse_args()


def run_engine(shared: SharedState, args):
    cfg = AppConfig(camera_index=args.camera, width=args.width, height=args.height, flip_view=args.flip)

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, session)
    flow = CaptureFlow(outdir=outdir, stable_frames=args.stable_frames, shots_per_pose=args.shots_per_pose)

    cap = cv2.VideoCapture(cfg.camera_index)
    set_camera_params(cap, cfg.width, cfg.height, args.fps, args.fourcc)

    estimator = HeadPoseEstimator(use_extrinsic_guess=True, refine_pnp=True)
    dir_state = DirectionState()
    stability = StabilityTracker()

    angle_smoother = EMASmoother(alpha=float(args.alpha))
    rt_smoother = EMASmoother(alpha=float(args.alpha_rt))
    anchor_smoother = EMASmoother(alpha=float(args.alpha_anchor))

    target_dt = 1.0 / max(1e-6, float(args.fps))

    last_fps_t = time.perf_counter()
    fps_counter = 0
    fps_now = 0.0

    while shared.running:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        if cfg.flip_view:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        out = estimator.estimate(frame)

        direction = "NO_FACE"
        gates_debug = ""
        saved_path = None

        if out is not None:
            rvec = out["rvec"].reshape(-1).astype(np.float32)
            tvec = out["tvec"].reshape(-1).astype(np.float32)
            rt = np.concatenate([rvec, tvec], axis=0)
            rt_s = rt_smoother.update(rt)
            rvec_s = rt_s[:3].reshape(3, 1).astype(np.float64)
            tvec_s = rt_s[3:].reshape(3, 1).astype(np.float64)

            yaw, pitch, roll = out["yaw"], out["pitch"], out["roll"]
            if args.invert_yaw:
                yaw = -yaw
            if args.invert_pitch:
                pitch = -pitch

            ang_s = angle_smoother.update(np.array([yaw, pitch, roll], dtype=np.float32))
            yaw_s, pitch_s, roll_s = float(ang_s[0]), float(ang_s[1]), float(ang_s[2])

            direction = dir_state.update(yaw_s, pitch_s)

            pts = out["image_points"]
            eye_dist = float(np.linalg.norm(pts[2] - pts[3]))
            axis_len = int(clamp(eye_dist * 0.9, 80, 180))

            nose_xy = pts[0].astype(np.float32)
            nose_xy_s = anchor_smoother.update(nose_xy)

            draw_axes(frame, rvec_s, tvec_s, out["camera_matrix"], out["dist_coeffs"], axis_len_px=axis_len, origin_2d=nose_xy_s)
            draw_pose_text(frame, yaw_s, pitch_s, roll_s)

            bbox = out["bbox"]
            crop = None
            sharp_ok = False

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                bb = expand_bbox(bbox, w, h, margin=0.25)
                crop = safe_crop(frame, bb)
                if crop is not None:
                    sharp_ok = (sharpness_score(crop) >= float(args.min_sharp))

            gates_ok = (
                bbox is not None
                and eye_dist >= float(args.min_eye_dist)
                and ((x2-x1)*(y2-y1) >= w*h*float(args.min_area_ratio))
                and direction_gate(direction, yaw_s, pitch_s)
            )
            target_ok = in_target_window(direction, yaw_s, pitch_s)
            stability_ok = stability.ok(yaw_s, pitch_s, bbox)

            if not gates_ok: gates_debug += "gates "
            if not target_ok: gates_debug += "window "
            if not stability_ok: gates_debug += "stable "
            if not sharp_ok: gates_debug += "sharp "
            if gates_debug.strip() == "":
                gates_debug = "OK"

            saved_path, _ = flow.update(
                direction, frame, bbox, eye_dist, yaw_s, pitch_s,
                gates_ok=gates_ok, target_ok=target_ok, stability_ok=stability_ok,
                sharp_ok=sharp_ok, crop_for_buffer=crop, min_sharp=args.min_sharp
            )

        # fps measure
        fps_counter += 1
        now = time.perf_counter()
        if now - last_fps_t >= 1.0:
            fps_now = fps_counter / (now - last_fps_t)
            fps_counter = 0
            last_fps_t = now

        # UI text
        target = flow.current_target() or "DONE"
        cv2.putText(frame, f"Target: {target}   Detected: {direction}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Progress: {flow.progress_text()}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Check: {gates_debug}   FPS:{fps_now:.1f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0) if gates_debug=="OK" else (0,255,255), 2, cv2.LINE_AA)
        if saved_path:
            cv2.putText(frame, f"SAVED {os.path.basename(saved_path)}", (20, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        # encode jpeg for API
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with shared.lock:
                shared.jpeg = jpg.tobytes()
                st = shared.status
                # reset command
                if st.get("cmd_reset"):
                    flow.reset()
                    st["cmd_reset"] = False

                shared.status = {
                    "yaw": yaw_s if out is not None else None,
                    "pitch": pitch_s if out is not None else None,
                    "roll": roll_s if out is not None else None,

                    "target": target,
                    "detected": direction,
                    "progress": flow.progress_text(),
                    "fps": fps_now,
                    "check": gates_debug,
                    "outdir": outdir,
                    "last_saved": os.path.basename(saved_path) if saved_path else st.get("last_saved"),
                    "cmd_reset": st.get("cmd_reset", False),
                }

        elapsed = time.perf_counter() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    cap.release()


def main():
    args = parse_args()
    shared = SharedState()

    t = threading.Thread(target=run_engine, args=(shared, args), daemon=True)
    t.start()

    app = build_app(shared)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    shared.running = False


if __name__ == "__main__":
    main()
