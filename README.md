# FSA Head Pose Capture
Estimate head pose (yaw/pitch/roll) with **MediaPipe Face Mesh + OpenCV solvePnP**, draw 3-axis arrows on the face, and auto-capture three poses (LEFT / RIGHT / DOWN). Defaults are relaxed so it’s easy to complete a session.

## What it does
- Detects a face, estimates yaw/pitch/roll each frame.
- Anchors the 3D axes at the nose pixel; smoothing on angles, pose vectors, and nose anchor for steady display.
- Guides through LEFT → RIGHT → DOWN poses; holds for ~2s (60 frames by default) then saves the sharpest buffered frame for that pose.
- Saves into timestamped session folders, further split by pose: `captures/YYYYMMDD_HHMMSS/{left|right|down}/01_*.jpg`.

## Requirements
- Python 3.9–3.12 (mediapipe compatible).
- A webcam reachable as index `0` (or set `--camera`).

## Install
```bash
cd D:\TinaSoft\Three-Dided-Sngle-Model\Fsa_Headpose_Project
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
```
If you prefer without install, set `PYTHONPATH=src` before running: `$env:PYTHONPATH=(Resolve-Path src).Path` (PowerShell) or `export PYTHONPATH=$(pwd)/src` (bash).

## Run (desktop GUI)
```bash
fsa-headpose --camera 0 --flip
```
- Keys: `ESC` quit, `R` restart capture flow.
- Default tolerance is very loose; only face + direction is needed. Hold each pose ~2s to auto-save.
- If you want faster: `fsa-headpose --stable-frames 30` (~1s hold).

## Run API server (FastAPI + MJPEG)
```bash
fsa-headpose-api --camera 0 --flip --port 8000
```
Visit `http://localhost:8000/` to view the stream. Snapshots and status are available via `/snapshot.jpg` and `/status`. Captures land in `captures_api/YYYYMMDD_HHMMSS/{left|right|down}/`.

## Output layout
- Desktop: `captures/<session>/<pose>/<index>_<timestamp>.jpg`
- API:    `captures_api/<session>/<pose>/<index>_<timestamp>.jpg`
Each session folder is created per run; inside, images are grouped by pose.

## Important defaults (desktop & API)
- Relaxed gates: no sharpness/area/border gating; just needs a detected face.
- Stable hold: 60 frames (~2s) for each pose. Adjustable via `--stable-frames`.
- Yaw/pitch sign: flipped by default so turning left/down shows LEFT/DOWN naturally. To restore original sign, add `--invert-yaw` or `--invert-pitch`.

## Useful flags
- `--flip` : mirror view (selfie).
- `--camera N` : choose webcam index.
- `--width/--height/--fps` : request capture parameters (tries but depends on camera).
- `--alpha/--alpha-rt/--alpha-anchor` : smoothing strengths (0–1, closer to 1 = smoother).
- `--shots-per-pose N` : save multiple images per pose in sequence.

## Troubleshooting
- **No images saved**: ensure app ran for at least the hold time while the pose label matched; files go under `captures/<timestamp>/...`. Check write permissions.
- **Directions look reversed**: add `--invert-yaw` (or `--invert-pitch` if down/up looks flipped).
- **Camera busy / black screen**: close other camera apps; verify index with `--camera 1`, etc.

## Project layout
```
src/fsa_headpose/
  main.py        # desktop GUI entry
  api_server.py  # FastAPI streaming entry
  pose.py        # MediaPipe Face Mesh + solvePnP
  draw.py        # axis and text overlays
  smoother.py    # EMA utilities
  config.py      # basic defaults
```
