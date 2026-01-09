# FSA Head Pose (MediaPipe + SolvePnP) — 3-axis arrows (Nose-anchored)

This project estimates head pose (yaw/pitch/roll) from a webcam using **MediaPipe Face Mesh** and **OpenCV solvePnP**,
and overlays **3 axis arrows** (X=Red, Y=Green, Z=Blue).

**Improvements in this version**
- Axes **start exactly at the nose landmark pixel** (anchor), like the sample image.
- Less jitter via:
  - `solvePnP` using previous pose as **extrinsic initial guess**
  - optional `solvePnPRefineLM/VVS` (if available)
  - EMA smoothing for `rvec/tvec` and nose anchor

## Quick start

### 1) Create venv (recommended)
**Windows**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run
```bash
python -m fsa_headpose.main --camera 0
```

Press **ESC** to quit.

## Useful flags
Mirror display:
```bash
python -m fsa_headpose.main --flip
```

Extra smoothing (less jitter):
```bash
python -m fsa_headpose.main --alpha-rt 0.90 --alpha-anchor 0.85
```
