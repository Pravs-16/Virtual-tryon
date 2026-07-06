# Virtual Try-On System

Real-time virtual clothing try-on in the browser. Turn on your laptop camera, stand in frame, and garments are overlaid on your body live — like a fitting-room mirror. Built with **MediaPipe Pose** for body keypoint detection, **OpenCV** for perspective warping and alpha blending, and **Flask** for browser-based streaming.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-black) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## How it works

1. **Capture** — a background thread reads the webcam continuously (`tryon/camera.py`), so the stream never blocks on hardware.
2. **Detect** — each frame is mirrored (fitting-room behaviour) and passed to MediaPipe Pose, which returns 33 body landmarks. We use the shoulders, hips, and elbows (`tryon/pose_detector.py`). An exponential moving average smooths landmarks across frames so the garment doesn't jitter.
3. **Fit** — from the four torso keypoints we compute a destination quadrilateral: a collar line slightly above the shoulders (padded outward, because clothes hang wider than the skeleton) and a hem line below the hips (`tryon/overlay.py`).
4. **Warp** — the garment PNG's four corners are mapped onto that body quad with `cv2.getPerspectiveTransform` + `warpPerspective`. Because it's a perspective warp rather than a flat paste, the garment leans, turns, and scales with you.
5. **Blend** — the warped garment is composited using its alpha channel, with a slight Gaussian soften on the edge so it doesn't look cut out. Two realism passes run here: **adaptive lighting** (the garment's brightness is matched to the scene inside the torso region, so it dims in a dark room) and **arm occlusion** (when MediaPipe's depth values say your wrist is in front of your chest, the forearm/hand region — intersected with the person segmentation mask — is restored on top of the garment, so your hand passes in front of the shirt).
6. **Stream** — frames are served as MJPEG to the browser, where a small UI lets you switch garments and toggle tracking dots.

```
Camera thread ──> mirror ──> MediaPipe Pose ──> torso quad ──> perspective warp
                                                                     │
Browser <── MJPEG stream <── JPEG encode <── alpha blend <───────────┘
```

## Quick start

**Prerequisites:** Python 3.10+ and a webcam (the camera is read server-side by OpenCV, so run the app on the machine the camera is attached to).

```bash
git clone https://github.com/Pravs-16/virtual-tryon.git
cd virtual-tryon
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_garments.py    # creates sample garments (only needed once)
python app.py
```

Open **http://localhost:5000**, allow nothing (the camera is read server-side by OpenCV), stand back so your shoulders and hips are visible, and pick a garment from the rack.

> **Tip:** good, even lighting and a plain background noticeably improve tracking stability.

## Adding your own garments

Drop any PNG **with a transparent background** into `static/garments/` and restart the app — it appears in the rack automatically. For the most realistic results:

- Use a front-facing product photo (ghost-mannequin shots work best).
- Run it through the prep script, which removes the background, crops, and sizes it automatically:

```bash
pip install "rembg[cpu]"   # one-time, for background removal
python scripts/prepare_garment.py path/to/photo.jpg my_shirt
```

- Or do it manually: remove the background (remove.bg or Photoshop) and crop tightly — the image's top edge should be the collar, the bottom edge the hem, and the left/right edges the sleeve tips, because the warp maps image corners to body corners.

## Project structure

```
virtual-tryon/
├── app.py                    # Flask app: MJPEG stream + garment API
├── tryon/
│   ├── camera.py             # threaded webcam capture
│   ├── pose_detector.py      # MediaPipe Pose wrapper + EMA smoothing
│   ├── overlay.py            # perspective warp + alpha blending
│   └── garments.py           # garment catalog (auto-loads PNGs)
├── scripts/
│   ├── generate_garments.py  # generates the sample tee/hoodie PNGs
│   └── prepare_garment.py    # converts a real garment photo into a try-on PNG
├── static/
│   ├── garments/             # garment PNGs (RGBA)
│   ├── css/style.css
│   └── js/main.js
├── templates/index.html
└── requirements.txt
```

## API

| Route | Method | Description |
|---|---|---|
| `/` | GET | Fitting-room UI |
| `/video_feed` | GET | MJPEG stream with overlay |
| `/api/garments` | GET | List available garments + current selection |
| `/api/garment` | POST `{"name": "..."}` | Switch garment |
| `/api/toggle` | POST `{"enabled": bool, "show_landmarks": bool}` | Overlay / tracking toggles |

## Troubleshooting

- **"Could not open the camera"** — another app is using it, or your camera isn't index 0. Change `Camera(index=1)` in `app.py`.
- **Garment doesn't appear** — make sure your full torso (shoulders *and* hips) is in frame; detection deliberately hides the garment when the torso is partly out of view.
- **Laggy stream** — lower the capture resolution in `Camera(width=..., height=...)`, or set `model_complexity=0` in `pose_detector.py` for a faster (slightly less accurate) model.

## Roadmap

- Cloth deformation along elbow keypoints for sleeves
- Photorealistic snapshot mode using a diffusion-based try-on model
- Size recommendation from estimated shoulder/hip measurements

## License

MIT
