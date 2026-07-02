"""Virtual Try-On — Flask entry point.

Streams the laptop camera as MJPEG with a garment perspective-warped
onto the wearer in real time. Switch garments from the browser UI.

Run:  python app.py   then open http://localhost:5000
"""

import threading

import cv2
from flask import Flask, Response, jsonify, render_template, request

from tryon import Camera, GarmentCatalog, GarmentOverlay, PoseDetector

app = Flask(__name__)

camera = Camera(index=0)
detector = PoseDetector()
overlay = GarmentOverlay()
catalog = GarmentCatalog()

_state_lock = threading.Lock()
_state = {
    "garment": catalog.first(),
    "enabled": True,
    "show_landmarks": False,
}


def _render_frame():
    """Grab a frame, mirror it, run pose detection, overlay the garment."""
    frame = camera.read()
    if frame is None:
        return None

    frame = cv2.flip(frame, 1)  # mirror: behaves like a fitting-room mirror

    with _state_lock:
        garment_name = _state["garment"]
        enabled = _state["enabled"]
        show_landmarks = _state["show_landmarks"]

    keypoints = detector.detect(frame)

    if enabled and keypoints is not None and garment_name:
        garment = catalog.get(garment_name)
        if garment is not None:
            frame = overlay.apply(frame, garment, keypoints)

    if show_landmarks and keypoints is not None:
        for pt in keypoints.values():
            cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 160), -1)

    return frame


def _mjpeg_generator():
    while True:
        frame = _render_frame()
        if frame is None:
            continue
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/garments")
def list_garments():
    with _state_lock:
        current = _state["garment"]
    return jsonify({"garments": catalog.names(), "current": current})


@app.route("/api/garment", methods=["POST"])
def set_garment():
    name = (request.get_json(silent=True) or {}).get("name")
    if name not in catalog.names():
        return jsonify({"error": "Unknown garment"}), 404
    with _state_lock:
        _state["garment"] = name
    return jsonify({"ok": True, "current": name})


@app.route("/api/toggle", methods=["POST"])
def toggle():
    body = request.get_json(silent=True) or {}
    with _state_lock:
        if "enabled" in body:
            _state["enabled"] = bool(body["enabled"])
        if "show_landmarks" in body:
            _state["show_landmarks"] = bool(body["show_landmarks"])
        snapshot = dict(_state)
    return jsonify(snapshot)


if __name__ == "__main__":
    if not camera.start():
        raise SystemExit(
            "Could not open the camera. Close other apps using it, or set "
            "a different index in Camera(index=...) inside app.py."
        )
    try:
        # threaded=True lets the MJPEG stream and the API run concurrently.
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        camera.stop()
        detector.close()
