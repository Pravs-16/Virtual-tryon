"""Pose detection built on MediaPipe Pose.

Detects body landmarks per frame and applies exponential moving average
(EMA) smoothing so the garment overlay doesn't jitter between frames.
Also returns per-landmark depth (z) and a person segmentation mask,
which power arm occlusion in the overlay stage.
"""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose landmark indices we care about
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_HIP = 23
RIGHT_HIP = 24

TORSO_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
ARM_LANDMARKS = [
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_INDEX, RIGHT_INDEX,
]


class PoseResult:
    """Smoothed pose data for a single frame."""

    __slots__ = ("points", "z", "mask")

    def __init__(self, points, z, mask):
        self.points = points  # {landmark: np.array([x, y])} in pixels
        self.z = z            # {landmark: float} depth; more negative = closer
        self.mask = mask      # float32 HxW person mask in [0, 1], or None


class PoseDetector:
    """Wraps MediaPipe Pose and returns smoothed pixel-space keypoints."""

    def __init__(self, smoothing: float = 0.35, min_visibility: float = 0.5):
        """
        Args:
            smoothing: EMA factor in [0, 1]. Higher = snappier, lower = smoother.
            min_visibility: minimum landmark visibility to trust a detection.
        """
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.smoothing = smoothing
        self.min_visibility = min_visibility
        self._prev_pts: dict[int, np.ndarray] | None = None
        self._prev_z: dict[int, float] | None = None

    def detect(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Run pose estimation on a BGR frame.

        Returns a PoseResult, or None when no reliable body is in view.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            self._prev_pts = None
            self._prev_z = None
            return None

        landmarks = result.pose_landmarks.landmark
        points: dict[int, np.ndarray] = {}
        z: dict[int, float] = {}

        for idx in TORSO_LANDMARKS:
            lm = landmarks[idx]
            if lm.visibility < self.min_visibility:
                # Torso must be fully visible for a believable overlay.
                self._prev_pts = None
                self._prev_z = None
                return None
            points[idx] = np.array([lm.x * w, lm.y * h], dtype=np.float32)
            z[idx] = lm.z

        for idx in ARM_LANDMARKS:
            lm = landmarks[idx]
            if lm.visibility > 0.3:
                points[idx] = np.array([lm.x * w, lm.y * h], dtype=np.float32)
                z[idx] = lm.z

        points = self._smooth_points(points)
        z = self._smooth_z(z)
        mask = getattr(result, "segmentation_mask", None)
        return PoseResult(points, z, mask)

    def _smooth_points(self, points):
        if self._prev_pts is None:
            self._prev_pts = points
            return points
        a = self.smoothing
        smoothed = {
            idx: a * pt + (1.0 - a) * self._prev_pts.get(idx, pt)
            for idx, pt in points.items()
        }
        self._prev_pts = smoothed
        return smoothed

    def _smooth_z(self, z):
        if self._prev_z is None:
            self._prev_z = z
            return z
        a = self.smoothing
        smoothed = {
            idx: a * v + (1.0 - a) * self._prev_z.get(idx, v)
            for idx, v in z.items()
        }
        self._prev_z = smoothed
        return smoothed

    def close(self):
        self._pose.close()
