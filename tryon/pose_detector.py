"""Pose detection built on MediaPipe Pose.

Detects body landmarks per frame and applies exponential moving average
(EMA) smoothing so the garment overlay doesn't jitter between frames.
"""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Pose landmark indices we care about
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_HIP = 23
RIGHT_HIP = 24

TORSO_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]


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
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.smoothing = smoothing
        self.min_visibility = min_visibility
        self._prev: dict[int, np.ndarray] | None = None

    def detect(self, frame_bgr: np.ndarray) -> dict[int, np.ndarray] | None:
        """Run pose estimation on a BGR frame.

        Returns:
            Dict mapping landmark index -> np.array([x, y]) in pixel
            coordinates, or None when no reliable body is in view.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            self._prev = None
            return None

        landmarks = result.pose_landmarks.landmark
        points: dict[int, np.ndarray] = {}
        for idx in TORSO_LANDMARKS + [LEFT_ELBOW, RIGHT_ELBOW]:
            lm = landmarks[idx]
            if idx in TORSO_LANDMARKS and lm.visibility < self.min_visibility:
                # Torso must be fully visible for a believable overlay.
                self._prev = None
                return None
            points[idx] = np.array([lm.x * w, lm.y * h], dtype=np.float32)

        points = self._smooth(points)
        return points

    def _smooth(self, points: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Blend current landmarks with the previous frame's (EMA)."""
        if self._prev is None:
            self._prev = points
            return points
        a = self.smoothing
        smoothed = {
            idx: a * pt + (1.0 - a) * self._prev.get(idx, pt)
            for idx, pt in points.items()
        }
        self._prev = smoothed
        return smoothed

    def close(self):
        self._pose.close()
