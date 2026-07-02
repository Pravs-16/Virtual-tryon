"""Garment overlay engine.

Takes a garment PNG (with alpha channel) plus detected torso keypoints,
warps the garment to fit the body with a perspective transform, and
alpha-blends it onto the camera frame. Perspective warping (instead of a
simple scale + paste) is what makes the garment lean and turn with the
wearer, which reads as far more realistic.
"""

import cv2
import numpy as np

from .pose_detector import LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP


class GarmentOverlay:
    """Warps and blends a garment image onto a frame using torso keypoints."""

    def __init__(
        self,
        shoulder_pad: float = 0.42,
        neck_offset: float = 0.14,
        hip_pad: float = 0.28,
        length_scale: float = 1.12,
    ):
        """
        Args:
            shoulder_pad: extra garment width beyond the shoulder span
                (fraction of shoulder width per side). Shirts hang wider
                than the skeleton, so 0 looks painted-on.
            neck_offset: how far above the shoulder line the collar sits
                (fraction of torso height).
            hip_pad: extra width at the hem (fraction of hip span per side).
            length_scale: garment length relative to shoulder->hip distance.
        """
        self.shoulder_pad = shoulder_pad
        self.neck_offset = neck_offset
        self.hip_pad = hip_pad
        self.length_scale = length_scale

    def apply(
        self,
        frame: np.ndarray,
        garment_rgba: np.ndarray,
        keypoints: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Overlay the garment onto the frame in place and return it."""
        dst_quad = self._body_quad(keypoints)
        if dst_quad is None:
            return frame

        h, w = frame.shape[:2]
        gh, gw = garment_rgba.shape[:2]

        # Source quad: the garment image corners
        # (top-left, top-right, bottom-right, bottom-left).
        src_quad = np.array(
            [[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
        warped = cv2.warpPerspective(
            garment_rgba,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return self._alpha_blend(frame, warped)

    def _body_quad(self, kp: dict[int, np.ndarray]) -> np.ndarray | None:
        """Compute the destination quad on the body for the garment corners.

        Order: top-left, top-right, bottom-right, bottom-left — matching how
        we read the garment image (left of image = viewer's left = the
        subject's right side in a mirrored feed, which is what we want).
        """
        try:
            ls, rs = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]
            lh, rh = kp[LEFT_HIP], kp[RIGHT_HIP]
        except KeyError:
            return None

        shoulder_vec = ls - rs                # right shoulder -> left shoulder
        shoulder_width = np.linalg.norm(shoulder_vec)
        if shoulder_width < 20:               # too far away / bad detection
            return None

        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        torso_vec = hip_mid - shoulder_mid
        torso_len = np.linalg.norm(torso_vec)
        if torso_len < 20:
            return None

        torso_dir = torso_vec / torso_len     # unit vector, shoulders -> hips
        shoulder_dir = shoulder_vec / shoulder_width

        # Collar line sits slightly above the shoulders.
        top_mid = shoulder_mid - torso_dir * (torso_len * self.neck_offset)
        top_half = shoulder_width * (0.5 + self.shoulder_pad)
        top_left = top_mid + shoulder_dir * top_half     # subject's left
        top_right = top_mid - shoulder_dir * top_half    # subject's right

        # Hem sits below the hips.
        hem_mid = shoulder_mid + torso_dir * (torso_len * self.length_scale)
        hip_width = max(np.linalg.norm(lh - rh), shoulder_width * 0.7)
        hem_half = hip_width * (0.5 + self.hip_pad)
        hem_left = hem_mid + shoulder_dir * hem_half
        hem_right = hem_mid - shoulder_dir * hem_half

        # In a mirrored selfie view the subject's left appears on the
        # right of the image; MediaPipe already accounts for this because
        # we mirror the frame before detection, so the quad maps directly.
        return np.array(
            [top_right, top_left, hem_left, hem_right], dtype=np.float32
        )

    @staticmethod
    def _alpha_blend(frame: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """Blend an RGBA overlay onto a BGR frame using its alpha channel."""
        alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        # Soften the garment edge a touch so it doesn't look cut out.
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[:, :, np.newaxis]
        overlay_bgr = overlay_rgba[:, :, :3].astype(np.float32)
        base = frame.astype(np.float32)
        blended = overlay_bgr * alpha + base * (1.0 - alpha)
        return blended.astype(np.uint8)
