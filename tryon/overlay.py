"""Garment overlay engine.

Takes a garment PNG (with alpha channel) plus detected torso keypoints,
warps the garment to fit the body with a perspective transform, and
alpha-blends it onto the camera frame.

Realism layers on top of the basic warp:
  * Adaptive lighting — the garment's brightness is matched to the scene
    so a shirt in a dim room looks dim.
  * Arm occlusion (ArmOccluder) — when your forearm/hand is in front of
    your chest, the original arm pixels are restored over the garment,
    so your hand passes IN FRONT of the shirt instead of under it.
"""

import cv2
import numpy as np

from .pose_detector import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_INDEX, RIGHT_INDEX,
)


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

    def apply(self, frame, garment_rgba, pose) -> np.ndarray:
        """Overlay the garment onto the frame and return it.

        `pose` is a PoseResult (or a bare {landmark: xy} dict in tests).
        """
        kp = pose.points if hasattr(pose, "points") else pose
        dst_quad = self._body_quad(kp)
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
        warped = self._match_lighting(frame, warped, dst_quad, garment_rgba)
        return self._alpha_blend(frame, warped)

    def _body_quad(self, kp) -> np.ndarray | None:
        """Compute the destination quad on the body for the garment corners."""
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

        return np.array(
            [top_right, top_left, hem_left, hem_right], dtype=np.float32
        )

    @staticmethod
    def _match_lighting(frame, warped, dst_quad, garment_rgba) -> np.ndarray:
        """Scale the garment's brightness toward the scene's brightness.

        We compare the average luma of the frame inside the torso quad
        against the garment's own average luma, and gain the garment
        toward the scene (clamped, so it never blows out or goes black).
        """
        h, w = frame.shape[:2]
        quad_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(quad_mask, dst_quad.astype(np.int32), 255)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_luma = cv2.mean(gray, mask=quad_mask)[0]

        alpha_mask = (garment_rgba[:, :, 3] > 10).astype(np.uint8) * 255
        b, g, r, _ = cv2.mean(garment_rgba, mask=alpha_mask)
        ref_luma = 0.114 * b + 0.587 * g + 0.299 * r
        if ref_luma < 1.0 or scene_luma < 1.0:
            return warped

        gain = float(np.clip(scene_luma / ref_luma, 0.55, 1.35))
        out = warped.copy()
        out[:, :, :3] = np.clip(
            out[:, :, :3].astype(np.float32) * gain, 0, 255
        ).astype(np.uint8)
        return out

    @staticmethod
    def _alpha_blend(frame, overlay_rgba) -> np.ndarray:
        """Blend an RGBA overlay onto a BGR frame using its alpha channel."""
        alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        # Soften the garment edge a touch so it doesn't look cut out.
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)[:, :, np.newaxis]
        overlay_bgr = overlay_rgba[:, :, :3].astype(np.float32)
        base = frame.astype(np.float32)
        blended = overlay_bgr * alpha + base * (1.0 - alpha)
        return blended.astype(np.uint8)


class ArmOccluder:
    """Restores forearms/hands over the garment when they're in front of it.

    MediaPipe gives each landmark a relative depth (z, more negative =
    closer to the camera). When a wrist is meaningfully closer than the
    chest plane, we rebuild that forearm as a thick capsule, intersect
    it with the person segmentation mask (so we only restore real body
    pixels, never background), and paint the original camera pixels back
    on top of the garment. Hysteresis on the depth test stops flicker.
    """

    ON_THRESHOLD = 0.10   # wrist this much closer than chest -> in front
    OFF_THRESHOLD = 0.04  # must retreat past this to go behind again

    _SIDES = {
        "L": (LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX),
        "R": (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX),
    }

    def __init__(self):
        self._in_front = {"L": False, "R": False}

    def apply(self, original, rendered, pose) -> np.ndarray:
        """Composite original arm pixels over the rendered frame."""
        mask = getattr(pose, "mask", None)
        if mask is None:
            return rendered

        chest = [
            pose.z[i]
            for i in (LEFT_SHOULDER, RIGHT_SHOULDER)
            if i in pose.z
        ]
        if not chest:
            return rendered
        chest_z = float(np.mean(chest))

        h, w = rendered.shape[:2]
        occ = np.zeros((h, w), dtype=np.float32)
        drew = False

        for side, (elbow_i, wrist_i, index_i) in self._SIDES.items():
            if (
                wrist_i not in pose.points
                or elbow_i not in pose.points
                or wrist_i not in pose.z
            ):
                self._in_front[side] = False
                continue

            closer_by = chest_z - pose.z[wrist_i]  # +ve => wrist nearer camera
            if self._in_front[side]:
                self._in_front[side] = closer_by > self.OFF_THRESHOLD
            else:
                self._in_front[side] = closer_by > self.ON_THRESHOLD
            if not self._in_front[side]:
                continue

            elbow = pose.points[elbow_i]
            wrist = pose.points[wrist_i]
            forearm_len = float(np.linalg.norm(wrist - elbow))
            thickness = int(np.clip(0.38 * forearm_len, 12, 64))

            cv2.line(
                occ,
                tuple(elbow.astype(int)),
                tuple(wrist.astype(int)),
                1.0,
                thickness,
            )
            cv2.circle(
                occ, tuple(wrist.astype(int)), int(thickness * 0.75), 1.0, -1
            )
            if index_i in pose.points:  # extend over the hand
                cv2.line(
                    occ,
                    tuple(wrist.astype(int)),
                    tuple(pose.points[index_i].astype(int)),
                    1.0,
                    int(thickness * 0.9),
                )
            drew = True

        if not drew:
            return rendered

        person = np.asarray(mask, dtype=np.float32)
        if person.shape[:2] != (h, w):
            person = cv2.resize(person, (w, h))
        occ = np.minimum(occ, (person > 0.5).astype(np.float32))
        occ = cv2.GaussianBlur(occ, (9, 9), 0)[:, :, np.newaxis]

        out = (
            rendered.astype(np.float32) * (1.0 - occ)
            + original.astype(np.float32) * occ
        )
        return out.astype(np.uint8)
