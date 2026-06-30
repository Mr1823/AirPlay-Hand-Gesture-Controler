# ======================================================
# hand_tracking.py — MediaPipe hand tracker with EMA
# ======================================================

import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class HandTracker:
    """
    High-accuracy hand tracker with:
      • Configurable detection / tracking confidence
      • Exponential Moving Average (EMA) landmark smoothing
      • Pixel-coordinate helper that eliminates repeated conversions
      • Handedness detection (left / right)
    """

    def __init__(
        self,
        detection_confidence: float = 0.8,
        tracking_confidence: float = 0.75,
        smooth_alpha: float = 0.6,
    ):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.results = None
        self.smooth_alpha = smooth_alpha

        # Previous smoothed landmarks (21 × 3 array, normalised)
        self._prev_landmarks: np.ndarray | None = None

        # Detected handedness string ("Left" / "Right")
        self.handedness: str = "Right"

    # ── public API ──────────────────────────────────────

    def process(self, frame: np.ndarray):
        """Run MediaPipe on a BGR frame. Returns raw results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        # Update handedness
        if self.results and self.results.multi_handedness:
            label = self.results.multi_handedness[0].classification[0].label
            # MediaPipe reports the label from the camera's perspective
            # (mirrored), so "Left" from MP means the user's right hand
            # when the frame is already flipped.  We store as-is since
            # the frame is flipped in our pipeline.
            self.handedness = label

        return self.results

    def get_landmarks(self, w: int, h: int) -> np.ndarray | None:
        """
        Return smoothed landmark pixel coords as (21, 2) int array,
        or None if no hand is detected.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            self._prev_landmarks = None
            return None

        hand = self.results.multi_hand_landmarks[0]
        raw = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float64
        )

        # EMA smoothing in normalised space
        if self._prev_landmarks is not None:
            a = self.smooth_alpha
            smoothed = a * self._prev_landmarks + (1 - a) * raw
        else:
            smoothed = raw
        self._prev_landmarks = smoothed

        # Convert to pixel coords (only x, y)
        px = smoothed[:, :2].copy()
        px[:, 0] *= w
        px[:, 1] *= h
        return px.astype(np.int32)

    def get_raw_landmarks(self):
        """Return the raw MediaPipe landmark list or None."""
        if self.results and self.results.multi_hand_landmarks:
            return self.results.multi_hand_landmarks[0].landmark
        return None

    def get_confidence(self) -> float:
        """Return detection confidence score (0–1), or 0 if nothing."""
        if self.results and self.results.multi_handedness:
            return self.results.multi_handedness[0].classification[0].score
        return 0.0

    def draw_landmarks(self, frame: np.ndarray):
        """Draw MediaPipe skeleton overlay (for debug)."""
        if self.results and self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

    def close(self):
        self.hands.close()