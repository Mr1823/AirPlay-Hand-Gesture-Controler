# ======================================================
# gestures.py — Robust gesture detection with debouncing
# ======================================================

import math
import numpy as np
from typing import List, Optional


# ── MediaPipe landmark indices ──
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]


def _dist(a, b) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _palm_size(lm: np.ndarray) -> float:
    """Approximate palm size as distance from wrist to middle-finger MCP."""
    return max(_dist(lm[WRIST], lm[MIDDLE_MCP]), 1.0)


# ── Stateless finger helpers ──────────────────────────

def fingers_up(lm: np.ndarray, handedness: str = "Right") -> List[bool]:
    """
    Detect which fingers are extended.

    Args:
        lm: (21, 2) pixel landmark array.
        handedness: "Left" or "Right" (MediaPipe label, camera-perspective).

    Returns:
        List of 5 bools: [thumb, index, middle, ring, pinky].
    """
    fingers = []

    # ── Thumb: angle-based detection ──
    # Vector from thumb MCP → thumb tip vs wrist → thumb MCP
    v1 = lm[THUMB_TIP] - lm[THUMB_IP]
    v2 = lm[THUMB_MCP] - lm[WRIST]

    # Also check if thumb tip is sufficiently away from palm centre
    palm_cx = int(np.mean([lm[WRIST][0], lm[INDEX_MCP][0],
                            lm[PINKY_MCP][0], lm[MIDDLE_MCP][0]]))
    palm_cy = int(np.mean([lm[WRIST][1], lm[INDEX_MCP][1],
                            lm[PINKY_MCP][1], lm[MIDDLE_MCP][1]]))
    palm_center = np.array([palm_cx, palm_cy])

    thumb_dist = _dist(lm[THUMB_TIP], palm_center)
    palm_sz = _palm_size(lm)

    # Thumb is "up" if its tip is far enough from the palm
    # AND it extends outward in the correct direction for the hand
    if handedness == "Right":
        thumb_extended = lm[THUMB_TIP][0] < lm[THUMB_IP][0]
    else:
        thumb_extended = lm[THUMB_TIP][0] > lm[THUMB_IP][0]

    fingers.append(thumb_extended and (thumb_dist / palm_sz) > 0.8)

    # ── Other fingers: tip above PIP ──
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(lm[tip][1] < lm[pip][1])

    return fingers


def is_shaka(fingers: List[bool]) -> bool:
    """Thumb and pinky extended, others folded."""
    return fingers[0] and fingers[4] and not fingers[1] and not fingers[2] and not fingers[3]


def is_index_only(fingers: List[bool]) -> bool:
    """Only the index finger is up."""
    return fingers[1] and not any(fingers[2:]) and not fingers[0]


def is_three_fingers(fingers: List[bool]) -> bool:
    """Index, middle, and ring up; thumb and pinky down."""
    return fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]


def is_open_palm(fingers: List[bool]) -> bool:
    """Index, middle, ring, pinky fingers are all extended (regardless of thumb)."""
    return all(fingers[1:])


def is_pinch(lm: np.ndarray, threshold: float = 0.06) -> bool:
    """Thumb tip touching index tip (normalised to palm size)."""
    dist = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
    return (dist / _palm_size(lm)) < threshold


# ── Stateful gesture detector ────────────────────────

class GestureDetector:
    """
    Wraps raw gesture functions with:
      • Frame-counter debouncing for hold-gestures (shaka, save)
      • Clean state reset when hand disappears
    """

    def __init__(
        self,
        shaka_hold: int = 12,
        save_hold: int = 12,
        cooldown: int = 15,
        pinch_threshold: float = 0.08,
        pinch_release_threshold: float = 0.12,
        ok_threshold: float = 0.055,
    ):
        self.shaka_hold = shaka_hold
        self.save_hold = save_hold
        self.cooldown = cooldown
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = pinch_release_threshold

        # Internal counters
        self._shaka_count = 0
        self._save_count = 0

        # Latch states (prevent repeated execution while holding)
        self._shaka_latched = False
        self._save_latched = False

        # Public gesture results (set each frame)
        self.is_drawing = False       # index-only → draw mode
        self.trigger_clear = False    # shaka held → clear canvas
        self.trigger_save = False     # two-finger held → save
        self.is_erasing = False       # open palm → erase mode
        self.is_pinching = False      # pinch active → shape placement
        self.fingers: List[bool] = [False] * 5

    def update(self, lm: Optional[np.ndarray], handedness: str = "Right"):
        """
        Call once per frame with the current pixel landmarks (or None).
        After calling, read the public gesture flags.
        """
        # Reset one-shot triggers
        self.trigger_clear = False
        self.trigger_save = False

        if lm is None:
            self._reset_counters()
            self.is_drawing = False
            self.is_erasing = False
            self.is_pinching = False
            self.fingers = [False] * 5
            return

        self.fingers = fingers_up(lm, handedness)
        self.is_drawing = is_index_only(self.fingers)
        self.is_erasing = is_open_palm(self.fingers)

        # Stateful pinch detection with hysteresis (prevents quick flickering)
        dist_val = _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / _palm_size(lm)
        if self.is_pinching:
            self.is_pinching = (dist_val < self.pinch_release_threshold)
        else:
            self.is_pinching = (dist_val < self.pinch_threshold)

        # ── Shaka → clear (hold gesture with latching) ──
        if is_shaka(self.fingers):
            if not self._shaka_latched:
                self._shaka_count += 1
                if self._shaka_count >= self.shaka_hold:
                    self.trigger_clear = True
                    self._shaka_latched = True
                    self._shaka_count = 0
        else:
            self._shaka_count = 0
            self._shaka_latched = False

        # ── Three fingers → save (hold gesture with latching) ──
        if is_three_fingers(self.fingers):
            if not self._save_latched:
                self._save_count += 1
                if self._save_count >= self.save_hold:
                    self.trigger_save = True
                    self._save_latched = True
                    self._save_count = 0
        else:
            self._save_count = 0
            self._save_latched = False

    def _reset_counters(self):
        self._shaka_count = 0
        self._save_count = 0
        self._shaka_latched = False
        self._save_latched = False