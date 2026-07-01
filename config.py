# ======================================================
# config.py — Central configuration for AI Draw Pro
# ======================================================

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class Config:
    """All tuneable application settings in one place."""

    # ── Resolution ──
    frame_width: int = 1280
    frame_height: int = 720

    # ── Drawing defaults ──
    tool: str = "brush"
    color: Tuple[int, int, int] = (0, 200, 255)   # warm orange (BGR)
    brush_size: int = 6
    eraser_size: int = 30
    max_brush_size: int = 40
    min_brush_size: int = 2

    # ── Smoothing ──
    cursor_alpha: float = 0.55          # EMA blend for cursor position
    landmark_alpha: float = 0.6         # EMA blend for all landmarks

    # ── Hand tracking ──
    detection_confidence: float = 0.8
    tracking_confidence: float = 0.75

    # ── Gesture thresholds ──
    shaka_hold_frames: int = 12          # frames shaka must be held to clear
    save_hold_frames: int = 12          # frames two-finger must be held to save
    gesture_cooldown_frames: int = 15   # cooldown after triggering a one-shot gesture
    pinch_threshold: float = 0.15       # normalised distance for pinch detection
    ok_threshold: float = 0.12          # normalised distance for OK gesture

    # ── Layers ──
    num_layers: int = 3
    current_layer: int = 0

    # ── Brush patterns ──
    brush_patterns: List[str] = field(
        default_factory=lambda: ["solid", "dotted", "dashed"]
    )
    current_pattern: str = "solid"


# ── Colour palette (BGR) ──
PALETTE = [
    (71, 99, 255),     # Coral Red
    (80, 200, 120),    # Emerald Green
    (255, 144, 30),    # Dodger Blue
    (0, 215, 255),     # Gold / Yellow
    (255, 0, 128),     # Magenta / Purple
    (208, 224, 64),    # Turquoise
    (255, 255, 255),   # White
    (180, 180, 180),   # Light Grey
]

# ── Tool list (order matches toolbar layout) ──
TOOLS = ["brush", "eraser", "line", "rect", "circle", "star", "arrow", "scan", "puzzle"]