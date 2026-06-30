# ======================================================
# shapes.py — Shape drawing and live preview
# ======================================================

import cv2
import numpy as np
import math
from typing import Tuple

Point = Tuple[int, int]


def draw_star(img: np.ndarray, center: Point, size: int,
              color: Tuple, thickness: int = 2):
    """Draw a 5-pointed star centred at *center* with given *size*."""
    cx, cy = center
    pts = []
    for i in range(10):
        angle = (i * 36 - 90) * math.pi / 180   # start from top
        r = size if i % 2 == 0 else size // 2
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        pts.append((x, y))
    pts_arr = np.array(pts, np.int32)
    cv2.polylines(img, [pts_arr], True, color, thickness)


def draw_triangle(img: np.ndarray, p1: Point, p2: Point,
                  color: Tuple, thickness: int = 2):
    """Equilateral-style triangle from bounding box defined by p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    mid_x = (x1 + x2) // 2
    pts = np.array([
        [mid_x, min(y1, y2)],
        [min(x1, x2), max(y1, y2)],
        [max(x1, x2), max(y1, y2)],
    ], np.int32)
    cv2.polylines(img, [pts], True, color, thickness)


def draw_shape(img: np.ndarray, tool: str, p1: Point, p2: Point,
               color: Tuple, thickness: int = 2):
    """Draw a shape on *img* between points *p1* and *p2*."""
    if tool == "line":
        cv2.line(img, p1, p2, color, thickness)

    elif tool == "rect":
        cv2.rectangle(img, p1, p2, color, thickness)

    elif tool == "circle":
        r = int(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        cv2.circle(img, p1, r, color, thickness)

    elif tool == "star":
        size = int(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        draw_star(img, p1, size, color, thickness)

    elif tool == "arrow":
        cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=0.25)

    elif tool == "triangle":
        draw_triangle(img, p1, p2, color, thickness)


def constrain_shape_point(p1: Point, p2: Point, tool: str) -> Point:
    """
    Constrain the second point (p2) to standard angles or aspect ratios:
      • Line/Arrow: snap to nearest 45° angle if within 7°
      • Rect/Triangle: snap to perfect square/equilateral if aspect ratio is close to 1:1
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if tool in ("line", "arrow"):
        dist = math.hypot(dx, dy)
        if dist < 5:
            return p2
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        # Snap to nearest 45 degrees
        snap_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        for snap in snap_angles:
            if abs(angle - snap) < 7:
                rad = math.radians(snap)
                new_x = int(x1 + dist * math.cos(rad))
                new_y = int(y1 + dist * math.sin(rad))
                return (new_x, new_y)

    elif tool in ("rect", "triangle"):
        w = abs(dx)
        h = abs(dy)
        if w > 0 and h > 0:
            ratio = w / h
            if 0.82 < ratio < 1.18:
                side = max(w, h)
                new_x = x1 + (side if dx >= 0 else -side)
                new_y = y1 + (side if dy >= 0 else -side)
                return (new_x, new_y)

    return p2


def draw_guidelines(img: np.ndarray, p1: Point, width: int, height: int):
    """Draw horizontal and vertical dashed guidelines passing through p1."""
    x1, y1 = p1
    color = (70, 70, 80)  # subtle dark grey/blue

    # Horizontal dashed line
    for x in range(0, width, 16):
        cv2.line(img, (x, y1), (min(x + 8, width), y1), color, 1)

    # Vertical dashed line
    for y in range(0, height, 16):
        cv2.line(img, (x1, y), (x1, min(y + 8, height)), color, 1)


def draw_measurements_hud(img: np.ndarray, tool: str, p1: Point, p2: Point):
    """Draw real-time size and shape information next to the cursor (p2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    text = ""
    if tool in ("line", "arrow"):
        dist = int(math.hypot(dx, dy))
        angle = int(math.degrees(math.atan2(-dy, dx)))  # standard math coords (y-up)
        if angle < 0:
            angle += 360
        text = f"L: {dist}px | A: {angle}deg"
    elif tool in ("rect", "triangle"):
        w = abs(dx)
        h = abs(dy)
        text = f"W: {w}px | H: {h}px"
        if w == h:
            text += " (Square)"
    elif tool in ("circle", "star"):
        r = int(math.hypot(dx, dy))
        text = f"R: {r}px"

    if text:
        tx, ty = p2[0] + 15, p2[1] + 15
        h, w = img.shape[:2]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        
        # Keep HUD tooltip inside window boundaries
        if tx + text_size[0] > w - 10:
            tx = p2[0] - text_size[0] - 15
        if ty + text_size[1] > h - 10:
            ty = p2[1] - 15
        if ty - text_size[1] < 10:
            ty = p2[1] + 25

        pad = 4
        # Draw background pill
        cv2.rectangle(img,
                      (tx - pad, ty - text_size[1] - pad),
                      (tx + text_size[0] + pad, ty + pad),
                      (20, 20, 25), -1)
        # Draw teal text
        cv2.putText(img, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1, cv2.LINE_AA)


def draw_shape_preview(display: np.ndarray, tool: str,
                       p1: Point, p2: Point, color: Tuple,
                       thickness: int = 2):
    """
    Draw horizontal/vertical guidelines, the snapped shape preview,
    and the measurements HUD tooltip.
    """
    h, w = display.shape[:2]
    
    # 1. Guidelines
    draw_guidelines(display, p1, w, h)
    
    # 2. Constrain/snap point
    snapped_p2 = constrain_shape_point(p1, p2, tool)
    
    # 3. Shape preview
    preview_color = tuple(min(255, int(c * 0.6 + 80)) for c in color)
    draw_shape(display, tool, p1, snapped_p2, preview_color, max(1, thickness - 1))
    
    # 4. Dimensions HUD
    draw_measurements_hud(display, tool, p1, snapped_p2)