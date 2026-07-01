# ======================================================
# ui.py — Premium dark-theme HUD for AI Draw Pro
# ======================================================

import cv2
import numpy as np
import time
from typing import Tuple, Dict, Optional
from config import PALETTE, TOOLS


# ── Layout constants ──
TOOLBAR_H = 65            # toolbar height in px
TOOL_BTN_W = 80           # width of each tool button
TOOL_BTN_H = 45           # height of each tool button
TOOL_BTN_Y = 10           # top margin
TOOL_BTN_GAP = 6          # gap between buttons
COLOR_SWATCH_R = 16       # radius of colour circles
STATUS_BAR_H = 32         # bottom status bar height

# Precompute tool button positions: {tool_name: (x1, y1, x2, y2)}
TOOL_BUTTONS: Dict[str, Tuple[int, int, int, int]] = {}
_x = 12
for t in TOOLS:
    TOOL_BUTTONS[t] = (_x, TOOL_BTN_Y, _x + TOOL_BTN_W, TOOL_BTN_Y + TOOL_BTN_H)
    _x += TOOL_BTN_W + TOOL_BTN_GAP

# Precompute colour swatch positions
COLOR_BUTTONS: Dict[int, Tuple[int, int]] = {}   # idx -> (cx, cy)
_cx_start = _x + 30
for i in range(len(PALETTE)):
    COLOR_BUTTONS[i] = (_cx_start + i * (COLOR_SWATCH_R * 2 + 10),
                        TOOLBAR_H // 2)


def get_all_button_coords() -> Dict[str, Tuple[int, int, int, int]]:
    """Return a unified dict of all clickable regions (tools + colours)."""
    coords = dict(TOOL_BUTTONS)
    for i, (cx, cy) in COLOR_BUTTONS.items():
        r = COLOR_SWATCH_R + 2
        coords[f"color_{i}"] = (cx - r, cy - r, cx + r, cy + r)
    return coords


# ── Tool icons (drawn with OpenCV primitives) ──

def _draw_brush_icon(img, cx, cy, color):
    cv2.circle(img, (cx, cy), 6, color, -1)
    cv2.line(img, (cx + 4, cy + 4), (cx + 12, cy + 12), color, 2)


def _draw_eraser_icon(img, cx, cy, color):
    cv2.rectangle(img, (cx - 7, cy - 5), (cx + 7, cy + 5), color, 2)
    cv2.line(img, (cx - 4, cy - 3), (cx + 4, cy + 3), color, 2)
    cv2.line(img, (cx + 4, cy - 3), (cx - 4, cy + 3), color, 2)


def _draw_line_icon(img, cx, cy, color):
    cv2.line(img, (cx - 10, cy + 6), (cx + 10, cy - 6), color, 2)


def _draw_rect_icon(img, cx, cy, color):
    cv2.rectangle(img, (cx - 9, cy - 6), (cx + 9, cy + 6), color, 2)


def _draw_circle_icon(img, cx, cy, color):
    cv2.circle(img, (cx, cy), 8, color, 2)


def _draw_star_icon(img, cx, cy, color):
    import math
    pts = []
    for i in range(10):
        angle = (i * 36 - 90) * math.pi / 180
        r = 9 if i % 2 == 0 else 4
        pts.append((int(cx + r * math.cos(angle)),
                     int(cy + r * math.sin(angle))))
    cv2.polylines(img, [np.array(pts, np.int32)], True, color, 1)


def _draw_arrow_icon(img, cx, cy, color):
    cv2.arrowedLine(img, (cx - 10, cy + 4), (cx + 10, cy - 4),
                    color, 2, tipLength=0.35)


def _draw_puzzle_icon(img, cx, cy, color):
    # Draw a 3x3 grid outline
    size = 18
    half = size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    # Grid lines
    third = size // 3
    cv2.line(img, (cx - half + third, y1), (cx - half + third, y2), color, 1)
    cv2.line(img, (cx - half + 2 * third, y1), (cx - half + 2 * third, y2), color, 1)
    cv2.line(img, (x1, cy - half + third), (x2, cy - half + third), color, 1)
    cv2.line(img, (x1, cy - half + 2 * third), (x2, cy - half + 2 * third), color, 1)


ICON_DRAWERS = {
    "brush": _draw_brush_icon,
    "eraser": _draw_eraser_icon,
    "line": _draw_line_icon,
    "rect": _draw_rect_icon,
    "circle": _draw_circle_icon,
    "star": _draw_star_icon,
    "arrow": _draw_arrow_icon,
    "puzzle": _draw_puzzle_icon,
}


# ── Gesture feedback overlay ──

class GestureFeedback:
    """Shows a brief animated text overlay when a gesture triggers."""

    def __init__(self, duration: float = 1.2):
        self.duration = duration
        self._text: str = ""
        self._start: float = 0

    def show(self, text: str):
        self._text = text
        self._start = time.time()

    def draw(self, frame: np.ndarray):
        if not self._text:
            return
        elapsed = time.time() - self._start
        if elapsed > self.duration:
            self._text = ""
            return

        # Fade-out alpha
        alpha = max(0.0, 1.0 - elapsed / self.duration)
        h, w = frame.shape[:2]

        # Draw text with fading opacity via overlay blend
        overlay = frame.copy()
        text_size = cv2.getTextSize(self._text, cv2.FONT_HERSHEY_SIMPLEX,
                                     1.2, 3)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 2

        # Background pill
        pad = 20
        cv2.rectangle(overlay,
                      (tx - pad, ty - text_size[1] - pad),
                      (tx + text_size[0] + pad, ty + pad),
                      (20, 20, 20), -1)
        cv2.putText(overlay, self._text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 3)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Main UI renderer ──

class UIRenderer:
    """
    Renders the full HUD overlay:
      • Semi-transparent toolbar with tool icons
      • Circular colour swatches
      • Active tool / colour highlighting
      • Bottom status bar (tool, size, layer, FPS)
      • Custom cursor
    """

    def __init__(self):
        self.feedback = GestureFeedback()
        self._cached_toolbar: Optional[np.ndarray] = None
        self._cached_state: tuple = ()

    def draw(
        self,
        frame: np.ndarray,
        active_tool: str,
        active_color: Tuple[int, int, int],
        brush_size: int,
        current_layer: int,
        fps: float,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ):
        """Draw the full UI onto *frame* (mutates in-place)."""
        h, w = frame.shape[:2]

        # ── Toolbar background (glass effect) ──
        state_key = (active_tool, active_color, w)
        if self._cached_toolbar is None or self._cached_state != state_key:
            self._cached_toolbar = self._render_toolbar(w, active_tool,
                                                         active_color)
            self._cached_state = state_key

        # Blend toolbar with semi-transparency
        roi = frame[0:TOOLBAR_H, 0:w]
        cv2.addWeighted(self._cached_toolbar[:, :w], 0.85,
                        roi, 0.15, 0, roi)

        # ── Status bar ──
        self._draw_status_bar(frame, active_tool, brush_size,
                               current_layer, fps)

        # ── Cursor ──
        if cursor_pos:
            self._draw_cursor(frame, cursor_pos, active_tool,
                               active_color, brush_size)

        # ── Gesture feedback ──
        self.feedback.draw(frame)

    def _render_toolbar(self, width: int, active_tool: str,
                        active_color: Tuple) -> np.ndarray:
        """Render the toolbar to a reusable buffer."""
        bar = np.zeros((TOOLBAR_H, width, 3), dtype=np.uint8)
        bar[:] = (25, 25, 30)  # dark charcoal

        # Bottom edge glow line
        cv2.line(bar, (0, TOOLBAR_H - 1), (width, TOOLBAR_H - 1),
                 (60, 60, 70), 1)

        # ── Tool buttons ──
        for tool, (x1, y1, x2, y2) in TOOL_BUTTONS.items():
            is_active = (tool == active_tool)

            # Button background
            bg = (50, 50, 60) if not is_active else (40, 70, 50)
            cv2.rectangle(bar, (x1, y1), (x2, y2), bg, -1)

            # Border
            border_color = (0, 220, 180) if is_active else (70, 70, 80)
            cv2.rectangle(bar, (x1, y1), (x2, y2), border_color, 2)

            # Glow effect for active button
            if is_active:
                cv2.rectangle(bar, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                              (0, 180, 140), 1)

            # Icon
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            icon_color = (0, 255, 200) if is_active else (160, 160, 170)
            if tool in ICON_DRAWERS:
                ICON_DRAWERS[tool](bar, cx, cy, icon_color)

            # Label below icon — small
            label = tool.upper()
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.28, 1)[0]
            tx = cx - text_size[0] // 2
            ty = y2 - 3
            cv2.putText(bar, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, icon_color, 1,
                        cv2.LINE_AA)

        # ── Colour swatches ──
        for i, (cx, cy) in COLOR_BUTTONS.items():
            if i < len(PALETTE):
                col = PALETTE[i]
                cv2.circle(bar, (cx, cy), COLOR_SWATCH_R, col, -1,
                           cv2.LINE_AA)

                # Highlight selected colour with a white ring
                if col == active_color:
                    cv2.circle(bar, (cx, cy), COLOR_SWATCH_R + 3,
                               (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(bar, (cx, cy), COLOR_SWATCH_R + 5,
                               (0, 220, 180), 1, cv2.LINE_AA)
                else:
                    cv2.circle(bar, (cx, cy), COLOR_SWATCH_R + 1,
                               (80, 80, 90), 1, cv2.LINE_AA)

        return bar

    def _draw_status_bar(self, frame: np.ndarray, tool: str,
                          brush_size: int, layer: int, fps: float):
        """Draw bottom status bar."""
        h, w = frame.shape[:2]
        y1 = h - STATUS_BAR_H

        # Semi-transparent background
        overlay = frame[y1:h, 0:w].copy()
        cv2.rectangle(overlay, (0, 0), (w, STATUS_BAR_H), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.8, frame[y1:h, 0:w], 0.2, 0,
                        frame[y1:h, 0:w])

        # Top edge line
        cv2.line(frame, (0, y1), (w, y1), (60, 60, 70), 1)

        ty = y1 + 22

        # Tool name
        cv2.putText(frame, f"Tool: {tool.upper()}", (15, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 180), 1,
                    cv2.LINE_AA)

        # Brush size
        cv2.putText(frame, f"Size: {brush_size}", (180, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1,
                    cv2.LINE_AA)

        # Layer
        cv2.putText(frame, f"Layer: {layer + 1}", (310, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 190), 1,
                    cv2.LINE_AA)

        # FPS
        fps_color = (0, 200, 0) if fps > 20 else (0, 200, 255)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1,
                    cv2.LINE_AA)

        # Gesture hints
        cv2.putText(frame, "SHAKA=Clear  3_FNGR=Save  PALM=Erase",
                    (430, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (100, 100, 110), 1, cv2.LINE_AA)

    def _draw_cursor(self, frame: np.ndarray, pos: Tuple[int, int],
                      tool: str, color: Tuple, brush_size: int):
        """Draw a context-aware cursor at *pos*."""
        x, y = pos

        if tool == "brush":
            # Circle showing brush size + crosshair
            cv2.circle(frame, (x, y), brush_size, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
            # Mini crosshair
            gap = brush_size + 4
            cv2.line(frame, (x - gap, y), (x - gap + 5, y),
                     (200, 200, 200), 1)
            cv2.line(frame, (x + gap, y), (x + gap - 5, y),
                     (200, 200, 200), 1)
            cv2.line(frame, (x, y - gap), (x, y - gap + 5),
                     (200, 200, 200), 1)
            cv2.line(frame, (x, y + gap), (x, y + gap - 5),
                     (200, 200, 200), 1)

        elif tool == "puzzle":
            # Selection cross/circle cursor for puzzle mode
            cv2.circle(frame, (x, y), 8, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        else:
            # Shape tools — dotted crosshair
            cv2.drawMarker(frame, (x, y), (0, 255, 200),
                           cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_puzzle(self, frame: np.ndarray, pieces: list,
                    grabbed_idx: Optional[int], cursor_pos: Optional[Tuple[int, int]],
                    solved: bool = False, hover_progress: float = 0.0):
        """
        Draw the 3x3 puzzle board centered on frame.
        pieces: list of dicts {"orig_idx": int, "current_idx": int, "img": np.ndarray}
        """
        BOARD_X = 317
        BOARD_Y = 65
        CELL_SZ = 215

        # Draw slot borders
        for k in range(9):
            row = k // 3
            col = k % 3
            x1 = BOARD_X + col * CELL_SZ
            y1 = BOARD_Y + row * CELL_SZ
            cv2.rectangle(frame, (x1, y1), (x1 + CELL_SZ, y1 + CELL_SZ), (60, 60, 70), 1)

        # Draw target highlight border
        if grabbed_idx is not None and cursor_pos is not None:
            cx, cy = cursor_pos
            target_col = (cx - BOARD_X) // CELL_SZ
            target_row = (cy - BOARD_Y) // CELL_SZ
            if 0 <= target_col < 3 and 0 <= target_row < 3:
                tx1 = BOARD_X + target_col * CELL_SZ
                ty1 = BOARD_Y + target_row * CELL_SZ
                cv2.rectangle(frame, (tx1, ty1), (tx1 + CELL_SZ, ty1 + CELL_SZ), (0, 255, 200), 2)

        # Draw all non-grabbed pieces
        grabbed_piece = None
        for piece in pieces:
            curr_idx = piece["current_idx"]
            orig_idx = piece["orig_idx"]
            img_slice = piece["img"]

            if orig_idx == grabbed_idx:
                grabbed_piece = piece
                continue

            row = curr_idx // 3
            col = curr_idx % 3
            x1 = BOARD_X + col * CELL_SZ
            y1 = BOARD_Y + row * CELL_SZ
            x2 = x1 + CELL_SZ
            y2 = y1 + CELL_SZ
            frame[y1:y2, x1:x2] = img_slice

        # Draw grabbed piece floating centered at the cursor
        if grabbed_piece is not None and cursor_pos is not None:
            cx, cy = cursor_pos
            gx1 = max(0, cx - CELL_SZ // 2)
            gy1 = max(0, cy - CELL_SZ // 2)
            gx2 = min(frame.shape[1], gx1 + CELL_SZ)
            gy2 = min(frame.shape[0], gy1 + CELL_SZ)

            pw = gx2 - gx1
            ph = gy2 - gy1
            if pw > 0 and ph > 0:
                # Crop if exceeds screen bounds
                slice_img = grabbed_piece["img"][:ph, :pw]
                frame[gy1:gy2, gx1:gx2] = slice_img
                # Highlight border for floating grabbed piece
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 200), 2)

        # ── Draw Control Panel on the Left Sidebar ──
        btn_x1, btn_y1, btn_x2, btn_y2 = 40, 330, 278, 390
        
        # Panel header text
        cv2.putText(frame, "PUZZLE CONTROLS", (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 110), 1, cv2.LINE_AA)
        
        # Semi-transparent button background (glass effect)
        sub = frame[btn_y1:btn_y2, btn_x1:btn_x2]
        rect_overlay = np.zeros_like(sub)
        rect_overlay[:] = (25, 25, 30)  # dark charcoal background
        cv2.addWeighted(rect_overlay, 0.7, sub, 0.3, 0, sub)
        
        # Hover progress bar at the bottom of the button
        if hover_progress > 0.0:
            bar_h = 4
            prog_w = int((btn_x2 - btn_x1) * hover_progress)
            cv2.rectangle(frame, (btn_x1, btn_y2 - bar_h), (btn_x1 + prog_w, btn_y2), (0, 255, 200), -1)
            # Glowing border
            cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 200), 2)
        else:
            # Default border
            cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (70, 70, 80), 2)
            
        # Draw "RETAKE & SHUFFLE" text centered in the button
        btn_text = "RETAKE & SHUFFLE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        text_size = cv2.getTextSize(btn_text, font, font_scale, thickness)[0]
        text_x = btn_x1 + (btn_x2 - btn_x1 - text_size[0]) // 2
        text_y = btn_y1 + (btn_y2 - btn_y1 + text_size[1]) // 2
        
        text_color = (0, 255, 200) if hover_progress > 0.0 else (200, 200, 210)
        cv2.putText(frame, btn_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        if solved:
            h, w = frame.shape[:2]
            
            # Semi-transparent overlay banner across the middle
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h // 2 - 80), (w, h // 2 + 80), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Glowing borders on banner
            cv2.line(frame, (0, h // 2 - 80), (w, h // 2 - 80), (0, 255, 200), 2)
            cv2.line(frame, (0, h // 2 + 80), (w, h // 2 + 80), (0, 255, 200), 2)

            import time
            import math
            # Pulsing color/scale
            pulse = math.sin(time.time() * 5)
            color1 = (0, 255, 200)  # glowing teal
            color2 = (255, 255, 255) # white
            
            t1 = "CONGRATULATIONS!"
            t2 = "PUZZLE SOLVED!"
            
            # Calculate text sizes and coordinates
            size1 = cv2.getTextSize(t1, cv2.FONT_HERSHEY_SIMPLEX, 1.1 + 0.05 * pulse, 2)[0]
            size2 = cv2.getTextSize(t2, cv2.FONT_HERSHEY_SIMPLEX, 0.8 + 0.04 * pulse, 2)[0]
            
            tx1 = w // 2 - size1[0] // 2
            ty1 = h // 2 - 10
            
            tx2 = w // 2 - size2[0] // 2
            ty2 = h // 2 + 40
            
            cv2.putText(frame, t1, (tx1, ty1), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.1 + 0.05 * pulse, color1, 2, cv2.LINE_AA)
            cv2.putText(frame, t2, (tx2, ty2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8 + 0.04 * pulse, color2, 2, cv2.LINE_AA)

    def draw_countdown(self, frame: np.ndarray, remaining: int, elapsed: float):
        """Draw a pulsing, glowing circular countdown timer in the center of the screen."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Determine pulse scale (fractional part of the current second)
        sub_sec = elapsed - int(elapsed)
        pulse = 1.0 - sub_sec  # goes from 1.0 down to 0.0 for a clean pulse animation
        
        # Draw dark dimmed circle background
        r = int(55 + 20 * pulse)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, (20, 20, 25), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        # Draw glowing outer ring
        cv2.circle(frame, (cx, cy), r, (0, 255, 200), 2, cv2.LINE_AA)
        
        # Draw countdown text number
        text = str(remaining)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8 * pulse + 0.5, 3)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.8 * pulse + 0.5, (255, 255, 255), 3, cv2.LINE_AA)