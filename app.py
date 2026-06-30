# ======================================================
# app.py — Main application controller for AI Draw Pro
# ======================================================

import cv2
import numpy as np
import time
from typing import Optional, Tuple

from config import Config, PALETTE, TOOLS
from camera import Camera
from hand_tracking import HandTracker
from gestures import GestureDetector, is_index_only
from shapes import draw_shape, draw_shape_preview
from ui import UIRenderer, get_all_button_coords, TOOLBAR_H


class DrawingApp:
    """
    Encapsulates all drawing state and the main event loop.

    Responsibilities:
      • Camera + hand tracker lifecycle
      • Canvas layers & undo/redo stacks
      • Brush / shape drawing
      • UI click handling
      • Gesture → action mapping
    """

    def __init__(self, camera_src: int = 0):
        self.cfg = Config()

        # ── Camera ──
        self.cam = Camera(camera_src,
                          width=self.cfg.frame_width,
                          height=self.cfg.frame_height)
        self._check_camera()

        # ── Hand tracker ──
        self.tracker = HandTracker(
            detection_confidence=self.cfg.detection_confidence,
            tracking_confidence=self.cfg.tracking_confidence,
            smooth_alpha=self.cfg.landmark_alpha,
        )

        # ── Gesture detector ──
        self.gestures = GestureDetector(
            shaka_hold=self.cfg.shaka_hold_frames,
            save_hold=self.cfg.save_hold_frames,
            cooldown=self.cfg.gesture_cooldown_frames,
            pinch_threshold=self.cfg.pinch_threshold,
            ok_threshold=self.cfg.ok_threshold,
        )

        # ── UI ──
        self.ui = UIRenderer()
        self._button_coords = get_all_button_coords()

        # ── Canvas & layers ──
        self.layers = []          # populated on first frame
        self.layer_vis = []
        self._canvas_ready = False

        # ── Undo / redo ──
        self.undo_stack = []
        self.redo_stack = []
        self._max_undo = 20

        # ── Drawing state ──
        self._draw_points = []
        self._max_draw_points = 20

        # ── Shape placement ──
        self._shape_start: Optional[Tuple[int, int]] = None
        self._was_pinching = False
        self._pinch_release_counter = 0

        # ── Puzzle states ──
        self.puzzle_pieces = []
        self.grabbed_piece_idx = None
        self.puzzle_active = False
        self.puzzle_solved = False
        self.confetti_particles = []
        self.countdown_active = False
        self.countdown_start_time = 0.0
        self._trigger_puzzle_capture = False

        # ── Cursor (smoothed separately from landmarks) ──
        self._cursor = (0, 0)

        # ── Button debounce ──
        self._btn_pressed = False

    # ── Lifecycle ──────────────────────────────────────

    def _check_camera(self):
        if not self.cam.cap.isOpened():
            print("=" * 60)
            print("ERROR: Camera failed to initialise.")
            print("macOS: System Settings → Privacy & Security → Camera")
            print("=" * 60)
            raise SystemExit(1)
        print("Camera opened. Warming up...")
        self.cam.warmup()
        print("Camera ready.")

    def _init_canvas(self, h: int, w: int):
        """Lazily create layer buffers matching the camera resolution."""
        self.layers = [np.zeros((h, w, 3), dtype=np.uint8)
                       for _ in range(self.cfg.num_layers)]
        self.layer_vis = [True] * self.cfg.num_layers
        self._canvas_ready = True

    # ── Undo / redo ──

    def _save_undo(self):
        layer = self.layers[self.cfg.current_layer]
        if len(self.undo_stack) >= self._max_undo:
            self.undo_stack.pop(0)
        self.undo_stack.append(layer.copy())
        self.redo_stack.clear()

    def _do_undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(
            self.layers[self.cfg.current_layer].copy())
        self.layers[self.cfg.current_layer] = self.undo_stack.pop()

    def _do_redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(
            self.layers[self.cfg.current_layer].copy())
        self.layers[self.cfg.current_layer] = self.redo_stack.pop()

    # ── Layer merging ──

    def _merge_layers(self) -> np.ndarray:
        merged = self.layers[0].copy()
        for i in range(1, len(self.layers)):
            if self.layer_vis[i]:
                cv2.add(merged, self.layers[i], merged)
        return merged

    # ── Drawing helpers ──

    def _draw_brush_stroke(self, lm: np.ndarray):
        """Accumulate points and draw continuous brush strokes."""
        ix, iy = int(lm[8][0]), int(lm[8][1])
        self._cursor = (ix, iy)
        self._draw_points.append((ix, iy))
        if len(self._draw_points) > self._max_draw_points:
            self._draw_points.pop(0)

        layer = self.layers[self.cfg.current_layer]
        pts = self._draw_points
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                if self.cfg.current_pattern == "solid":
                    cv2.line(layer, pts[i - 1], pts[i],
                             self.cfg.color, self.cfg.brush_size,
                             cv2.LINE_AA)
                elif self.cfg.current_pattern == "dotted":
                    if i % 4 == 0:
                        cv2.circle(layer, pts[i],
                                   self.cfg.brush_size // 2,
                                   self.cfg.color, -1, cv2.LINE_AA)
                elif self.cfg.current_pattern == "dashed":
                    if i % 6 < 3:
                        cv2.line(layer, pts[i - 1], pts[i],
                                 self.cfg.color, self.cfg.brush_size,
                                 cv2.LINE_AA)

    def _erase_at_cursor(self, lm: np.ndarray):
        ix, iy = int(lm[8][0]), int(lm[8][1])
        self._cursor = (ix, iy)
        cv2.circle(self.layers[self.cfg.current_layer],
                   (ix, iy), self.cfg.eraser_size, (0, 0, 0), -1)

    # ── Shape placement via pinch ──

    def _handle_shape(self, lm: np.ndarray, display: np.ndarray):
        """Pinch to set start → drag → release pinch with hysteresis to commit shape."""
        ix, iy = int(lm[8][0]), int(lm[8][1])
        self._cursor = (ix, iy)

        pinching = self.gestures.is_pinching

        if pinching != self._was_pinching:
            from gestures import _dist, _palm_size, THUMB_TIP, INDEX_TIP
            curr_dist = _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / _palm_size(lm)
            print(f"[Shape Mode] Pinch state: {'Pinch Active' if pinching else 'Released'} | Dist: {curr_dist:.3f} | Thresh: {self.cfg.pinch_threshold}")

        if pinching:
            self._pinch_release_counter = 0
            if self._shape_start is None:
                # Pinch just started → set anchor
                self._shape_start = (ix, iy)
            else:
                # Still pinching → show preview
                draw_shape_preview(display, self.cfg.tool,
                                   self._shape_start, (ix, iy),
                                   self.cfg.color, self.cfg.brush_size)
        else:
            if self._shape_start is not None:
                self._pinch_release_counter += 1
                # Hysteresis: wait for 5 frames of release before committing
                if self._pinch_release_counter >= 5:
                    from shapes import constrain_shape_point
                    snapped_p2 = constrain_shape_point(self._shape_start, (ix, iy), self.cfg.tool)
                    
                    self._save_undo()
                    draw_shape(self.layers[self.cfg.current_layer],
                               self.cfg.tool, self._shape_start, snapped_p2,
                               self.cfg.color, self.cfg.brush_size)
                    self._shape_start = None
                    self._pinch_release_counter = 0
                else:
                    # Still show preview during the release debouncing window
                    draw_shape_preview(display, self.cfg.tool,
                                       self._shape_start, (ix, iy),
                                       self.cfg.color, self.cfg.brush_size)

        self._was_pinching = pinching

    # ── Puzzle Mode Logic ──

    def _start_puzzle(self, composite_img: np.ndarray):
        """Slice the composite screen capture into a 3x3 grid of pieces and shuffle them."""
        import random
        BOARD_X = 317
        BOARD_Y = 65
        CELL_SZ = 215

        center_img = composite_img[BOARD_Y:BOARD_Y+645, BOARD_X:BOARD_X+645]
        if center_img.shape[:2] != (645, 645):
            center_img = cv2.resize(center_img, (645, 645))

        self.puzzle_pieces = []
        for row in range(3):
            for col in range(3):
                orig_idx = row * 3 + col
                y1 = row * CELL_SZ
                x1 = col * CELL_SZ
                img_slice = center_img[y1:y1+CELL_SZ, x1:x1+CELL_SZ].copy()
                self.puzzle_pieces.append({
                    "orig_idx": orig_idx,
                    "current_idx": orig_idx,
                    "img": img_slice
                })

        # Shuffle indices
        indices = list(range(9))
        while True:
            random.shuffle(indices)
            if any(indices[i] != i for i in range(9)):
                break

        for i, piece in enumerate(self.puzzle_pieces):
            piece["current_idx"] = indices[i]

        self.puzzle_active = True
        self.puzzle_solved = False
        self.confetti_particles = []
        self.grabbed_piece_idx = None

    def _handle_puzzle(self, lm: np.ndarray):
        """Pinch to grab a piece → drag to grid slot → release with hysteresis to swap (with edge-snapping)."""
        ix, iy = int(lm[8][0]), int(lm[8][1])
        self._cursor = (ix, iy)

        BOARD_X = 317
        BOARD_Y = 65
        CELL_SZ = 215

        pinching = self.gestures.is_pinching

        if pinching:
            self._pinch_release_counter = 0
            if self.grabbed_piece_idx is None:
                # Find if cursor is over a grid slot
                col = (ix - BOARD_X) // CELL_SZ
                row = (iy - BOARD_Y) // CELL_SZ
                if 0 <= col < 3 and 0 <= row < 3:
                    slot_idx = row * 3 + col
                    # Find which piece occupies this slot
                    for piece in self.puzzle_pieces:
                        if piece["current_idx"] == slot_idx:
                            self.grabbed_piece_idx = piece["orig_idx"]
                            break
        else:
            if self.grabbed_piece_idx is not None:
                self._pinch_release_counter += 1
                # Hysteresis: wait for 5 frames of release to prevent drop due to tracking noise
                if self._pinch_release_counter >= 5:
                    # Released → determine target slot (clamped to board slots for edge accuracy)
                    col = int(np.clip((ix - BOARD_X) // CELL_SZ, 0, 2))
                    row = int(np.clip((iy - BOARD_Y) // CELL_SZ, 0, 2))
                    target_idx = row * 3 + col
                    
                    # Find the grabbed piece and target piece
                    grabbed_piece = None
                    target_piece = None
                    for piece in self.puzzle_pieces:
                        if piece["orig_idx"] == self.grabbed_piece_idx:
                            grabbed_piece = piece
                        if piece["current_idx"] == target_idx:
                            target_piece = piece

                    if grabbed_piece is not None and target_piece is not None:
                        # Swap slots
                        temp = grabbed_piece["current_idx"]
                        grabbed_piece["current_idx"] = target_piece["current_idx"]
                        target_piece["current_idx"] = temp

                        # Check if solved
                        solved = all(p["current_idx"] == p["orig_idx"] for p in self.puzzle_pieces)
                        if solved:
                            self.puzzle_solved = True
                            self._init_confetti()
                    
                    self.grabbed_piece_idx = None
                    self._pinch_release_counter = 0

    def _init_confetti(self):
        """Initialise colorful floating confetti particles."""
        import random
        self.confetti_particles = []
        h, w = 720, 1280
        for _ in range(120):
            self.confetti_particles.append({
                "x": random.randint(0, w),
                "y": random.randint(-400, 0),
                "vx": random.uniform(-3, 3),
                "vy": random.uniform(2, 6),
                "color": (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)),
                "size": random.randint(3, 8)
            })

    def _update_confetti(self, frame: np.ndarray):
        """Update physics and draw particles falling down the screen."""
        import random
        h, w = frame.shape[:2]
        for p in self.confetti_particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.18  # gravity

            cv2.circle(frame, (int(p["x"]), int(p["y"])), p["size"], p["color"], -1, cv2.LINE_AA)

            if p["y"] > h or p["x"] < 0 or p["x"] > w:
                p["x"] = random.randint(0, w)
                p["y"] = random.randint(-50, -10)
                p["vx"] = random.uniform(-3, 3)
                p["vy"] = random.uniform(1, 4)

    # ── UI click handling ──

    def _handle_ui_click(self, ix: int, iy: int) -> bool:
        """Check if cursor is over a button. Returns True if a button hit."""
        for name, (x1, y1, x2, y2) in self._button_coords.items():
            if x1 < ix < x2 and y1 < iy < y2:
                if name in TOOLS:
                    self.cfg.tool = name
                    if name == "puzzle":
                        import time
                        self.countdown_active = True
                        self.countdown_start_time = time.time()
                        self._trigger_puzzle_capture = False
                        self.puzzle_active = False
                        self.puzzle_pieces = []
                        self.grabbed_piece_idx = None
                    else:
                        self.puzzle_active = False
                        self.puzzle_pieces = []
                        self.grabbed_piece_idx = None
                        self.countdown_active = False
                elif name.startswith("color_"):
                    idx = int(name.split("_")[1])
                    if idx < len(PALETTE):
                        self.cfg.color = PALETTE[idx]
                        self.cfg.tool = "brush"
                        self.puzzle_active = False
                        self.puzzle_pieces = []
                        self.grabbed_piece_idx = None
                        self.countdown_active = False
                return True
        return False

    # ── Main loop ──────────────────────────────────────

    def run(self):
        """Main application loop. Press ESC to exit."""
        print("AI Draw Pro running. Press ESC to quit.")

        while True:
            ret, frame = self.cam.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if not self._canvas_ready:
                self._init_canvas(h, w)

            # ── Hand tracking (every frame for accuracy) ──
            self.tracker.process(frame)
            lm = self.tracker.get_landmarks(w, h)

            # ── Gesture detection ──
            self.gestures.update(lm, self.tracker.handedness)

            display = frame.copy()

            if lm is not None:
                if self.gestures.is_pinching:
                    # Average of index tip and thumb tip for stable dragging
                    ix = int((lm[8][0] + lm[4][0]) / 2)
                    iy = int((lm[8][1] + lm[4][1]) / 2)
                else:
                    ix, iy = int(lm[8][0]), int(lm[8][1])
                self._cursor = (ix, iy)

                # ── UI click (index only in toolbar zone) ──
                if self.gestures.is_drawing and iy < TOOLBAR_H:
                    hit = self._handle_ui_click(ix, iy)
                    if hit and not self._btn_pressed:
                        self._draw_points.clear()
                        self._btn_pressed = True
                    elif not hit:
                        self._btn_pressed = False
                else:
                    self._btn_pressed = False

                # ── Drawing & Puzzle Logic ──
                if self.countdown_active:
                    pass
                elif self.puzzle_active:
                    self._handle_puzzle(lm)
                else:
                    if self.gestures.is_erasing:
                        if iy > TOOLBAR_H:
                            self._erase_at_cursor(lm)
                    elif self.cfg.tool == "brush" and self.gestures.is_drawing:
                        if iy > TOOLBAR_H:
                            self._draw_brush_stroke(lm)
                    elif self.cfg.tool == "eraser" and self.gestures.is_drawing:
                        if iy > TOOLBAR_H:
                            self._erase_at_cursor(lm)
                    elif self.cfg.tool in ("line", "rect", "circle",
                                            "star", "arrow", "triangle"):
                        self._handle_shape(lm, display)
                    else:
                        self._draw_points.clear()

                # ── Gesture actions ──
                if self.gestures.trigger_clear:
                    self._save_undo()
                    self.layers[self.cfg.current_layer][:] = 0
                    self.ui.feedback.show("Canvas Cleared")

                if self.gestures.trigger_save:
                    merged = self._merge_layers()
                    fname = f"drawing_{int(time.time())}.png"
                    cv2.imwrite(fname, merged)
                    self.ui.feedback.show(f"Saved: {fname}")
                    print(f"Saved {fname}")
            else:
                self._draw_points.clear()
                self._shape_start = None
                self._was_pinching = False
                self.grabbed_piece_idx = None
                self._pinch_release_counter = 0

            # ── Merge layers + composite ──
            if self.countdown_active:
                import time
                elapsed = time.time() - self.countdown_start_time
                remaining = 3 - int(elapsed)
                if elapsed >= 3.0:
                    self.countdown_active = False
                    self._trigger_puzzle_capture = True
                else:
                    self.ui.draw_countdown(display, remaining, elapsed)

            if self.puzzle_active:
                display = (display * 0.25).astype(np.uint8)
                cursor = self._cursor if lm is not None else None
                self.ui.draw_puzzle(display, self.puzzle_pieces, self.grabbed_piece_idx, cursor, solved=self.puzzle_solved)
                if self.puzzle_solved:
                    self._update_confetti(display)
            elif not self.countdown_active:
                merged = self._merge_layers()
                display = cv2.addWeighted(display, 0.65, merged, 1.0, 0)

                if self._trigger_puzzle_capture:
                    self._trigger_puzzle_capture = False
                    self._start_puzzle(display)
                    display = (display * 0.25).astype(np.uint8)
                    cursor = self._cursor if lm is not None else None
                    self.ui.draw_puzzle(display, self.puzzle_pieces, self.grabbed_piece_idx, cursor, solved=self.puzzle_solved)

            # ── Draw UI ──
            cursor = self._cursor if lm is not None else None
            active_tool_to_draw = "eraser" if (lm is not None and self.gestures.is_erasing) else self.cfg.tool
            self.ui.draw(
                display,
                active_tool=active_tool_to_draw,
                active_color=self.cfg.color,
                brush_size=self.cfg.brush_size,
                current_layer=self.cfg.current_layer,
                fps=self.cam.fps,
                cursor_pos=cursor,
            )

            cv2.imshow("AI Draw Pro", display)

            # ── Keyboard ──
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("p"):
                # Cycle brush pattern
                patterns = self.cfg.brush_patterns
                idx = patterns.index(self.cfg.current_pattern)
                self.cfg.current_pattern = patterns[(idx + 1) % len(patterns)]
                self.ui.feedback.show(
                    f"Pattern: {self.cfg.current_pattern}")
            elif key == ord("+") or key == ord("="):
                self.cfg.brush_size = min(self.cfg.max_brush_size,
                                          self.cfg.brush_size + 2)
            elif key == ord("-"):
                self.cfg.brush_size = max(self.cfg.min_brush_size,
                                          self.cfg.brush_size - 2)

        self._cleanup()

    def _cleanup(self):
        print("Shutting down...")
        self.cam.release()
        self.tracker.close()
        cv2.destroyAllWindows()
