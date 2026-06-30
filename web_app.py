import os
import time
import json
import threading
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

from config import Config, PALETTE, TOOLS
from camera import Camera
from hand_tracking import HandTracker
from gestures import GestureDetector
from shapes import draw_shape, draw_shape_preview
from ui import UIRenderer

# Initialize configuration
cfg = Config()

# Global variables for web stream
global_frame = None
frame_lock = threading.Lock()

class WebDrawingApp:
    def __init__(self):
        self.cfg = cfg
        self.cam = Camera(0)
        self.tracker = HandTracker()
        self.gestures = GestureDetector()
        self.ui = UIRenderer()
        
        # Dimensions are initialized lazily
        self._canvas_ready = False
        self.layers = []
        self.layer_vis = []
        
        self.undo_stack = []
        self._max_undo = 10
        self._draw_points = []
        self._shape_start = None
        self._was_pinching = False
        self._pinch_release_counter = 0
        self._cursor = (0, 0)
        self.dark_mode = True  # Neon mode by default

    def _init_canvas(self, h, w):
        self.layers = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(self.cfg.num_layers)]
        self.layer_vis = [True] * self.cfg.num_layers
        self._canvas_ready = True

    def _merge_layers(self):
        merged = np.zeros_like(self.layers[0])
        for i, layer in enumerate(self.layers):
            if self.layer_vis[i]:
                cv2.add(merged, layer, merged)
        return merged

    def _save_undo(self):
        layer = self.layers[self.cfg.current_layer]
        if len(self.undo_stack) >= self._max_undo:
            self.undo_stack.pop(0)
        self.undo_stack.append(layer.copy())

    def _draw_brush_stroke(self, lm):
        ix, iy = self._cursor
        self._draw_points.append((ix, iy))
        if len(self._draw_points) > 12:
            self._draw_points.pop(0)
        
        layer = self.layers[self.cfg.current_layer]
        if len(self._draw_points) >= 2:
            for i in range(1, len(self._draw_points)):
                cv2.line(layer, self._draw_points[i-1], self._draw_points[i],
                         self.cfg.color, self.cfg.brush_size)

    def _erase_at_cursor(self, lm):
        ix, iy = self._cursor
        cv2.circle(self.layers[self.cfg.current_layer], (ix, iy),
                   self.cfg.brush_size * 2, (0, 0, 0), -1)

    def _handle_shape(self, lm, display):
        ix, iy = self._cursor
        pinching = self.gestures.is_pinching

        if pinching:
            self._pinch_release_counter = 0
            if self._shape_start is None:
                self._shape_start = (ix, iy)
            else:
                draw_shape_preview(display, self.cfg.tool,
                                   self._shape_start, (ix, iy),
                                   self.cfg.color, self.cfg.brush_size)
        else:
            if self._shape_start is not None:
                self._pinch_release_counter += 1
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
                    draw_shape_preview(display, self.cfg.tool,
                                       self._shape_start, (ix, iy),
                                       self.cfg.color, self.cfg.brush_size)

        self._was_pinching = pinching

    def run_loop(self):
        global global_frame
        self.cam.warmup()
        
        while True:
            ret, frame = self.cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if not self._canvas_ready:
                self._init_canvas(h, w)

            self.tracker.process(frame)
            lm = self.tracker.get_landmarks(w, h)
            self.gestures.update(lm, self.tracker.handedness)

            display = frame.copy()

            if lm is not None:
                if self.gestures.is_pinching:
                    ix = int((lm[8][0] + lm[4][0]) / 2)
                    iy = int((lm[8][1] + lm[4][1]) / 2)
                else:
                    ix, iy = int(lm[8][0]), int(lm[8][1])
                self._cursor = (ix, iy)

                if self.gestures.is_erasing:
                    self._erase_at_cursor(lm)
                elif self.cfg.tool == "brush":
                    self._draw_brush_stroke(lm)
                elif self.cfg.tool == "eraser":
                    self._erase_at_cursor(lm)
                elif self.cfg.tool in ("line", "rect", "circle", "star", "arrow", "triangle"):
                    self._handle_shape(lm, display)
                else:
                    self._draw_points.clear()

                if self.gestures.trigger_clear:
                    self._save_undo()
                    self.layers[self.cfg.current_layer][:] = 0
            else:
                self._draw_points.clear()
                self._shape_start = None
                self._was_pinching = False

            # Compositing
            merged = self._merge_layers()
            if self.dark_mode:
                display = merged
            else:
                display = cv2.addWeighted(display, 0.65, merged, 1.0, 0)

            # Draw cursor ring
            if lm is not None:
                cv2.circle(display, self._cursor, self.cfg.brush_size, self.cfg.color, 1, cv2.LINE_AA)

            ret_jpg, jpeg = cv2.imencode('.jpg', display)
            if ret_jpg:
                with frame_lock:
                    global_frame = jpeg.tobytes()

            time.sleep(0.01)

# Start background thread
drawing_engine = WebDrawingApp()
t = threading.Thread(target=drawing_engine.run_loop, daemon=True)
t.start()

# Flask setup
# Serve templates from the misspelled "templetes" folder
app = Flask(__name__, template_folder='templetes')

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n')
        time.sleep(0.04)  # ~25 fps

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    data = request.get_json()
    if not data:
        return jsonify(ok=False)

    if 'tool' in data:
        cfg.tool = data['tool']
    if 'brush_size' in data:
        cfg.brush_size = data['brush_size']
    if 'dark_mode' in data:
        drawing_engine.dark_mode = data['dark_mode']
    if 'clear' in data and data['clear']:
        drawing_engine._save_undo()
        for layer in drawing_engine.layers:
            layer[:] = 0
    if 'undo' in data and data['undo']:
        if drawing_engine.undo_stack:
            drawing_engine.layers[drawing_engine.cfg.current_layer] = drawing_engine.undo_stack.pop()

    return jsonify(ok=True)

@app.route('/save', methods=['POST'])
def save():
    merged = drawing_engine._merge_layers()
    fname = f"web_drawing_{int(time.time())}.png"
    cv2.imwrite(fname, merged)
    return jsonify(ok=True, filename=fname)

if __name__ == '__main__':
    # Listen on all interfaces on port 5001
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
