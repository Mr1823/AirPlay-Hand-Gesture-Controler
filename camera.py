# ======================================================
# camera.py — Threaded camera capture with FPS tracking
# ======================================================

import cv2
import threading
import time


class Camera:
    """Threaded video capture with FPS measurement and warm-up."""

    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # FPS tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start = time.time()

        # Read first frame
        self.ret, self.frame = self.cap.read()

        # Start background capture thread
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    @property
    def fps(self) -> float:
        """Current measured FPS of the camera capture."""
        return self._fps

    def _update(self):
        """Background loop — continuously grabs frames."""
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

            # Update FPS counter every second
            self._frame_count += 1
            elapsed = time.time() - self._fps_start
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start = time.time()

    def warmup(self, n_frames: int = 10, timeout: float = 3.0):
        """Block until *n_frames* valid frames have been captured or timeout."""
        start = time.time()
        good = 0
        while good < n_frames and (time.time() - start) < timeout:
            with self.lock:
                if self.ret and self.frame is not None:
                    good += 1
            time.sleep(0.03)
        return good >= n_frames

    def read(self):
        """Return the latest frame (thread-safe copy)."""
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        """Stop capture thread and release hardware."""
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()