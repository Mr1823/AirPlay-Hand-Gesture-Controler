# ✋ AirPlay — Hand Gesture Controller & Virtual Drawing System

A high-performance, real-time, gesture-controlled virtual drawing and canvas system. Using a standard computer webcam, **AirPlay** tracks hand landmarks in 3D space, applying advanced noise filtering and gesture detection to allow painting, geometric shape creation, and canvas management entirely through air gestures — no mouse, keyboard, or touch screen needed.

---

## ✨ Advanced Features

*   **⚡ Zero-Lag Threaded Capture**: Employs a dedicated background capture thread to ensure camera frame grabbing is independent of rendering, maximizing frame rate and minimizing lag.
*   **🎯 Jitter-Free Hand Tracking**: Applies an **Exponential Moving Average (EMA) filter** to MediaPipe hand landmarks in normalized space before pixel projection, yielding silky-smooth brush movement.
*   **📐 CAD-Style Shape Placement**:
    *   **Pinch-to-Place**: Pinch (thumb + index) to set shape anchor, drag to resize, and release to commit. Supports *Lines, Rectangles, Circles, Triangles, Stars, and Arrows*.
    *   **Auto-Snapping (Alignment Lock)**: Snaps lines/arrows to the nearest 45° angle if within a 7° range. Snaps rectangles/triangles to perfect squares/equilateral boxes when close to a 1:1 aspect ratio.
    *   **Dashed Guidelines**: Automatically projects horizontal and vertical dashed guidelines from the starting anchor across the screen to help align shapes.
    *   **Measurements HUD Tooltip**: Displays real-time length, angles, and dimensions in pixels next to the cursor.
*   **🎨 Premium Dark HUD Toolbar**:
    *   Semi-transparent toolbar with rounded tool buttons.
    *   OpenCV vector-drawn tool icons (brush, eraser, shape templates).
    *   8-color circular palette with highlight ring on selected color.
    *   Context-aware cursors matching the active tool.
*   **🧹 Smart Gesture Actions**:
    *   **Open Palm**: Instantly triggers the eraser brush (no menu navigation needed).
    *   **Shaka Gesture**: Hold to clear the entire canvas.
    *   **Three Fingers**: Hold to save the drawing as a PNG image.
    *   **Hysteresis & Latching**: Gestures require specific frame hold times to trigger and latch once, avoiding duplicate clear/save triggers.

---

## 🖐️ Gesture Control Reference

| Gesture | Action | Description |
|:---|:---|:---|
| **Index finger up** | **Navigate / Draw** | Move cursor; hover over top toolbar buttons to switch colors/tools. |
| **Open Palm** (4-5 fingers up) | **Instant Erase** | Erases drawings under the cursor with a larger eraser radius. |
| **Pinch** (Thumb + Index touching) | **Place Shapes** | Pinch to set anchor, drag to size, release to commit the active shape. |
| **Shaka Sign** (Thumb + Pinky up) | **Clear Canvas** | Hold for 12 frames (~0.4s) to clear the current drawing layer. |
| **Three Fingers up** (Index + Middle + Ring) | **Save Image** | Hold for 12 frames (~0.4s) to save the drawing as a PNG file. |

### 🎹 Keyboard Shortcuts

*   `P` : Cycle brush pattern (Solid ──, Dotted •••, Dashed - -)
*   `+` / `-` : Increase or decrease brush size
*   `ESC` : Close application and shut down cleanly

---

## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| **Python** | Core programming language |
| **OpenCV** | Video capture, frame manipulation, custom HUD, and drawing |
| **MediaPipe** | Hand landmark tracking and handedness detection |
| **NumPy** | Multidimensional array operations for layers and drawing buffers |

---

## 📦 Installation & Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Mr1823/AirPlay-Hand-Gesture-Controler.git
    cd AirPlay-Hand-Gesture-Controler
    ```

2.  **Initialize Virtual Environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install opencv-python "mediapipe<=0.10.14" numpy
    ```

4.  **Run the Application**
    ```bash
    python main.py
    ```

---

## 👨‍💻 Author

**Mr1823**
*   GitHub: [@Mr1823](https://github.com/Mr1823)

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
