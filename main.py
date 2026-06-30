# ======================================================
# main.py — Entry point for AI Draw Pro
# ======================================================

import sys
from app import DrawingApp


def main():
    """Launch the gesture-controlled virtual drawing application."""
    camera_src = 0
    if len(sys.argv) > 1:
        try:
            camera_src = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python main.py [camera_index]")
            print(f"  camera_index: integer (default 0)")
            sys.exit(1)

    app = DrawingApp(camera_src=camera_src)
    app.run()


if __name__ == "__main__":
    main()