"""
Adaptive Threshold live-tuning tool.
Drag sliders to preview in real time.
Press S to save parameters, Q/ESC to quit.
"""

import cv2
import numpy as np
import sys

DEFAULT_BLOCK_SIZE = 11   # must be odd
DEFAULT_C          = 2
DEFAULT_METHOD     = 0    # 0=MEAN, 1=GAUSSIAN
DEFAULT_TYPE       = 0    # 0=BINARY, 1=BINARY_INV

params = {
    "block_size": DEFAULT_BLOCK_SIZE,
    "C":          DEFAULT_C,
    "method":     DEFAULT_METHOD,
    "thresh_type": DEFAULT_TYPE,
    "blur_ksize": 0,   # 0=none, 1=3x3, 2=5x5, 3=7x7
}

WIN_RESULT = "AdaptiveThreshold Result"

src_gray = None


def apply_threshold(gray, p):
    blur_map = {0: None, 1: (3, 3), 2: (5, 5), 3: (7, 7)}
    ksize = blur_map[p["blur_ksize"]]
    if ksize:
        gray = cv2.GaussianBlur(gray, ksize, 0)

    method = cv2.ADAPTIVE_THRESH_MEAN_C if p["method"] == 0 \
             else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    ttype  = cv2.THRESH_BINARY if p["thresh_type"] == 0 \
             else cv2.THRESH_BINARY_INV

    bs = max(3, p["block_size"])
    if bs % 2 == 0:
        bs += 1

    return cv2.adaptiveThreshold(gray, 255, method, ttype, bs, p["C"])


def render():
    if src_gray is None:
        return

    result = apply_threshold(src_gray, params)

    bs = max(3, params["block_size"])
    if bs % 2 == 0:
        bs += 1

    info = [
        f"Method    : {'MEAN' if params['method'] == 0 else 'GAUSSIAN'}",
        f"BlockSize : {bs}",
        f"C         : {params['C']}",
        f"Type      : {'BINARY' if params['thresh_type'] == 0 else 'BINARY_INV'}",
        f"PreBlur   : {['None', '3x3', '5x5', '7x7'][params['blur_ksize']]}",
    ]
    display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    for i, line in enumerate(info):
        cv2.putText(display, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow(WIN_RESULT, display)


def on_block_size(val):
    params["block_size"] = val * 2 + 3   # slider 1-100 → odd 3-203
    render()

def on_c(val):
    params["C"] = val - 50   # slider 0-100 → C: -50 to 50
    render()

def on_method(val):
    params["method"] = val
    render()

def on_type(val):
    params["thresh_type"] = val
    render()

def on_blur(val):
    params["blur_ksize"] = val
    render()


def main():
    global src_gray

    image_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images\Shore 00-10\img_0070.jpg"
    use_camera = image_path is None

    cap = None
    if use_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if ret:
            src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read image: {image_path}")
            sys.exit(1)
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

    # BlockSize slider: value v → blockSize = v*2+3 (odd, range 3-203)
    cv2.createTrackbar("BlockSize(x2+3)", WIN_RESULT,
                       (DEFAULT_BLOCK_SIZE - 3) // 2, 100, on_block_size)
    # C slider: 0-100, actual C = val-50, range -50 to 50
    cv2.createTrackbar("C (+50offset)",   WIN_RESULT,
                       DEFAULT_C + 50, 100, on_c)
    cv2.createTrackbar("Method(0M 1G)",   WIN_RESULT, DEFAULT_METHOD, 1, on_method)
    cv2.createTrackbar("Type(0B 1Inv)",   WIN_RESULT, DEFAULT_TYPE,   1, on_type)
    cv2.createTrackbar("PreBlur(0-3)",    WIN_RESULT, 0, 3, on_blur)

    print("Controls:")
    print("  Drag sliders  → live preview")
    print("  S             → save params to best_params.py")
    print("  Q / ESC       → quit")

    render()

    while True:
        if use_camera:
            ret, frame = cap.read()
            if ret:
                src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                render()

        key = cv2.waitKey(16) & 0xFF   # ~60 Hz
        if key in (ord('q'), 27):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
