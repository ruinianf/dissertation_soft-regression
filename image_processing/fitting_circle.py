import cv2
import numpy as np

PATH = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images\Shore 00-10\img_0000.jpg"

points = []
WINDOW_NAME = "6 Points Circle Fitting"
SCALE = 1.0
img_orig = None
img_draw = None


def mouse_callback(event, x, y, flags, param):
    global points, img_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 6:
            points.append((x, y))
            print(f"Selected point {len(points)}: ({x}, {y})")

            cv2.circle(img_draw, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(img_draw, str(len(points)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(WINDOW_NAME, img_draw)

            if len(points) == 6:
                print(">>>")
                fit_and_draw_circle()


def fit_and_draw_circle():
    global points, img_draw
    pts_arr = np.array(points, dtype=np.float32)

    try:
        x = pts_arr[:, 0]
        y = pts_arr[:, 1]

        M_arr = np.vstack([x, y, np.ones(len(x))]).T
        Y_arr = -(x**2 + y**2)

        result = np.linalg.lstsq(M_arr, Y_arr, rcond=None)
        A, B, C = result[0]

        cx = -A / 2
        cy = -B / 2
        radius = np.sqrt((A**2 + B**2) / 4 - C)

        cv2.circle(img_draw, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
        cv2.circle(img_draw, (int(cx), int(cy)), 2, (0, 255, 0), -1)

        cv2.putText(img_draw, f"Fit Radius: {int(radius)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"Fit Centre:({int(cx)}, {int(cy)}), Fit Radius:{int(radius)}")

    except Exception as e:
        print(f"Fit failed, error: {e}")
        cv2.putText(img_draw, "Fit Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, img_draw)
    print("Press 'r' to reset, press 'q' to quit.")


def get_display_size(img, max_w=1400, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return int(w * scale), int(h * scale), scale


def mouse_callback(event, x, y, flags, param):
    global points, img_draw, SCALE

    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x / SCALE)
        real_y = int(y / SCALE)

        if len(points) < 6:
            points.append((real_x, real_y))
            print(f"Selected point {len(points)}: ({real_x}, {real_y})")

            cv2.circle(img_draw, (real_x, real_y), 3, (0, 0, 255), -1)
            cv2.putText(img_draw, str(len(points)), (real_x + 10, real_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            show_scaled()

            if len(points) == 6:
                print(">>>")
                fit_and_draw_circle()


def show_scaled():
    disp_w, disp_h, _ = get_display_size(img_draw)
    img_resized = cv2.resize(img_draw, (disp_w, disp_h))
    cv2.imshow(WINDOW_NAME, img_resized)


if __name__ == "__main__":
    img_orig = cv2.imread(PATH)

    if img_orig is None:
        print("Failed to load image!")
        exit()

    img_draw = img_orig.copy()

    disp_w, disp_h, SCALE = get_display_size(img_orig)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, disp_w, disp_h)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        show_scaled()
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            points = []
            img_draw = img_orig.copy()
            print(">>> Reset. Please select 6 points again.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
