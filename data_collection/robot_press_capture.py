import cri.controller
import cv2
import os
import time
import threading
import random
from datetime import datetime
from cri.robot import SyncRobot
from cri.dobot.mg400_controller import MG400Controller

capturing = False


def capture_thread(camera_id, save_dir):
    global capturing
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)
    if not cap.isOpened():
        print("Warning! Camera not be initialized.")
        return

    frame_count = 0
    print(f"Camera starts to capture frame {frame_count}.")

    while capturing:
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(save_dir, f"img_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            frame_count += 1
            time.sleep(0.02)

    cap.release()
    print(f"Finished! Save {frame_count} images.")


def main():
    global capturing

    # config
    CAMERA_ID = 1
    ROBOT_IP = "192.168.1.6"
    NUM_TRIALS = 20 ##

    # Offset range, Yaw ±5.0°
    OFFSET_R = 5.0

    base_save_path = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images\Shore 00-90" ##

    ## (X, Y, Z, R, j1, j2)
    base_initial = (320.0, 10.0, -30.0, 0.0, 0.0, 0.0)
    base_press_start = (320.0, 10.0, -56.80, 0.0, 0.0, 0.0) ##
    base_press_end = (base_press_start[0], base_press_start[1], base_press_start[2] - 5.0, base_press_start[3],
                      base_press_start[4], base_press_start[5])  ##

    print(f"Connecting to Dobot at {ROBOT_IP}...")
    robot = SyncRobot(MG400Controller(ROBOT_IP))
    robot.speed = 2

    print(f" {NUM_TRIALS} loops in total.")

    for trial in range(1, NUM_TRIALS + 1):  ##
        print(f"\n---Loop {trial}/{NUM_TRIALS}---")

        trial_dir = os.path.join(base_save_path, f"Trial_{trial:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        dr = random.uniform(-OFFSET_R, OFFSET_R)
        print(f"Random offset R:{dr:+.2f}°")

        curr_initial = list(base_initial)
        curr_initial[3] += dr

        curr_start = list(base_press_start)
        curr_start[3] += dr

        curr_end = list(base_press_end)
        curr_end[3] += dr

        print("Move to initial position...")
        robot.speed = 20 ##
        robot.move_linear(tuple(curr_initial))
        time.sleep(1)

        print("Move to press start position...")
        robot.move_linear(tuple(curr_start))
        time.sleep(1)

        print("Start to capture...")
        capturing = True
        t = threading.Thread(target=capture_thread, args=(CAMERA_ID, trial_dir))
        t.start()
        time.sleep(1)

        print("Move to press end position...")
        robot.speed = 2 ##
        robot.move_linear(tuple(curr_end))

        capturing = False
        t.join()
        print("Wait for 2 seconds...")
        time.sleep(2)

        capturing = False
        t.join()

        print("Move back to initial position...")
        robot.speed = 20  ##
        robot.move_linear(tuple(curr_initial))
        time.sleep(1)

    print("\nOVER!")
    print("Disconnecting Dobot...")


if __name__ == "__main__":
    main()