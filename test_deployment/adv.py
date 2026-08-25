import os
import cv2
import time
import threading
import glob
import csv
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from cri.robot import SyncRobot
from cri.dobot.mg400_controller import MG400Controller

# 1. config
CAMERA_ID = 1
ROBOT_IP = "192.168.1.6"

BASE_INITIAL = (320.0, 10.0, 0.0, 0.0, 0.0, 0.0)
SEARCH_START_Z = -45.0 ##
PRESS_DEPTH = 5.0

MODEL_DIR = r"C:\Users\Nick\Desktop\UK\learning\dissertation\code\train_model"
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib") ##
MODEL_PATH = os.path.join(MODEL_DIR, "rf_regressor_final.joblib") ##

BLACKBOX_SAVE_DIR = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\blackbox_tests"

capturing = False

# 2. core function

def get_fast_centroids(img):
    BLOCK_SIZE = 73
    C_VALUE = 7
    CIRCLE_CX, CIRCLE_CY, CIRCLE_RADIUS = 312, 233, 169

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE, C_VALUE)

    mask = np.zeros_like(gray)
    cv2.circle(mask, (CIRCLE_CX, CIRCLE_CY), CIRCLE_RADIUS, 255, -1)
    final_binary = cv2.bitwise_and(binary, mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_binary, connectivity=8,
                                                                            ltype=cv2.CV_32S)

    valid_centroids = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] > 5:
            valid_centroids.append(centroids[label])

    return np.array(valid_centroids)


def auto_detect_contact(robot, camera_id):
    print("\nContact Detection")

    search_start_pose = (BASE_INITIAL[0], BASE_INITIAL[1], SEARCH_START_Z, BASE_INITIAL[3], BASE_INITIAL[4],
                         BASE_INITIAL[5])
    robot.speed = 20
    robot.move_linear(search_start_pose)
    time.sleep(1)

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)

    if not cap.isOpened():
        print("Warning: fail to open camera!")
        return SEARCH_START_Z

    # Base Frame
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    ret, base_img = cap.read()
    base_centroids = get_fast_centroids(base_img)

    if len(base_centroids) == 0:
        print("Error: No markers detected in base frame!")
        cap.release()
        return SEARCH_START_Z

    contact_z = SEARCH_START_Z
    step_size = 0.1
    max_search_depth = -75.0

    print("Gently descending to detect the surface...")
    robot.speed = 20

    current_z = SEARCH_START_Z
    while current_z > max_search_depth:
        current_z -= step_size
        robot.move_linear(
            (BASE_INITIAL[0], BASE_INITIAL[1], current_z, BASE_INITIAL[3], BASE_INITIAL[4], BASE_INITIAL[5]))
        time.sleep(0.5)

        for _ in range(3):
            cap.read()

        ret, curr_img = cap.read()
        curr_centroids = get_fast_centroids(curr_img)

        if len(curr_centroids) > 0:
            dist_matrix = cdist(base_centroids, curr_centroids)
            min_dists = np.min(dist_matrix, axis=1)
            mean_disp = np.mean(min_dists)
        else:
            mean_disp = 0.0

        print(f"      [Debug] Z: {current_z:.2f} | mean displacement: {mean_disp:.3f} pixels")

        if mean_disp > 0.125:  ###*
            print(f"Contact confirmed! Current Z coordinate: {current_z:.2f} (Mean displacement: {mean_disp:.3f})")
            contact_z = current_z
            break

    cap.release()

    print("Return to initial position (releasing stress)...")
    robot.move_linear(
        (BASE_INITIAL[0], BASE_INITIAL[1], contact_z + 2.0, BASE_INITIAL[3], BASE_INITIAL[4], BASE_INITIAL[5]))
    time.sleep(3)

    return contact_z


def capture_thread(camera_id, save_dir):
    global capturing
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)

    if not cap.isOpened():
        print("Warning: failed to open camera!")
        return

    frame_count = 0
    while capturing:
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(save_dir, f"img_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            frame_count += 1
            time.sleep(0.02)

    cap.release()
    print(f"Finish capturing {camera_id}.")


def perform_robot_press(trial_dir):
    global capturing
    print(f"Connecting DOBOT arm ({ROBOT_IP})...")
    robot = SyncRobot(MG400Controller(ROBOT_IP))

    print("Moving to initial position...")
    robot.speed = 15
    robot.move_linear(BASE_INITIAL)
    time.sleep(1)

    # 1. detect contact position
    contact_z = auto_detect_contact(robot, CAMERA_ID)

    # 2. calculate
    dynamic_press_start = (BASE_INITIAL[0], BASE_INITIAL[1], contact_z, BASE_INITIAL[3], BASE_INITIAL[4],
                           BASE_INITIAL[5])
    dynamic_press_end = (BASE_INITIAL[0], BASE_INITIAL[1], contact_z - PRESS_DEPTH, BASE_INITIAL[3], BASE_INITIAL[4],
                         BASE_INITIAL[5])

    print("\nMoving to dynamic pres start...")
    robot.speed = 10
    robot.move_linear(dynamic_press_start)
    time.sleep(1)

    print(f"Start press (Target Z: {contact_z - PRESS_DEPTH:.2f})...")
    capturing = True
    t = threading.Thread(target=capture_thread, args=(CAMERA_ID, trial_dir))
    t.start()
    time.sleep(1)

    robot.speed = 2
    robot.move_linear(dynamic_press_end)
    time.sleep(3)

    capturing = False
    t.join()

    print("Return to initial position...")
    robot.speed = 15
    robot.move_linear(BASE_INITIAL)
    time.sleep(1)


def extract_centroids(image_dir, output_csv):
    BLOCK_SIZE = 73
    C_VALUE = 7
    CIRCLE_CX, CIRCLE_CY, CIRCLE_RADIUS = 312, 233, 169

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image_Name", "Dot_ID", "Centroid_X", "Centroid_Y", "Area"])

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None: continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, BLOCK_SIZE,
                                           C_VALUE)

            mask = np.zeros_like(gray)
            cv2.circle(mask, (CIRCLE_CX, CIRCLE_CY), CIRCLE_RADIUS, 255, -1)
            final_binary = cv2.bitwise_and(binary, mask)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_binary, connectivity=8,
                                                                                    ltype=cv2.CV_32S)

            dot_id = 1
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > 5:
                    cx, cy = centroids[label]
                    writer.writerow([filename, dot_id, round(cx, 2), round(cy, 2), area])
                    dot_id += 1


def track_markers(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df_clean = df[(df['Area'] > 5) & (df['Area'] < 300)].copy()
    images = sorted(df['Image_Name'].unique())

    base_frame = df_clean[df_clean['Image_Name'] == images[0]].copy()
    center_x, center_y = base_frame['Centroid_X'].mean(), base_frame['Centroid_Y'].mean()
    base_frame['Dist_to_Center'] = np.sqrt(
        (base_frame['Centroid_X'] - center_x) ** 2 + (base_frame['Centroid_Y'] - center_y) ** 2)
    base_frame = base_frame.sort_values(by='Dist_to_Center').reset_index(drop=True)
    base_frame['Tracked_ID'] = range(1, len(base_frame) + 1)

    prev_coords = base_frame[['Centroid_X', 'Centroid_Y']].values
    prev_ids = base_frame['Tracked_ID'].values
    tracked_data = [base_frame.drop(columns=['Dist_to_Center'])]

    for img_name in images[1:]:
        curr_frame = df_clean[df_clean['Image_Name'] == img_name].copy()
        curr_coords = curr_frame[['Centroid_X', 'Centroid_Y']].values
        dist_matrix = cdist(prev_coords, curr_coords)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        curr_frame_matched = curr_frame.iloc[col_ind].copy()
        curr_frame_matched['Tracked_ID'] = prev_ids[row_ind]
        tracked_data.append(curr_frame_matched)

        prev_coords = curr_coords[col_ind]
        prev_ids = prev_ids[row_ind]

    final_df = pd.concat(tracked_data, ignore_index=True)
    final_df = final_df.sort_values(by=['Image_Name', 'Tracked_ID'])
    final_df.to_csv(output_csv, index=False)


def calculate_ml_features(tracked_csv):
    df = pd.read_csv(tracked_csv)
    images = sorted(df['Image_Name'].unique())
    df['Time_Step'] = df['Image_Name'].map({img: idx for idx, img in enumerate(images)})

    frame_0 = df[df['Time_Step'] == 0][['Tracked_ID', 'Centroid_X', 'Centroid_Y']].rename(
        columns={'Centroid_X': 'Start_X', 'Centroid_Y': 'Start_Y'})
    df = df.merge(frame_0, on='Tracked_ID', how='left')
    df['Displacement'] = np.sqrt((df['Centroid_X'] - df['Start_X']) ** 2 + (df['Centroid_Y'] - df['Start_Y']) ** 2)

    mean_disp_per_frame = df.groupby('Time_Step')['Displacement'].mean()
    max_frame = mean_disp_per_frame.idxmax()
    max_disp = mean_disp_per_frame.max()

    threshold_90 = max_disp * 0.9
    rise_time_frames = int(mean_disp_per_frame[mean_disp_per_frame >= threshold_90].index[0])

    df_max = df[df['Time_Step'] == max_frame].copy()
    center_x, center_y, fit_radius = 312, 233, 169
    df_max['Radius'] = np.sqrt((df_max['Centroid_X'] - center_x) ** 2 + (df_max['Centroid_Y'] - center_y) ** 2)
    disp_center = df_max[df_max['Radius'] < (fit_radius * 0.3)]['Displacement'].mean()
    disp_rim = df_max[(df_max['Radius'] > (fit_radius * 0.7)) & (df_max['Radius'] <= fit_radius)]['Displacement'].mean()
    dropoff_ratio = disp_center / (disp_rim + 1e-6)

    max_area = df.groupby('Time_Step')['Area'].mean().max()

    return [round(max_disp, 4), round(max_area, 4), round(dropoff_ratio, 4), rise_time_frames]


# 3.
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir = os.path.join(BLACKBOX_SAVE_DIR, f"Test_{timestamp}")
    os.makedirs(trial_dir, exist_ok=True)

    print("\n[STEP 1/4] Use DOBOT arm to press...")
    perform_robot_press(trial_dir)

    print("\n[Step 2/4] Preprocess dataset...")
    centroids_csv = os.path.join(trial_dir, "centroids_data.csv")
    extract_centroids(trial_dir, centroids_csv)

    print("\n[Step 3/4] Track points...")
    tracked_csv = os.path.join(trial_dir, "tracked_centroids_data.csv")
    track_markers(centroids_csv, tracked_csv)

    print("\n[Step 4/4] Extract features...")
    features = calculate_ml_features(tracked_csv)

    print("\nLoad model and predict...")
    try:
        rf_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"Error! Fail to load model: {e}")
        return

    features_df = pd.DataFrame([features],
                               columns=['Max_Displacement', 'Max_Area', 'Dropoff_Ratio',
                                        'Rise_Time_Frames'])
    features_scaled = scaler.transform(features_df)
    predicted_hardness = rf_model.predict(features_scaled)[0]

    print(f"System Estimated Hardness: {predicted_hardness:.2f} Shore 00")

    true_hardness_input = input(
        "Please measure the object with a Durometer and enter the true Shore 00 value (or type 'q'): ")

    if true_hardness_input.lower() != 'q':
        true_hardness = float(true_hardness_input)

        results_csv = os.path.join(BLACKBOX_SAVE_DIR, "blackbox_results_log.csv")
        file_exists = os.path.isfile(results_csv)

        with open(results_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "True_Hardness", "Predicted_Hardness", "Absolute_Error"])

            abs_error = abs(true_hardness - predicted_hardness)
            writer.writerow([timestamp, true_hardness, predicted_hardness, abs_error])

        print(f"Result logged! Error: {abs_error:.2f} Shore 00")

        df_results = pd.read_csv(results_csv)
        if len(df_results) > 1:
            y_true = df_results["True_Hardness"]
            y_pred = df_results["Predicted_Hardness"]

            current_mae = mean_absolute_error(y_true, y_pred)
            current_r2 = r2_score(y_true, y_pred)

            print(f"\nBlack-Box Test Performance (Total {len(df_results)} objects):")
            print(f"MAE: {current_mae:.2f} Shore 00")
            print(f"R² Score: {current_r2:.4f}")


if __name__ == "__main__":
    main()