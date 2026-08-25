import os
import glob
import pandas as pd
import numpy as np


def extract_features(csv_path):
    df = pd.read_csv(csv_path)

    # 1. initialize Time_Step
    if 'Time_Step' not in df.columns:
        images = sorted(df['Image_Name'].unique())
        frame_mapping = {img: idx for idx, img in enumerate(images)}
        df['Time_Step'] = df['Image_Name'].map(frame_mapping)

    # 2. absolute displacement
    frame_0 = df[df['Time_Step'] == 0][['Tracked_ID', 'Centroid_X', 'Centroid_Y']].copy()
    frame_0 = frame_0.rename(columns={'Centroid_X': 'Start_X', 'Centroid_Y': 'Start_Y'})
    df = df.merge(frame_0, on='Tracked_ID', how='left')

    df['Displacement'] = np.sqrt((df['Centroid_X'] - df['Start_X']) ** 2 +
                                 (df['Centroid_Y'] - df['Start_Y']) ** 2)

    # 3. global displacement curve
    mean_disp_per_frame = df.groupby('Time_Step')['Displacement'].mean()
    max_frame = mean_disp_per_frame.idxmax()
    max_disp = mean_disp_per_frame.max()

    # Feature A: Rise Time (T_90)
    threshold_90 = max_disp * 0.9
    rise_time_frames = mean_disp_per_frame[mean_disp_per_frame >= threshold_90].index[0]

    # Feature B: Drop-off Ratio - centre / outer
    df_max = df[df['Time_Step'] == max_frame].copy()

    center_x, center_y = 312, 233
    fit_radius = 169

    df_max['Dist_to_Phys_Center'] = np.sqrt((df_max['Centroid_X'] - center_x) ** 2 +
                                            (df_max['Centroid_Y'] - center_y) ** 2)

    inner_threshold = fit_radius * 0.3
    outer_threshold = fit_radius * 0.7

    center_markers = df_max[df_max['Dist_to_Phys_Center'] < inner_threshold]
    rim_markers = df_max[(df_max['Dist_to_Phys_Center'] > outer_threshold) &
                         (df_max['Dist_to_Phys_Center'] <= fit_radius)]

    disp_center = center_markers['Displacement'].mean()
    disp_rim = rim_markers['Displacement'].mean()

    dropoff_ratio = disp_center / (disp_rim + 1e-6)

    # Feature C: Max Area
    max_area = df.groupby('Time_Step')['Area'].mean().max()

    path_parts = os.path.normpath(csv_path).split(os.sep)
    try:
        label = [p for p in path_parts if "Shore" in p][0]
    except IndexError:
        label = "Unknown"

    return {
        'Hardness_Label': label,
        'Trial_File': os.path.basename(os.path.dirname(csv_path)),
        'Max_Displacement': round(max_disp, 4),
        'Max_Area': round(max_area, 4),
        'Dropoff_Ratio': round(dropoff_ratio, 4),
        'Rise_Time_Frames': int(rise_time_frames)
    }


def batch_process_features(base_directory, output_csv, target_trials=None):
    search_pattern = os.path.join(base_directory, "**", "tracked_centroids_data.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        print("Not find csv files!")
        return

    if target_trials:
        filtered_files = []
        for file_path in csv_files:
            trial_folder_name = os.path.basename(os.path.dirname(file_path))

            if trial_folder_name in target_trials:
                filtered_files.append(file_path)

        csv_files = filtered_files
        print(f"Current trial list: {target_trials}")

    print(f"Find {len(csv_files)} csv files，start to extract features...")

    features_list = []
    for i, file_path in enumerate(csv_files):
        try:
            features = extract_features(file_path)
            features_list.append(features)
        except Exception as e:
            print(f"path error at {file_path}: {e}")

    feature_matrix_df = pd.DataFrame(features_list)
    feature_matrix_df.to_csv(output_csv, index=False)

    print(f"Finished! {len(features_list)} trials in total.")
    print(f"Save to: {output_csv}\n")


if __name__ == "__main__":
    BASE_DIR = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images"
    OUTPUT_DIR = r"C:\Users\Nick\Desktop\UK\learning\dissertation\code\tactile-toolbox-main\python\images"

    print("Saving (ML_feature_matrix.csv)")
    TRAIN_TRIALS = [f"Trial_{i:02d}" for i in range(1, 21)]
    OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "ML_feature_matrix.csv")

    batch_process_features(BASE_DIR, OUTPUT_TRAIN, target_trials=TRAIN_TRIALS)