import pandas as pd
import numpy as np
import os
import re
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def track_markers_radially(input_csv, output_csv, max_area=300):
    path_parts = os.path.normpath(input_csv).split(os.sep)

    trial_folder = path_parts[-2]
    trial_id_match = re.search(r'\d+', trial_folder)
    trial_id = int(trial_id_match.group()) if trial_id_match else -1

    label_folder = path_parts[-3]
    hardness_label = label_folder.replace("Shore ", "")

    print(f"hardness_label: {hardness_label}, trial_id: Trial_{trial_id}")

    df = pd.read_csv(input_csv)
    df_clean = df[(df['Area'] > 5) & (df['Area'] < max_area)].copy()

    images = sorted(df['Image_Name'].unique())

    if len(images) == 0:
        print("No data in CSV!")
        return

    base_frame = df_clean[df_clean['Image_Name'] == images[0]].copy()

    center_x = base_frame['Centroid_X'].mean()
    center_y = base_frame['Centroid_Y'].mean()

    distances = np.sqrt((base_frame['Centroid_X'] - center_x) ** 2 + (base_frame['Centroid_Y'] - center_y) ** 2)
    base_frame['Dist_to_Center'] = distances
    base_frame = base_frame.sort_values(by='Dist_to_Center').reset_index(drop=True)
    base_frame['Tracked_ID'] = range(1, len(base_frame) + 1)

    prev_coords = base_frame[['Centroid_X', 'Centroid_Y']].values
    prev_ids = base_frame['Tracked_ID'].values

    tracked_data = [base_frame.drop(columns=['Dist_to_Center'])]

    print(f"centroid: X={center_x:.2f}, Y={center_y:.2f}")
    print(f"The base frame {images[0]} has {len(base_frame)} key points.")
    print("Start frame tracking...")

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
    final_df['Trial_ID'] = trial_id
    final_df['Hardness_Label'] = hardness_label

    cols = ['Trial_ID', 'Hardness_Label', 'Image_Name', 'Tracked_ID', 'Centroid_X', 'Centroid_Y', 'Area']
    final_df = final_df[cols]
    final_df = final_df.sort_values(by=['Image_Name', 'Tracked_ID'])

    final_df.to_csv(output_csv, index=False)
    print(f"Finished! Save to {output_csv}\n")


if __name__ == "__main__":
    base_dir = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images\Shore 00-90"  ##

    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
    else:
        # go through
        for folder_name in os.listdir(base_dir):
            trial_dir = os.path.join(base_dir, folder_name)

            if os.path.isdir(trial_dir) and folder_name.startswith("Trial_"):
                input_csv = os.path.join(trial_dir, "centroids_data.csv")
                output_csv = os.path.join(trial_dir, "tracked_centroids_data.csv")

                if os.path.exists(input_csv):
                    print(f"Tracking for {folder_name}...")
                    track_markers_radially(input_csv, output_csv)
                else:
                    print(f"Warning: {input_csv} not found, run batch process first.")

        print("ALL TRACKING COMPLETED SUCCESSFULLY!")