import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_spatial_dropoff(base_dir, save_dir, target_trials=None):
    category_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "Shore *")) if os.path.isdir(d)])
    if not category_dirs: return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(category_dirs)))

    center_x, center_y = 312, 233
    bins = np.arange(0, 180, 20)
    bin_centers = bins[:-1] + np.diff(bins) / 2

    for idx, cat_dir in enumerate(category_dirs):
        category_name = os.path.basename(cat_dir)
        csv_files = glob.glob(os.path.join(cat_dir, "Trial_*", "tracked_centroids_data.csv"))

        if target_trials:
            csv_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)) in target_trials]

        category_binned_disp = []

        for file_path in csv_files:
            df = pd.read_csv(file_path)
            if 'Time_Step' not in df.columns:
                images = sorted(df['Image_Name'].unique())
                df['Time_Step'] = df['Image_Name'].map({img: i for i, img in enumerate(images)})
            if category_name == "Shore 00-20":
                df = df[df['Time_Step'] < 89].copy()

            frame_0 = df[df['Time_Step'] == 0].set_index('Tracked_ID')[['Centroid_X', 'Centroid_Y']]
            max_frame_idx = df.groupby('Time_Step').apply(
                lambda x: np.mean(
                    np.sqrt((x['Centroid_X'].values - frame_0.loc[x['Tracked_ID'], 'Centroid_X'].values) ** 2 +
                            (x['Centroid_Y'].values - frame_0.loc[x['Tracked_ID'], 'Centroid_Y'].values) ** 2))
            ).idxmax()

            df_max = df[df['Time_Step'] == max_frame_idx].copy()
            df_max = df_max.merge(frame_0, on='Tracked_ID', suffixes=('', '_Start'))
            df_max['Displacement'] = np.sqrt((df_max['Centroid_X'] - df_max['Centroid_X_Start']) ** 2 +
                                             (df_max['Centroid_Y'] - df_max['Centroid_Y_Start']) ** 2)
            df_max = df_max[df_max['Displacement'] < 20]

            df_max['Radius'] = np.sqrt((df_max['Centroid_X'] - center_x) ** 2 + (df_max['Centroid_Y'] - center_y) ** 2)
            df_max['Distance_Bin'] = pd.cut(df_max['Radius'], bins=bins, labels=False, include_lowest=True)
            binned_disp = df_max.groupby('Distance_Bin')['Displacement'].mean()
            category_binned_disp.append(binned_disp)

        if category_binned_disp:
            cat_mean = pd.concat(category_binned_disp, axis=1).mean(axis=1)
            valid_bins = cat_mean.index.astype(int)
            plt.plot(bin_centers[valid_bins], cat_mean.values, color=colors[idx], linewidth=2.5, marker='o',
                     label=category_name)

    plt.axvline(x=169 * 0.3, color='gray', linestyle='--', alpha=0.5, label='Inner Threshold (30%)')
    plt.axvline(x=169 * 0.7, color='gray', linestyle='-.', alpha=0.5, label='Outer Threshold (70%)')
    plt.title("Spatial Drop-off: Displacement vs. Distance from Center", fontsize=15, pad=15)
    plt.xlabel("Distance from Contact Center (Pixels)", fontsize=12)
    plt.ylabel("Mean Radial Displacement (Pixels)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "Spatial_Dropoff.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Save to: {save_path}")


def visualize_all_categories_overlay(base_dir, save_dir, target_trials=None):
    category_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, "Shore *")) if os.path.isdir(d)])
    if not category_dirs: return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(category_dirs)))

    for idx, cat_dir in enumerate(category_dirs):
        category_name = os.path.basename(cat_dir)
        csv_files = glob.glob(os.path.join(cat_dir, "Trial_*", "tracked_centroids_data.csv"))

        if target_trials:
            csv_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)) in target_trials]

        all_trials_disp = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            if 'Time_Step' not in df.columns:
                images = sorted(df['Image_Name'].unique())
                df['Time_Step'] = df['Image_Name'].map({img: i for i, img in enumerate(images)})

            frame_0 = df[df['Time_Step'] == 0][['Tracked_ID', 'Centroid_X', 'Centroid_Y']]
            frame_0 = frame_0.rename(columns={'Centroid_X': 'Base_X', 'Centroid_Y': 'Base_Y'})
            df = df.merge(frame_0, on='Tracked_ID', how='left')
            df['Radial_Disp'] = np.sqrt((df['Centroid_X'] - df['Base_X']) ** 2 + (df['Centroid_Y'] - df['Base_Y']) ** 2)

            trial_mean = df.groupby('Time_Step')['Radial_Disp'].mean().reset_index()
            all_trials_disp.append(trial_mean)

        if all_trials_disp:
            master_df = pd.concat(all_trials_disp, ignore_index=True)
            summary = master_df.groupby('Time_Step')['Radial_Disp'].mean().reset_index()
            plt.plot(summary['Time_Step'], summary['Radial_Disp'], color=colors[idx], linewidth=2.5,
                     label=category_name)

    plt.title(f"Overlay of Mean Radial Displacement Across All {len(category_dirs)} Categories", fontsize=16, pad=15)
    plt.xlabel("Time Step (Frames)", fontsize=13)
    plt.ylabel("Mean Radial Displacement (Pixels)", fontsize=13)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "Overlay_All_Categories.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Save to: {save_path}")


def visualize_category_trials(category_dir, save_dir, target_trials=None):
    search_pattern = os.path.join(category_dir, "Trial_*", "tracked_centroids_data.csv")
    csv_files = glob.glob(search_pattern)

    if target_trials:
        csv_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)) in target_trials]

    if not csv_files: return

    category_name = os.path.basename(os.path.normpath(category_dir))
    all_trials_stats = []

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    for file_path in csv_files:
        trial_name = os.path.basename(os.path.dirname(file_path))
        df = pd.read_csv(file_path)
        if 'Time_Step' not in df.columns:
            images = sorted(df['Image_Name'].unique())
            frame_mapping = {img: idx for idx, img in enumerate(images)}
            df['Time_Step'] = df['Image_Name'].map(frame_mapping)

        if category_name == "Shore 00-20":
            df = df[df['Time_Step'] < 89].copy()

        base_frame = df[df['Time_Step'] == 0]['Image_Name'].iloc[0]
        base_coords = df[df['Image_Name'] == base_frame][['Tracked_ID', 'Centroid_X', 'Centroid_Y', 'Area']]
        base_coords = base_coords.rename(columns={'Centroid_X': 'Base_X', 'Centroid_Y': 'Base_Y', 'Area': 'Base_Area'})
        df = df.merge(base_coords[['Tracked_ID', 'Base_X', 'Base_Y']], on='Tracked_ID', how='left')
        df['Radial_Disp'] = np.sqrt((df['Centroid_X'] - df['Base_X']) ** 2 + (df['Centroid_Y'] - df['Base_Y']) ** 2)

        trial_stats = df.groupby('Time_Step').agg(
            Mean_Radial_Disp=('Radial_Disp', 'mean'), Mean_Area=('Area', 'mean')
        ).reset_index()
        trial_stats['Trial'] = trial_name
        all_trials_stats.append(trial_stats)

    master_df = pd.concat(all_trials_stats, ignore_index=True)
    summary_stats = master_df.groupby('Time_Step').agg(
        Disp_Mean=('Mean_Radial_Disp', 'mean'), Disp_Std=('Mean_Radial_Disp', 'std'),
        Area_Mean=('Mean_Area', 'mean'), Area_Std=('Mean_Area', 'std')
    ).reset_index()

    for trial_name, trial_data in master_df.groupby('Trial'):
        ax1.plot(trial_data['Time_Step'], trial_data['Mean_Radial_Disp'], color='blue', alpha=0.15)
    ax1.plot(summary_stats['Time_Step'], summary_stats['Disp_Mean'], color='darkblue', linewidth=2.5,
             label='Mean Trajectory')
    ax1.fill_between(summary_stats['Time_Step'], summary_stats['Disp_Mean'] - summary_stats['Disp_Std'],
                     summary_stats['Disp_Mean'] + summary_stats['Disp_Std'], color='blue', alpha=0.2,
                     label='±1 Std Dev')

    max_disp = summary_stats['Disp_Mean'].max()
    threshold_90 = max_disp * 0.9
    rise_time_idx = summary_stats[summary_stats['Disp_Mean'] >= threshold_90].index[0]
    rise_time_frame = summary_stats.loc[rise_time_idx, 'Time_Step']

    ax1.axhline(y=threshold_90, color='red', linestyle='--', alpha=0.6, label=f'90% Max Disp ({threshold_90:.2f})')
    ax1.axvline(x=rise_time_frame, color='orange', linestyle='--', alpha=0.8,
                label=f'Rise Time (Frame {int(rise_time_frame)})')

    ax1.set_title(f"[{category_name}] Radial Displacement", fontsize=14)
    ax1.set_xlabel("Time Step (Frames)")
    ax1.set_ylabel("Displacement (Pixels)")
    ax1.legend()

    for trial_name, trial_data in master_df.groupby('Trial'):
        ax2.plot(trial_data['Time_Step'], trial_data['Mean_Area'], color='green', alpha=0.15)
    ax2.plot(summary_stats['Time_Step'], summary_stats['Area_Mean'], color='darkgreen', linewidth=2.5,
             label='Mean Area')
    ax2.fill_between(summary_stats['Time_Step'], summary_stats['Area_Mean'] - summary_stats['Area_Std'],
                     summary_stats['Area_Mean'] + summary_stats['Area_Std'], color='green', alpha=0.2,
                     label='±1 Std Dev')

    ax2.set_title(f"[{category_name}] Marker Area", fontsize=14)
    ax2.set_xlabel("Time Step (Frames)")
    ax2.set_ylabel("Area (Pixels^2)")
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{category_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Save to: {save_path}")


if __name__ == "__main__":
    BASE_DIRECTORY = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images"

    TARGET_TRIALS = [f"Trial_{i:02d}" for i in range(1, 21)]

    SAVE_DIRECTORY = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\visual results"

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    print(f"Generating charts for {TARGET_TRIALS}...")
    print(f"Images will be saved to: {SAVE_DIRECTORY}\n")

    visualize_spatial_dropoff(BASE_DIRECTORY, SAVE_DIRECTORY, target_trials=TARGET_TRIALS)
    visualize_all_categories_overlay(BASE_DIRECTORY, SAVE_DIRECTORY, target_trials=TARGET_TRIALS)

    category_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIRECTORY, "Shore *")) if os.path.isdir(d)])
    for cat_dir in category_dirs:
        visualize_category_trials(cat_dir, SAVE_DIRECTORY, target_trials=TARGET_TRIALS)

    print("\nFinished!")