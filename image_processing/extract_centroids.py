import os
import cv2
import numpy as np
import glob
import csv


def process_single_trial(trial_dir, block_size, c_value, circle_cx, circle_cy, circle_radius):
    output_dir = os.path.join(trial_dir, "processed_images")
    csv_path = os.path.join(trial_dir, "centroids_data.csv")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image_Name", "Dot_ID", "Centroid_X", "Centroid_Y", "Area"])

        image_paths = glob.glob(os.path.join(trial_dir, "*.jpg"))
        if not image_paths:
            print(f"Warning: No images found in {trial_dir}")
            return

        print(f"[{os.path.basename(trial_dir)}] Found {len(image_paths)} images, processing...")

        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # preprocess
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, block_size, c_value)
            mask = np.zeros_like(gray)
            cv2.circle(mask, (circle_cx, circle_cy), circle_radius, 255, -1)
            final_binary = cv2.bitwise_and(binary, mask)

            # save
            cv2.imwrite(os.path.join(output_dir, filename), final_binary)

            # feature
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                final_binary, connectivity=8, ltype=cv2.CV_32S
            )

            dot_id = 1
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area > 5:
                    cx, cy = centroids[label]
                    writer.writerow([filename, dot_id, round(cx, 2), round(cy, 2), area])
                    dot_id += 1


def main():
    # 1.
    base_dir = r"C:\Users\Nick\Desktop\UK\learning\dissertation\experiment\images\time series images\Shore 00-90" ##

    BLOCK_SIZE = 73
    C_VALUE = 7
    CIRCLE_CX = 312
    CIRCLE_CY = 233
    CIRCLE_RADIUS = 169

    # 2. go through
    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return

    for folder_name in os.listdir(base_dir):
        trial_dir = os.path.join(base_dir, folder_name)

        if os.path.isdir(trial_dir) and folder_name.startswith("Trial_"):
            process_single_trial(trial_dir, BLOCK_SIZE, C_VALUE, CIRCLE_CX, CIRCLE_CY, CIRCLE_RADIUS)

    print("\nALL TRIALS PROCESSED SUCCESSFULLY!")


if __name__ == '__main__':
    main()