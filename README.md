# Vision-Based Continuous Softness Regression from a Single Tactile Press

This repository contains the source code for my MSc Biorobotics dissertation project at the Bristol Robotics Laboratory (BRL). 

The project investigates whether a soft optical tactile sensor can accurately estimate a material's continuous stiffness (Shore 00 scale) from a single, straightforward 5.0 mm robotic press. By capturing the internal marker displacement dynamics during the deformation process, the system extracts spatial and temporal features to train a Random Forest regressor.

## Hardware Setup
- Robotic Arm: DOBOT MG400 (4-DoF)
- Sensor: dome-shaped soft tactile sensor
- Camera Interface: Internal USB endoscope camera tracking a marker array

## Repository Structure

The codebase is organised into sequential stages of the experimental pipeline:

### 1. Data Collection
- `robot_press_capture.py`: Coordinates the DOBOT MG400 to perform a 5.0 mm indentation and captures the time series interaction frames. Includes a random Yaw-axis perturbation to augment spatial data. (Requires the DOBOT `cri` environment).

### 2. Image Processing
- `thresholdGUI.py`: A live-tuning GUI to determine the optimal block size and C-value for adaptive thresholding.
- `fitting_circle.py`: A manual 6-point calibration tool to map the sensor's physical boundary.
- `extract_centroids.py`: Batch processes trial images to extract the X, Y coordinates and area of internal markers.
- `track_markers.py`: Links marker centroids across continuous frames using the Hungarian algorithm to build displacement trajectories.

### 3. Data Analysis & Feature Engineering
- `extract_features.py`: Computes the 4 physical features.
- `feature_analysis.py`: Evaluates the Pearson and Spearman correlation between features and physical hardness.
- `results.py`: Generates visualisations of radial displacement, spatial drop-off gradients, and tracking validation.

### 4. Machine Learning
- `data_utils.py`: Helper script for standardising inputs using `StandardScaler`.
- `classification.py`: Baseline classification model with 5-fold cross-validation and feature importance ranking.
- `regression.py`: The primary Random Forest regressor script. Trains the final continuous estimation model and saves the weights (`.joblib`).

### 5. Test Deployment
- `adv.py`: The end-to-end black-box testing script. It autonomously approaches an unknown object, detects the zero-stress contact point via visual feedback, executes the press, processes the frames, and predicts the Shore 00 hardness in real-time. (Requires the DOBOT `cri` environment).

## Dependencies
A list of required packages is provided in `requirements.txt`.
