import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_regression_cv(train_csv_path):
    print("Random Forest (5-Fold Cross Validation)")

    # 1. load dataset
    try:
        df = pd.read_csv(train_csv_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    features = ['Max_Displacement', 'Max_Area', 'Dropoff_Ratio', 'Rise_Time_Frames']
    X_raw = df[features]
    y_raw = df['Hardness_Label']

    # 2. Map text labels to continuous Shore 00 physical values
    hardness_mapping = {
        'Shore 00-10': 10.0,
        'Shore 00-15': 15.0,
        'Shore 00-20': 20.0,
        'Shore 00-25': 25.0,
        'Shore 00-30': 30.0,
        'Shore 00-40': 40.0,
        'Shore 00-50': 50.0,
        'Shore 00-60': 60.0,
        'Shore 00-70': 70.0,
        'Shore 00-80': 80.0,
        'Shore 00-90': 90.0,
    }
    y = y_raw.map(hardness_mapping)

    # 3. standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X, columns=features)

    # 4. initialize random forest regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # 5. 5-fold cross validation
    print("\nExecute 5-fold cross validation...")
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring_metrics = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']
    cv_results = cross_validate(rf_reg, X, y, cv=cv_strategy, scoring=scoring_metrics)

    mae_scores = -cv_results['test_neg_mean_absolute_error']
    rmse_scores = -cv_results['test_neg_root_mean_squared_error']

    print("\nValidation results:")
    for i in range(5):
        print(f" - Fold {i + 1} MAE: {mae_scores[i]:.2f} Shore 00 | RMSE: {rmse_scores[i]:.2f} Shore 00")

    print(f"\n平均绝对误差 (MAE): {mae_scores.mean():.2f} Shore 00 (standard deviation: ±{mae_scores.std():.2f})")
    print(f"均方根误差 (RMSE): {rmse_scores.mean():.2f} Shore 00")

    # 6.
    y_pred_cv = cross_val_predict(rf_reg, X, y, cv=cv_strategy)
    overall_r2 = r2_score(y, y_pred_cv)
    print(f"整体拟合优度 (R² Score): {overall_r2:.4f}")

    # 7.
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 7))

    sns.scatterplot(x=y, y=y_pred_cv, alpha=0.6, edgecolor=None, color='#4C72B0')

    min_val = min(y.min(), y_pred_cv.min()) - 2
    max_val = max(y.max(), y_pred_cv.max()) + 2
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')

    plt.title(f"Regression CV: True vs Predicted Hardness\n(Avg MAE: {mae_scores.mean():.2f}, R²: {overall_r2:.2f})",
              fontsize=14, pad=15)
    plt.xlabel("True Hardness (Shore 00)", fontsize=12)
    plt.ylabel("Predicted Hardness (Shore 00)", fontsize=12)
    plt.legend()
    plt.tight_layout()

    save_path = "CV_Regression_Plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSave to: {save_path}")


    print("\nSave model weights...")
    final_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_rf_model.fit(X, y)

    model_filename = "rf_regressor_final.joblib"
    joblib.dump(final_rf_model, model_filename)

    scaler_filename = "feature_scaler.joblib"
    joblib.dump(scaler, scaler_filename)

    print(f"Model saved to: {model_filename}")
    print(f"Standardiser saved to: {scaler_filename}")


if __name__ == "__main__":
    TRAIN_MATRIX_PATH = "ML_feature_matrix.csv"
    run_regression_cv(TRAIN_MATRIX_PATH)