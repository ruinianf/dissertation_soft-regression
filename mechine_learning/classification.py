import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold


def run_classification_cv(train_csv_path):
    print("Random Forest (5-Fold Cross Validation)...")

    # 1. load dataset
    try:
        df = pd.read_csv(train_csv_path)
    except FileNotFoundError as e:
        print(f"File not found!{e}")
        return

    features = ['Max_Displacement', 'Max_Area', 'Dropoff_Ratio', 'Rise_Time_Frames']

    X_raw = df[features]
    y_raw = df['Hardness_Label']

    # 2. ecode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_

    # 3. standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    X = pd.DataFrame(X, columns=features)

    # 4. initialize random forest classification
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # 5. 5-fold cross validation
    print("\nExecute 5-fold cross validation...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(rf, X, y, cv=cv_strategy, scoring='accuracy')

    print("\nValidation results:")
    for i, score in enumerate(cv_scores):
        print(f" - Accuracy of fold {i + 1}: {score * 100:.2f}%")
    print(f"\nAverage accuracy: {cv_scores.mean() * 100:.2f}% (standard deviation: ±{cv_scores.std() * 100:.2f}%)")

    # 6. get results
    y_pred_cv = cross_val_predict(rf, X, y, cv=cv_strategy)

    print(classification_report(y, y_pred_cv, target_names=class_names))

    # 7. draw confusion matrix
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    cm = confusion_matrix(y, y_pred_cv)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f"Cross-Validation Confusion Matrix (Avg Acc: {cv_scores.mean() * 100:.1f}%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Hardness", fontsize=12)
    plt.ylabel("True Hardness", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    save_path = "CV_Confusion_Matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nConfusion matrix saved to: {save_path}")

    print("\nFeature Importances...")
    rf_final = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_final.fit(X, y)
    importances = rf_final.feature_importances_

    feature_importance_dict = dict(zip(features, importances))
    sorted_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

    for feature_name, importance_score in sorted_importances:
        print(f" - {feature_name}: {importance_score * 100:.2f}%")

    plt.figure(figsize=(8, 5))
    names = [item[0] for item in sorted_importances]
    scores = [item[1] for item in sorted_importances]

    sns.barplot(x=scores, y=names, palette='viridis')
    plt.title("Random Forest Feature Importances", fontsize=14, pad=15)
    plt.xlabel("Importance Score (Percentage)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.xlim(0, 1)
    plt.tight_layout()

    save_path_imp = "Feature_Importances.png"
    plt.savefig(save_path_imp, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSave to: {save_path_imp}")


if __name__ == "__main__":
    TRAIN_MATRIX_PATH = "ML_feature_matrix.csv"
    run_classification_cv(TRAIN_MATRIX_PATH)