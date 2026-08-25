import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_scale_data(train_csv_path, test_csv_path):
    """
    load and preprocess dataset
    return: X_train_scaled, X_test_scaled, y_train_raw, y_test_raw, features
    """
    try:
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Feature matrix file {train_csv_path} not found!")

    features = ['Max_Displacement', 'Max_Area', 'Dropoff_Ratio', 'Rise_Time_Frames']

    # extract features and labels
    X_train = df_train[features]
    y_train_raw = df_train['Hardness_Label']

    X_test = df_test[features]
    y_test_raw = df_test['Hardness_Label']

    # features standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

    return X_train_scaled, X_test_scaled, y_train_raw, y_test_raw, features, scaler