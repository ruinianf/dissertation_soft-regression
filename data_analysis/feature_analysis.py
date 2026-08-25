import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('ML_feature_matrix.csv')

    # We need a numeric representation of hardness to calculate correlation.
    # We will map the categorical labels to a rough numerical scale based on Shore 00 values.
    hardness_mapping = {
        'Shore 00-10': 10,
        'Shore 00-15': 15,
        'Shore 00-20': 20,
        'Shore 00-25': 25,
        'Shore 00-30': 30,
        'Shore 00-40': 40,
        'Shore 00-50': 50,
        'Shore 00-60': 60,
        'Shore 00-70': 70,
        'Shore 00-80': 80,
        'Shore 00-90': 90,
    }

    df['Hardness_Numeric'] = df['Hardness_Label'].map(hardness_mapping)

    # Define features to analyze
    features = ['Max_Displacement', 'Max_Area', 'Dropoff_Ratio', 'Rise_Time_Frames']

    print("--- Correlation Analysis with Hardness ---")
    results = []
    for feature in features:
        # Pearson (linear correlation)
        pearson_corr, p_value_p = pearsonr(df[feature], df['Hardness_Numeric'])
        # Spearman (monotonic rank correlation - better for non-linear relationships)
        spearman_corr, p_value_s = spearmanr(df[feature], df['Hardness_Numeric'])

        results.append({
            'Feature': feature,
            'Pearson_Corr': round(pearson_corr, 4),
            'Pearson_p_value': round(p_value_p, 4),
            'Spearman_Corr': round(spearman_corr, 4),
            'Spearman_p_value': round(p_value_s, 4)
        })

    corr_df = pd.DataFrame(results)
    print(corr_df.to_string(index=False))

    print("\nSummary Statistics to assess variability...")
    desc = df[features].describe().round(3)
    print(desc)

except Exception as e:
    print(f"Error during analysis: {e}")