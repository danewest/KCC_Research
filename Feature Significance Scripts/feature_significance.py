import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import shap
from pandas import DataFrame
import sklearn
import numpy as np

def calculateFeatureImportance(df: DataFrame , target_col: str, excluded_features: list[str] = ['VT90_TAIR_diff', 'VT90_VT20_diff', 'VT90', 'VT20', 'TAIR', 'SM04', 'ST04', 'UTCTimestampCollected', 'NetSiteAbbrev', 'County']):
    # Prepare X and y
    X = df.drop(columns=[target_col] + excluded_features)
    y = df[target_col]

    # Split into train/test for more accurate performance results
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Decision tree training/testing...\n")
    # Decision Tree
    print("Initializing decision tree regressor...")
    dt = DecisionTreeRegressor(random_state=42)
    print("Starting decision tree fitting...")
    dt.fit(X_train, y_train)
    print("Decision tree fitting complete. Starting prediction...")
    y_pred_dt = dt.predict(X_test)
    print("Prediction complete. Calculating feature importances...")
    dt_imp = pd.Series(dt.feature_importances_, index=X.columns)
    print("DT feature importances calculated.")

    print("Random forest training/testing...\n")
    # Random Forest
    print("Initializing random forest regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("Starting random forest fitting...")
    rf.fit(X_train, y_train)
    print("Random forest fitting complete. Starting prediction...")
    y_pred_rf = rf.predict(X_test)
    print("Prediction complete. Starting permutation importance...")
    rf_perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    print("Permutation importance calculation complete.")
    rf_imp = pd.Series(rf_perm.importances_mean, index=X.columns)
    print("RF importance series created.")

    print("SHAP training/testing...\n")
    # SHAP values (using Random Forest)
    print("Initializing SHAP explainer...")
    explainer = shap.PermutationExplainer(rf.predict, X_test, n_jobs=-1)
    print("Explainer initialized. Starting SHAP value calculation...")
    sample_size = 1000
    print(f"Sampling {sample_size} rows from X_test for SHAP explanation...")

    # Use pandas' sample() method for simplicity
    X_test_sample = X_test.sample(n=sample_size, random_state=42)

    print("Sample created. Now calculating SHAP values...")
    shap_values = explainer(X_test_sample) # This will run on a smaller dataset
    print("SHAP values calculated. Starting plots...")

    print("Plotting decision trees...\n")
    # --- Feature Importance Plots --- #
    # Plot Decision Tree
    plt.figure(figsize=(10,6))
    dt_imp.nlargest(10).plot.barh()
    plt.title("Feature Importance (Decision Tree)")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("Plotting random forest...\n")
    # Plot Random Forest Permutation Importance
    plt.figure(figsize=(10, 6))
    rf_imp.nlargest(10).plot.barh()
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Mean Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print("Plotting SHAP...\n")
    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", max_display=10)
    print(" ")

    # ---- Model Performance Comparison ---- #
    dt_r2 = r2_score(y_test, y_pred_dt)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))

    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    performance_data = {
        'Decision Tree': [dt_r2, dt_rmse],
        'Random Forest': [rf_r2, rf_rmse]
    }
    performance_df = pd.DataFrame(performance_data, index=['R²', 'RMSE'])

    # Plotting Model Performance
    plt.figure(figsize=(10,6))

    performance_df.T.plot(kind='bar', ax=plt.gca(), width=0.8)

    plt.title("Model Performance Comparison (R² and RMSE)", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model Type", fontsize=12)
    plt.xticks(rotation=0, ha='center', fontsize=10) # Keep labels horizontal
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

if __name__ == "__main__":
    try:
        # Load the CSVs
        # GRDR_df = pd.read_csv("../Processed Data/GRDR_interpolated.csv")
        WOOD_df = pd.read_csv("../Processed Data/WOOD_interpolated.csv")

        # Create difference features for GRDR
        GRDR_df['VT90_TAIR_diff'] = GRDR_df['VT90'] - GRDR_df['TAIR']
        GRDR_df['VT90_VT20_diff'] = GRDR_df['VT90'] - GRDR_df['VT20']

        # Create difference features for WOOD
        WOOD_df['VT90_TAIR_diff'] = WOOD_df['VT90'] - WOOD_df['TAIR']
        WOOD_df['VT90_VT20_diff'] = WOOD_df['VT90'] - WOOD_df['VT20']

        # Execution
        print("Calculating feature significance for GRDR")
        print("First, VT90 - TAIR...\n")
        calculateFeatureImportance(GRDR_df, target_col='VT90_TAIR_diff')

        print("VT90 - TAIR done. Now, VT90 - VT20...\n")
        calculateFeatureImportance(GRDR_df, target_col='VT90_VT20_diff')

        print("GRDR done. Now calculating feature significance for WOOD.")
        print("First, VT90 - TAIR...\n")
        calculateFeatureImportance(WOOD_df, target_col='VT90_TAIR_diff')

        print("VT90 - TAIR done. Now, VT90 - VT20...\n")
        calculateFeatureImportance(WOOD_df, target_col='VT90_VT20_diff')

        print("Feature significance calculations completed.\n")
    except Exception as e:
        print("Something went wrong:", e)