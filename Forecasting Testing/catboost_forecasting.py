import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def load_data(file_path: str):
    df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
    return df

def train_test_split(df, date_col, split_date, test_end):
    train = df[df[date_col] < split_date]
    test = df[(df[date_col] >= split_date) & (df[date_col] <= test_end)]

    # Ensure these features exist in the processed CSVs!
    features = ["hour", "dayofweek", "month", "day", "year", "minute",
                "inversion_diff_lag1", "rolling_mean_3"]
    target = "VT90_VT20_diff"

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    return X_train, X_test, y_train, y_test, test

def evaluate_and_plot(y_true, y_pred, test_df, name, label, out_dir, out_prefix):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)

    print(f"{label} MAE: {mae:.4f}")
    print(f"{label} RMSE: {rmse:.4f}")

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(test_df["UTCTimestampCollected"], y_true, label="Actual", color="black")
    plt.plot(test_df["UTCTimestampCollected"], y_pred, label="Predicted", color="green")
    plt.legend()
    plt.title(f"{name} {label} Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature Difference (VT90-VT20)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_{out_prefix}_forecast.png"))
    plt.close()

    return mae, rmse

# --- Main Script ---

if __name__ == "__main__":
    MODEL_NAME = "CatBoost"
    OUTPUT_DIR = "CatBoost" # Specific directory for CatBoost outputs
    OUT_PREFIX = "cb"

    os.makedirs(OUTPUT_DIR, exist_ok=True) # Makes the directory if it doesn't exist

    try:
        # Determine categorical features. Adjust this based on your actual data types
        # and if any of the numeric-looking features are actually categories.
        # For 'hour', 'dayofweek', 'month', 'day', 'year', 'minute', these are often treated as
        # numerical by default, but CatBoost can benefit from treating them as categorical
        # if they truly represent discrete categories with no inherent ordinal relationship beyond their number.
        # If your data has actual string/object categorical columns, you'd add them here.
        # Assuming your 'features' list is all numerical for now.
        # If 'feature2' in your dummy data (from prior example) was a real categorical, you'd add it here.
        # For simplicity, assuming no explicit categorical features from your 'features' list
        # that CatBoost *must* know about. If you want it to treat 'dayofweek' as categorical,
        # you'd need to convert it to a string/category dtype or pass its index.
        # For example: categorical_features = ['dayofweek', 'month'] if they were strings in X_train.
        # Or, if they are numerical but should be treated as categories:
        
        # Example to get categorical feature indices (assuming your X_train is a pandas DataFrame)
        # This will convert time-based integers to object type, and then select them as categorical
        
        def get_catboost_cat_features(X_df):
            cat_features_names = []
            # These are typically treated as categorical by tree models if their unique values are few,
            # but you need to explicitly tell CatBoost
            potential_categorical_time_features = ["hour", "dayofweek", "month", "day", "year", "minute"]
            
            # Check which of the 'features' list columns are present and if they should be treated as categorical
            for col in X_df.columns:
                if col in potential_categorical_time_features:
                    # You might also check X_df[col].nunique() < some_threshold or X_df[col].dtype == 'object'
                    cat_features_names.append(col)
            
            # Get indices
            return [X_df.columns.get_loc(col) for col in cat_features_names if col in X_df.columns]

        # GRDR
        print(f"--- {MODEL_NAME} for GRDR ---")
        print("Loading GRDR_processed...\n")
        df_grdr = load_data("../Processed Data/GRDR_processed.csv")

        X_train_grdr, X_test_grdr, y_train_grdr, y_test_grdr, test_df_grdr = \
            train_test_split(df_grdr, "UTCTimestampCollected", "2020-04-10", "2021-04-10")
        
        # Get categorical feature indices for CatBoost specifically
        categorical_features_indices_grdr = get_catboost_cat_features(X_train_grdr)
        if categorical_features_indices_grdr:
            print(f"CatBoost will treat these columns as categorical (indices): {categorical_features_indices_grdr}")


        print("Fitting CatBoost model...")
        # CatBoost Regressor setup
        model_cb_grdr = CatBoostRegressor(
            iterations=1000,             # Number of boosting rounds
            learning_rate=0.05,          # Step size shrinkage
            depth=6,                     # Depth of the tree
            loss_function='RMSE',        # Regression objective
            eval_metric='RMSE',          # Metric for early stopping
            random_seed=42,
            verbose=False,               # Set to True or an integer (e.g., 100) for verbose output
            early_stopping_rounds=50,    # Stop if validation metric doesn't improve for 50 rounds
            cat_features=categorical_features_indices_grdr # Pass categorical features here
        )
        
        # CatBoost uses Pool for efficient handling, especially with early stopping
        train_pool_grdr = Pool(X_train_grdr, y_train_grdr, cat_features=categorical_features_indices_grdr)
        test_pool_grdr = Pool(X_test_grdr, y_test_grdr, cat_features=categorical_features_indices_grdr)

        model_cb_grdr.fit(train_pool_grdr, eval_set=test_pool_grdr)
        print("Predicting/forecasting...")
        predictions_cb_grdr = model_cb_grdr.predict(X_test_grdr)

        mae_grdr, rmse_grdr = evaluate_and_plot(y_test_grdr, predictions_cb_grdr, test_df_grdr,
                                                 "GRDR", MODEL_NAME, OUTPUT_DIR, OUT_PREFIX)

        # Save model and metrics
        joblib.dump(model_cb_grdr, os.path.join(OUTPUT_DIR, f"GRDR_{OUT_PREFIX}_model.pkl"))
        with open(os.path.join(OUTPUT_DIR, f"GRDR_{OUT_PREFIX}_metrics.txt"), "w") as f:
            f.write(f"MAE: {mae_grdr:.4f}\n")
            f.write(f"RMSE: {rmse_grdr:.4f}\n")
        print(f"{MODEL_NAME} GRDR model complete. Results saved.")

        # WOOD
        print(f"\n--- {MODEL_NAME} for WOOD ---")
        print("Loading WOOD_processed...\n")
        df_wood = load_data("../Processed Data/WOOD_processed.csv")

        X_train_wood, X_test_wood, y_train_wood, y_test_wood, test_df_wood = \
            train_test_split(df_wood, "UTCTimestampCollected", "2024-08-10", "2025-05-11")
        
        categorical_features_indices_wood = get_catboost_cat_features(X_train_wood)
        if categorical_features_indices_wood:
            print(f"CatBoost will treat these columns as categorical (indices): {categorical_features_indices_wood}")

        print("Fitting CatBoost model...")
        model_cb_wood = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
            cat_features=categorical_features_indices_wood
        )
        train_pool_wood = Pool(X_train_wood, y_train_wood, cat_features=categorical_features_indices_wood)
        test_pool_wood = Pool(X_test_wood, y_test_wood, cat_features=categorical_features_indices_wood)

        model_cb_wood.fit(train_pool_wood, eval_set=test_pool_wood)
        print("Predicting/forecasting...")
        predictions_cb_wood = model_cb_wood.predict(X_test_wood)

        mae_wood, rmse_wood = evaluate_and_plot(y_test_wood, predictions_cb_wood, test_df_wood,
                                                 "WOOD", MODEL_NAME, OUTPUT_DIR, OUT_PREFIX)

        # Save model and metrics
        joblib.dump(model_cb_wood, os.path.join(OUTPUT_DIR, f"WOOD_{OUT_PREFIX}_model.pkl"))
        with open(os.path.join(OUTPUT_DIR, f"WOOD_{OUT_PREFIX}_metrics.txt"), "w") as f:
            f.write(f"MAE: {mae_wood:.4f}\n")
            f.write(f"RMSE: {rmse_wood:.4f}\n")
        print(f"{MODEL_NAME} WOOD model complete. Results saved.")

    except Exception as e:
        print(f"Error during {MODEL_NAME} processing:", e)