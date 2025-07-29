import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
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
    MODEL_NAME = "LightGBM"
    OUTPUT_DIR = "LightGBM" # Specific directory for LightGBM outputs
    OUT_PREFIX = "lgbm"

    os.makedirs(OUTPUT_DIR, exist_ok=True) # Makes the directory if it doesn't exist

    try:
        # GRDR
        print(f"--- {MODEL_NAME} for GRDR ---")
        print("Loading GRDR_processed...\n")
        df_grdr = load_data("../Processed Data/GRDR_processed.csv")

        X_train_grdr, X_test_grdr, y_train_grdr, y_test_grdr, test_df_grdr = \
            train_test_split(df_grdr, "UTCTimestampCollected", "2020-04-10", "2021-04-10")

        print("Fitting LightGBM model...")
        # LightGBM Regressor setup
        model_lgbm_grdr = lgb.LGBMRegressor(
            objective='regression',      # For regression tasks
            n_estimators=1000,           # Number of boosting rounds
            learning_rate=0.05,          # Step size shrinkage
            num_leaves=31,               # Max number of leaves in one tree (default is 31)
            max_depth=-1,                # No limit on tree depth
            random_state=42,
            n_jobs=-1                    # Use all available CPU cores
        )
        # LightGBM needs eval_set and callbacks for early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] # verbose=True to see stopping messages

        model_lgbm_grdr.fit(X_train_grdr, y_train_grdr,
                            eval_set=[(X_test_grdr, y_test_grdr)],
                            eval_metric='rmse', # Evaluation metric for early stopping
                            callbacks=callbacks)
        print("Predicting/forecasting...")
        predictions_lgbm_grdr = model_lgbm_grdr.predict(X_test_grdr)

        mae_grdr, rmse_grdr = evaluate_and_plot(y_test_grdr, predictions_lgbm_grdr, test_df_grdr,
                                                 "GRDR", MODEL_NAME, OUTPUT_DIR, OUT_PREFIX)

        # Save model and metrics
        joblib.dump(model_lgbm_grdr, os.path.join(OUTPUT_DIR, f"GRDR_{OUT_PREFIX}_model.pkl"))
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

        print("Fitting LightGBM model...")
        model_lgbm_wood = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=42,
            n_jobs=-1
        )
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        model_lgbm_wood.fit(X_train_wood, y_train_wood,
                            eval_set=[(X_test_wood, y_test_wood)],
                            eval_metric='rmse',
                            callbacks=callbacks)
        print("Predicting/forecasting...")
        predictions_lgbm_wood = model_lgbm_wood.predict(X_test_wood)

        mae_wood, rmse_wood = evaluate_and_plot(y_test_wood, predictions_lgbm_wood, test_df_wood,
                                                 "WOOD", MODEL_NAME, OUTPUT_DIR, OUT_PREFIX)

        # Save model and metrics
        joblib.dump(model_lgbm_wood, os.path.join(OUTPUT_DIR, f"WOOD_{OUT_PREFIX}_model.pkl"))
        with open(os.path.join(OUTPUT_DIR, f"WOOD_{OUT_PREFIX}_metrics.txt"), "w") as f:
            f.write(f"MAE: {mae_wood:.4f}\n")
            f.write(f"RMSE: {rmse_wood:.4f}\n")
        print(f"{MODEL_NAME} WOOD model complete. Results saved.")

    except Exception as e:
        print(f"Error during {MODEL_NAME} processing:", e)