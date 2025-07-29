import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
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

def evaluate_and_plot(y_true, y_pred, test_df, name, label, out_dir, out_prefix): # Added out_dir
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
    # Use os.path.join for robust path creation
    plt.savefig(os.path.join(out_dir, f"{name}_{out_prefix}_forecast.png"))
    plt.close()

    return mae, rmse

# --- Main Script ---

if __name__ == "__main__":
    MODEL_NAME = "Decision Tree"
    OUTPUT_DIR = "Decision Tree" # Specific directory for Decision Tree outputs
    OUT_PREFIX = "dt"

    os.makedirs(OUTPUT_DIR, exist_ok=True) # Makes the directory if it doesn't exist

    try:
        # GRDR
        print(f"--- {MODEL_NAME} for GRDR ---")
        print("Loading GRDR_processed...\n")
        df_grdr = load_data("../Processed Data/GRDR_processed.csv")

        X_train_grdr, X_test_grdr, y_train_grdr, y_test_grdr, test_df_grdr = \
            train_test_split(df_grdr, "UTCTimestampCollected", "2020-04-10", "2021-04-10")

        print("Fitting decision tree model...")
        # Decision Tree Regressor setup
        model_dt_grdr = DecisionTreeRegressor(max_depth=10, random_state=42) # Added max_depth to prevent overfitting
        model_dt_grdr.fit(X_train_grdr, y_train_grdr)
        print("Predicting/forecasting...")
        predictions_dt_grdr = model_dt_grdr.predict(X_test_grdr)

        mae_grdr, rmse_grdr = evaluate_and_plot(y_test_grdr, predictions_dt_grdr, test_df_grdr,
                                                 "GRDR", MODEL_NAME, OUTPUT_DIR, OUT_PREFIX)

        # Save model and metrics
        joblib.dump(model_dt_grdr, os.path.join(OUTPUT_DIR, f"GRDR_{OUT_PREFIX}_model.pkl"))
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

        print("Fitting decision tree model...")
        model_dt_wood = DecisionTreeRegressor(max_depth=10, random_state=42)
        model_dt_wood.fit(X_train_wood, y_train_wood)
        print("Predicting/forecasting...")
        predictions_dt_wood = model_dt_wood.predict(X_test_wood)

        mae_wood, rmse_wood = evaluate_and_plot(y_test_wood, predictions_dt_wood, test_df_wood,
                                                 "WOOD", MODEL_NAME, OUTPUT_DIR, OUT_PREFIX)

        # Save model and metrics
        joblib.dump(model_dt_wood, os.path.join(OUTPUT_DIR, f"WOOD_{OUT_PREFIX}_model.pkl"))
        with open(os.path.join(OUTPUT_DIR, f"WOOD_{OUT_PREFIX}_metrics.txt"), "w") as f:
            f.write(f"MAE: {mae_wood:.4f}\n")
            f.write(f"RMSE: {rmse_wood:.4f}\n")
        print(f"{MODEL_NAME} WOOD model complete. Results saved.")

    except Exception as e:
        print(f"Error during {MODEL_NAME} processing:", e)