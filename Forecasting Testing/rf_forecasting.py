import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def load_data(file_path: str):
    df = pd.read_csv(file_path, parse_dates=["UTCTimestampCollected"])
    return df

def train_test_split(df, date_col, split_date, test_end):
    train = df[df[date_col] < split_date]
    test = df[(df[date_col] >= split_date) & (df[date_col] <= test_end)]

    features = ["hour", "dayofweek", "month", "day", "year", "minute",
                "inversion_diff_lag1", "rolling_mean_3"]
    target = "VT90_VT20_diff"

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    return X_train, X_test, y_train, y_test, test

def evaluate_and_plot(y_true, y_pred, test_df, name ,label="Random Forest", out_prefix="rf"):
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
    plt.savefig(f"Random Forest/{name}_{out_prefix}_forecast.png")
    plt.close()

    return mae, rmse

if __name__ == "__main__":
    output_dir = "Random Forest"
    os.makedirs(output_dir, exist_ok=True) # Makes the directory if it doesn't exist
    try:
        
        # GRDR
        print("Loading GRDR_processed...\n")
        df = load_data("../Processed Data/GRDR_processed.csv")

        X_train, X_test, y_train, y_test, test_df = train_test_split(df, "UTCTimestampCollected", "2020-04-10", "2021-04-10")

        print("Fitting random forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Predicting/forecasting...")
        predictions = model.predict(X_test)

        mae, rmse = evaluate_and_plot(y_test, predictions, test_df, "GRDR")

        # Save model and metrics
        joblib.dump(model, "Random Forest/GRDR_rf_model.pkl")
        with open("Random Forest/GRDR_rf_metrics.txt", "w") as f:
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")

        print("Random Forest GRDR model complete. Results saved.")

        # WOOD
        print("Loading WOOD_processed...\n")
        df = load_data("../Processed Data/WOOD_processed.csv")

        X_train, X_test, y_train, y_test, test_df = train_test_split(df, "UTCTimestampCollected", "2024-08-10", "2025-05-11")

        print("Fitting random forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Predicting/forecasting...")
        predictions = model.predict(X_test)

        mae, rmse = evaluate_and_plot(y_test, predictions, test_df, "WOOD")

        # Save model and metrics
        joblib.dump(model, "Random Forest/WOOD_rf_model.pkl")
        with open("Random Forest/WOOD_rf_metrics.txt", "w") as f:
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")

        print("Random Forest WOOD model complete. Results saved.")

    except Exception as e:
        print("Error during Random Forest processing:", e)
