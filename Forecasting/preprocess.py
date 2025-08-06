import pandas as pd

def preprocess(df):
    df['VT90_VT20_diff'] = df['VT90'] - df['VT20']
    df["UTCTimestampCollected"] = pd.to_datetime(df["UTCTimestampCollected"])

    df["hour"] = df["UTCTimestampCollected"].dt.hour
    df["dayofweek"] = df["UTCTimestampCollected"].dt.dayofweek
    df["month"] = df["UTCTimestampCollected"].dt.month
    df["day"] = df["UTCTimestampCollected"].dt.day
    df["year"] = df["UTCTimestampCollected"].dt.year
    df["minute"] = df["UTCTimestampCollected"].dt.minute

    df["inversion_diff_lag1"] = df["VT90_VT20_diff"].shift(1)
    df["rolling_mean_3"] = df["VT90_VT20_diff"].rolling(window=3).mean()

    return df.dropna()

if __name__ == "__main__":
    try:
        GRDR_df = pd.read_csv("../Processed Data/GRDR_interpolated.csv")
        WOOD_df = pd.read_csv("../Processed Data/WOOD_interpolated.csv")

        GRDR_df = preprocess(GRDR_df)
        WOOD_df = preprocess(WOOD_df)

        GRDR_df.to_csv("Test Data/GRDR_processed.csv", index=False)
        WOOD_df.to_csv("Test Data/WOOD_processed.csv", index=False)

        print("Preprocessing completed successfully.")

    except Exception as e:
        print("Something went wrong:", e)
