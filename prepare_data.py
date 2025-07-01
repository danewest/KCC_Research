import pandas as pd
import numpy as np

def prepare_data(input_path: str, output_path: str, time_col: str, freq: str = '5T'):
    """
    Reads a CSV, converts the time column to a DatetimeIndex, resamples to a
    regular frequency, and linearly interpolates missing values.

    Args:
        input_path (str): The file path to the raw CSV data.
        output_path (str): The file path to save the processed CSV data.
        time_col (str): The name of the time column in the CSV.
        freq (str): The resampling frequency (e.g., '5T' for 5 minutes, '30T' for 30 minutes).
    """
    try:
        print(f"Reading raw data from: {input_path}")
        # Read the CSV, specifying the time column for parsing
        df = pd.read_csv(input_path, parse_dates=[time_col])

        # Set the time column as the DataFrame index
        df.set_index(time_col, inplace=True)

        # Get the full time range for the specified frequency
        full_time_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        print(f"Original data spans from {df.index.min()} to {df.index.max()}.")
        print(f"Resampling to a frequency of {freq}...")

        # Reindex the DataFrame to the full, regular time range.
        # This will introduce NaNs where data is missing.
        df_reindexed = df.reindex(full_time_range)

        # Perform linear interpolation to fill the NaNs
        print("Performing linear interpolation...")
        df_reindexed.interpolate(method='linear', limit_direction='both', inplace=True)

        # After interpolation, you might still have NaNs at the very beginning or end
        # of the time series if the original data doesn't start or end with a value.
        # Let's fill any remaining NaNs with the last/next valid observation.
        # This is a safe fallback.
        df_reindexed.fillna(method='ffill', inplace=True)
        df_reindexed.fillna(method='bfill', inplace=True)

        # Let's check for any remaining NaNs. If there are any, they can't be filled.
        if df_reindexed.isnull().values.any():
            print("\nWarning: Some NaN values remain after interpolation and filling.")
            print("Dropping rows with remaining NaNs...")
            df_reindexed.dropna(inplace=True)
        
        print(f"Saving processed data to: {output_path}")
        # Save the processed DataFrame to a new CSV file
        # The index (Timestamp) will be saved as the first column.
        df_reindexed.to_csv(output_path, index=True)

        print(f"Data preparation for {input_path} is complete.")
        print(f"Final DataFrame shape: {df_reindexed.shape}\n")

    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Define input and output file paths
    # Make sure the 'Processed Data' directory exists
    input_grdr_path = "Finalized Data/GRDR_final.csv"
    output_grdr_path = "Processed Data/GRDR_interpolated.csv"

    input_wood_path = "Finalized Data/WOOD_final.csv"
    output_wood_path = "Processed Data/WOOD_interpolated.csv"

    # Prepare the GRDR dataset
    prepare_data(input_grdr_path, output_grdr_path, time_col='UTCTimestampCollected')

    # Prepare the WOOD dataset
    prepare_data(input_wood_path, output_wood_path, time_col='UTCTimestampCollected')

    print("All data preparation tasks completed.")