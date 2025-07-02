import pandas as pd
import numpy as np
import os

# --- Configuration (ensure these match your other scripts) ---
DATETIME_COLUMN = 'UTCTimestampCollected'

# Input/Output Directory Structure (adjust these paths as needed for your setup)
FILLED_CSVS_DIR = 'Gap_Filled_CSVs' # Assuming the fill_gaps.py saves into this directory

# --- Function to combine gap-filled data for a single site (with soil) ---
def combine_gap_filled_soil_data(site_name, filled_csvs_dir):
    """
    Combines the standard-frequency and soil-frequency gap-filled data for a given site,
    retaining ALL variables from the initial 5-minute resolution DataFrame,
    and updating soil columns with their 30-minute filled values.

    Args:
        site_name (str): The name of the site (e.g., 'GRDR', 'WOOD').
        filled_csvs_dir (str): Directory where the gap-filled CSVs are located.

    Returns:
        pd.DataFrame: The combined DataFrame with all columns, or None if files cannot be loaded.
    """
    print(f"\n--- Combining Gap-Filled Data for {site_name} (With Soil - Retaining ALL Original Columns) ---")

    # Define paths for the two input CSVs from fill_gaps.py
    standard_filled_path = os.path.join(filled_csvs_dir, f"{site_name}_data_gap_filled_standard_filled_rf_ws.csv")
    soil_filled_path = os.path.join(filled_csvs_dir, f"{site_name}_data_gap_filled_soil_filled_rf_ws.csv")

    # Define the output path for the combined file
    combined_output_path = os.path.join(filled_csvs_dir, f"{site_name}_data_gap_filled_combined_ws.csv")

    try:
        # Load the standard-frequency filled data (5-minute resolution).
        # This DataFrame should contain ALL columns from the trimmed data,
        # with standard variables filled and soil variables potentially still sparse or NaN.
        df_final = pd.read_csv(standard_filled_path, index_col=DATETIME_COLUMN, parse_dates=True)
        print(f"  Loaded standard-res data from: {standard_filled_path} (Shape: {df_final.shape})")
        print(f"  Initial columns in combined data: {df_final.columns.tolist()}")

        # Load the soil-frequency filled data (30-minute resolution).
        # This DataFrame contains the filled soil variables.
        df_soil_filled = pd.read_csv(soil_filled_path, index_col=DATETIME_COLUMN, parse_dates=True)
        print(f"  Loaded soil-res data from: {soil_filled_path} (Shape: {df_soil_filled.shape})")
        print(f"  Columns in soil-res data: {df_soil_filled.columns.tolist()}")

    except FileNotFoundError as e:
        print(f"  Error: One or both input files not found for {site_name}: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred while loading files for {site_name}: {e}")
        return None

    # Identify the specific soil columns you want to update in df_final
    # This list should ideally match your SOIL_FREQ_VARS from your gap analysis configuration
    soil_cols_to_update = ['SM02', 'SM04', 'ST02', 'ST04'] # Extend this list if you have more soil vars

    # Filter df_soil_filled to only include the soil columns we want to merge.
    # This ensures we don't accidentally update other columns that might have the same name.
    df_soil_for_update = df_soil_filled[[col for col in soil_cols_to_update if col in df_soil_filled.columns]]

    if df_soil_for_update.empty:
        print("  Warning: No valid soil columns found in soil-resolution data for updating.")
    else:
        # Before updating, ensure df_soil_for_update's index is aligned with df_final's frequency.
        # This will introduce NaNs for the 5-min intervals where no 30-min reading exists.
        df_soil_for_update_reindexed = df_soil_for_update.reindex(df_final.index)

        # Use the .update() method to replace values in df_final with non-NaN values from df_soil_for_update_reindexed.
        # This is the most efficient and cleanest way to "overlay" data based on index alignment.
        # The 'overwrite' parameter is implicit for non-NaN values.
        # It's important that df_final already has these soil columns present, even if with original NaNs,
        # which it should if it carries all non-critical columns.
        df_final.update(df_soil_for_update_reindexed)
        print(f"  Successfully updated and merged specified soil variables using .update().")


    # Save the combined DataFrame
    df_final.to_csv(combined_output_path, index=True, index_label=DATETIME_COLUMN)
    print(f"  Combined gap-filled data saved to: {combined_output_path}")
    print(f"  Final combined data shape: {df_final.shape}")
    print(f"  Final columns in combined data: {df_final.columns.tolist()}")
    print(f"  Missing values in combined data after merge:\n{df_final.isnull().sum()}")

    return df_final

# --- Main execution block for the combination script (for testing) ---
if __name__ == "__main__":
    print("--- Starting Combined Gap-Filled Data Process (With Soil - Standalone Test) ---")

    # IMPORTANT: Adjust this path based on your actual file structure
    CURRENT_FILLED_CSVS_DIR = 'With Soil' # Example: assuming filled CSVs are directly in 'With Soil' folder

    # Create the directory if it doesn't exist (for output)
    if not os.path.exists(CURRENT_FILLED_CSVS_DIR):
        print(f"Error: Input directory not found: {CURRENT_FILLED_CSVS_DIR}. Please ensure your fill_gaps.py output to this location.")
    else:
        # Process GRDR
        grdr_combined_df = combine_gap_filled_soil_data('GRDR', CURRENT_FILLED_CSVS_DIR)

        print("\n" + "="*50 + "\n") # Separator

        # Process WOOD
        wood_combined_df = combine_gap_filled_soil_data('WOOD', CURRENT_FILLED_CSVS_DIR)

    print("\n--- Combined Gap-Filled Data Process Completed ---")