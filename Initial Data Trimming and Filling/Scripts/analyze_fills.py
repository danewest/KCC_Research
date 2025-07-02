import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for path joining

# --- Configuration ---
DATETIME_COLUMN = 'UTCTimestampCollected'

# Define the base directory where your combined filled CSVs are located
# This should be the same directory where your 'combine_csvs.py' outputs its files.
COMBINED_FILLED_CSVS_DIR = '.' # As per your last update

# Define the base directory where your original trimmed CSVs are located
ORIGINAL_TRIMMED_CSVS_DIR_WS = 'With Soil/Filtered CSVs' # For 'With Soil' trimmed data


# --- Paths for GRDR (With Soil) ---
GRDR_ORIGINAL_TRIMMED_WS_PATH = os.path.join(ORIGINAL_TRIMMED_CSVS_DIR_WS, 'GRDR_trimmed_data.csv')
GRDR_COMBINED_FILLED_WS_PATH = os.path.join(COMBINED_FILLED_CSVS_DIR, 'GRDR_data_gap_filled_combined_ws.csv')

# --- Paths for WOOD (With Soil) ---
WOOD_ORIGINAL_TRIMMED_WS_PATH = os.path.join(ORIGINAL_TRIMMED_CSVS_DIR_WS, 'WOOD_trimmed_data.csv')
WOOD_COMBINED_FILLED_WS_PATH = os.path.join(COMBINED_FILLED_CSVS_DIR, 'WOOD_data_gap_filled_combined_ws.csv')



def load_and_prepare_for_analysis(file_path, datetime_col):
    """Loads a CSV and sets DatetimeIndex, handling potential issues."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}. Skipping load.")
        return None
    try:
        df = pd.read_csv(file_path, header=0)
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', format='mixed')
        df.dropna(subset=[datetime_col], inplace=True)
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)
        print(f"Loaded {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_fills(original_df, filled_df, variable, title_suffix="", start_time=None, end_time=None):
    """Plots original and filled data for a given variable."""
    if variable not in original_df.columns or variable not in filled_df.columns:
        print(f"Warning: Variable '{variable}' not found in one or both DataFrames. Skipping plot.")
        return
        
    plt.figure(figsize=(15, 6))
    
    # Ensure both dataframes are aligned on the same index for comparison
    # Use the filled_df's index as it should be the more complete one (5-min resolution)
    common_index = filled_df.index.intersection(original_df.index)
    
    plot_df_orig = original_df.loc[common_index, variable]
    plot_df_filled = filled_df.loc[common_index, variable] # Ensure it's using the correct data from combined

    if start_time and end_time:
        plot_df_orig = plot_df_orig.loc[start_time:end_time]
        plot_df_filled = plot_df_filled.loc[start_time:end_time]
        plt.title(f'{variable} - Original vs. Filled Data ({title_suffix}) - Zoomed In')
    else:
        plt.title(f'{variable} - Original vs. Filled Data ({title_suffix})')

    plt.plot(plot_df_orig, label='Original Data', color='blue', alpha=0.7)
    plt.plot(plot_df_filled, label='Filled Data', color='red', linestyle='--', alpha=0.8)
    
    # Highlight filled regions (where original was NaN and filled is not NaN)
    # This comparison needs to be done carefully because original might have NaNs where filled has values.
    # We should look at original NaNs within the common index range.
    original_nans_in_range = original_df.loc[common_index, variable].isna()
    filled_values_in_range = filled_df.loc[common_index, variable].notna()
    
    # Points that were originally NaN and are now filled
    newly_filled_mask = original_nans_in_range & filled_values_in_range
    
    if newly_filled_mask.any():
        plt.scatter(filled_df.index[newly_filled_mask], filled_df[variable][newly_filled_mask],
                    color='green', marker='o', s=15, label='Filled Points', zorder=5) # zorder to ensure visibility

    plt.xlabel('Time')
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_statistics(original_df, filled_df, variable, title_suffix=""):
    """Prints and plots statistical comparison for a given variable."""
    if variable not in original_df.columns or variable not in filled_df.columns:
        print(f"Warning: Variable '{variable}' not found for statistical comparison. Skipping.")
        return

    print(f"\n--- Statistical Comparison for {variable} ({title_suffix}) ---")
    
    # Ensure both dataframes are aligned for comparison
    common_index = filled_df.index.intersection(original_df.index)
    original_var_series = original_df.loc[common_index, variable]
    filled_var_series = filled_df.loc[common_index, variable]

    print(f"\nOriginal (trimmed) {variable} Statistics:")
    print(original_var_series.describe())
    print(f"NaNs in original {variable}: {original_var_series.isna().sum()}")

    print(f"\nFilled {variable} Statistics:")
    print(filled_var_series.describe())
    print(f"NaNs in filled {variable}: {filled_var_series.isna().sum()}")

    # Compare non-NaN counts directly
    original_non_nan_count = original_var_series.notna().sum()
    filled_non_nan_count = filled_var_series.notna().sum()
    points_filled = filled_non_nan_count - original_non_nan_count
    
    print(f"\nPoints originally valid for {variable}: {original_non_nan_count}")
    print(f"Points valid after filling for {variable}: {filled_non_nan_count}")
    if points_filled > 0:
        print(f"Net points filled for {variable}: {points_filled}")
    else:
        print(f"No net points filled for {variable} or data lost.")


    # Plot Histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    original_var_series.hist(bins=50, alpha=0.7, label='Original')
    plt.title(f'Original {variable} Distribution')
    plt.legend()
    plt.xlabel(variable)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    filled_var_series.hist(bins=50, alpha=0.7, label='Filled', color='red')
    plt.title(f'Filled {variable} Distribution')
    plt.legend()
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--- Starting Fill Analysis and Comparison ---")

    # --- Scenario 1: GRDR (With Soil) ---
    print("\n##### Analyzing GRDR (With Soil) #####")
    original_grdr_ws = load_and_prepare_for_analysis(GRDR_ORIGINAL_TRIMMED_WS_PATH, DATETIME_COLUMN)
    filled_grdr_ws_combined = load_and_prepare_for_analysis(GRDR_COMBINED_FILLED_WS_PATH, DATETIME_COLUMN)
    
    if original_grdr_ws is not None and filled_grdr_ws_combined is not None:
        # Variables to plot/compare for With Soil scenario
        # Include standard and soil variables
        variables_to_analyze_ws = ['TAIR', 'PRES', 'VT90', 'VT20', 'SM02', 'SM04', 'ST02', 'ST04', 'WS', 'WDIR', 'RH', 'BATT'] # Add more columns here if you have them in combined!

        for var in variables_to_analyze_ws:
            plot_fills(original_grdr_ws, filled_grdr_ws_combined, var, 'GRDR (With Soil - Combined)')
            compare_statistics(original_grdr_ws, filled_grdr_ws_combined, var, 'GRDR (With Soil - Combined)')
            
            # Example: Zoom in on a specific gap for PRES (replace with actual gap times from your reports)
            # This is illustrative; you'd find actual gaps from your gap reports
            # plot_fills(original_grdr_ws, filled_grdr_ws_combined, 'PRES', 'GRDR (With Soil - Combined) - Zoomed',
            #            start_time='2025-04-22 16:00:00', end_time='2025-04-22 19:00:00')

    print("\n" + "="*80 + "\n") # Separator

    # --- Scenario 2: WOOD (With Soil) ---
    print("\n##### Analyzing WOOD (With Soil) #####")
    original_wood_ws = load_and_prepare_for_analysis(WOOD_ORIGINAL_TRIMMED_WS_PATH, DATETIME_COLUMN)
    filled_wood_ws_combined = load_and_prepare_for_analysis(WOOD_COMBINED_FILLED_WS_PATH, DATETIME_COLUMN)

    if original_wood_ws is not None and filled_wood_ws_combined is not None:
        variables_to_analyze_ws = ['TAIR', 'PRES', 'VT90', 'VT20', 'SM02', 'SM04', 'ST02', 'ST04', 'WS', 'WDIR', 'RH', 'BATT'] # Add more columns here if you have them in combined!
        for var in variables_to_analyze_ws:
            plot_fills(original_wood_ws, filled_wood_ws_combined, var, 'WOOD (With Soil - Combined)')
            compare_statistics(original_wood_ws, filled_wood_ws_combined, var, 'WOOD (With Soil - Combined)')


    print("\n--- All Fill Analysis Complete. Check generated plots and console output. ---")