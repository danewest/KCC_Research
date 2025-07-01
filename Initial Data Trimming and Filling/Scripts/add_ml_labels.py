import pandas as pd
import numpy as np

# --- Configuration (ensure these match your other scripts) ---
DATETIME_COLUMN = 'UTCTimestampCollected'
NUM_DECIMALS_TO_ROUND = 4 # Adjust this to your desired precision for the labels

# Define the labels you want to create and the base variables for them
LABELS_TO_CREATE = {
    'TAIR-VT20': {'numerator': 'TAIR', 'denominator': 'VT20'}, # Here 'denominator' refers to the subtrahend
    'VT90-VT20': {'numerator': 'VT90', 'denominator': 'VT20'}
}

# --- Helper function to add labels to a single DataFrame ---
def add_labels_to_dataframe(df_input, site_name, data_type_suffix, output_base_filename, datetime_col_name):
    """
    Adds specified label columns (e.g., TAIR-VT20) to a DataFrame and saves it.

    Args:
        df_input (pd.DataFrame): The DataFrame to add labels to (should be gap-filled).
        site_name (str): The name of the site (e.g., 'GRDR').
        data_type_suffix (str): A suffix indicating the type of filled data (e.g., '_standard_filled_rf_ws').
        output_base_filename (str): Base name for the output file (e.g., 'GRDR_data_gap_filled_standard_filled_rf_ws').
        datetime_col_name (str): Name of the datetime index column.
    Returns:
        pd.DataFrame: The DataFrame with added labels, or None if input is invalid.
    """
    print(f"\n--- Adding ML Labels to {site_name} Data ({data_type_suffix.strip('_')}) ---")

    if df_input is None or df_input.empty:
        print(f"Skipping label addition for {site_name} as DataFrame is empty or None.")
        return None

    df_with_labels = df_input.copy()
    
    for label_name, components in LABELS_TO_CREATE.items():
        num_col = components['numerator']
        den_col = components['denominator'] # Renamed for clarity in subtraction

        if num_col in df_with_labels.columns and den_col in df_with_labels.columns:
            # Create the label by subtracting the 'denominator' from the 'numerator'
            df_with_labels[label_name] = df_with_labels[num_col] - df_with_labels[den_col]
            
            # --- ADD ROUNDING FOR THE NEW LABEL HERE ---
            df_with_labels[label_name] = df_with_labels[label_name].round(NUM_DECIMALS_TO_ROUND)
            # --- END ADD ROUNDING ---
            
            print(f"  Added label: '{label_name}' (from '{num_col}' - '{den_col}')")
            # Note: The new label column will contain NaNs where either source column was NaN.
        else:
            print(f"  Warning: Cannot create label '{label_name}'. Missing one or both source columns ('{num_col}', '{den_col}').")
            # Create the column with all NaNs if source columns are missing
            df_with_labels[label_name] = np.nan

    # Define the output path for the new file
    output_path = f"{output_base_filename}_with_labels.csv"
    
    # Save the DataFrame. Ensure the datetime index is saved as a column.
    df_with_labels.to_csv(output_path, index=True, index_label=datetime_col_name)
    print(f"Successfully saved data with labels to: {output_path}")

    return df_with_labels


# --- Main execution block for the label addition script ---
if __name__ == "__main__":
    print("--- Starting ML Label Generation Process ---")

    # --- Configuration for input/output files ---
    # These paths should point to the output of your 'fill_gaps.py' script.
    # Adjust paths if your directory structure differs.
    
    # Paths for 'With Soil' filled data
    GRDR_FILLED_STANDARD_WS_PATH = 'With Soil/GRDR_data_gap_filled_standard_filled_rf_ws.csv'
    GRDR_FILLED_SOIL_WS_PATH = 'With Soil/GRDR_data_gap_filled_soil_filled_rf_ws.csv'
    WOOD_FILLED_STANDARD_WS_PATH = 'With Soil/WOOD_data_gap_filled_standard_filled_rf_ws.csv'
    WOOD_FILLED_SOIL_WS_PATH = 'With Soil/WOOD_data_gap_filled_soil_filled_rf_ws.csv'

    # Paths for 'No Soil' filled data
    GRDR_FILLED_NO_SOIL_NS_PATH = 'No Soil/GRDR_data_gap_filled_no_soil_filled_rf_ns.csv'
    WOOD_FILLED_NO_SOIL_NS_PATH = 'No Soil/WOOD_data_gap_filled_no_soil_filled_rf_ns.csv'

    # --- Process GRDR (With Soil - Standard) ---
    try:
        file_path = GRDR_FILLED_STANDARD_WS_PATH # Define the file path for this specific load
        data_type_suffix = '_standard_filled_rf_ws' # Define the suffix for this specific type
        site_name_val = 'GRDR' # Define the site name for this load

        # --- ROBUST CSV LOADING ---
        # Load the CSV without initially parsing dates or setting index_col
        df = pd.read_csv(file_path, header=0)
        
        # Explicitly convert the timestamp column to datetime, coercing errors to NaT
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        
        # Drop any rows where the timestamp could not be parsed (resulted in NaT)
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from {file_path} due to unparseable timestamps.")
        
        # Now set the cleaned datetime column as the DataFrame's index
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        
        print(f"DEBUG_LOAD: Successfully loaded {file_path}. Index name: {df.index.name}, Index type: {type(df.index)}")
        # --- END ROBUST CSV LOADING ---

        add_labels_to_dataframe(df, site_name_val, data_type_suffix,
                                 file_path.replace('.csv', ''), DATETIME_COLUMN)
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found. Skipping. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {site_name_val} ({data_type_suffix.strip('_')}): {e}")

    print("\n" + "="*50 + "\n")

    # --- Process GRDR (With Soil - Soil) ---
    try:
        file_path = GRDR_FILLED_SOIL_WS_PATH
        data_type_suffix = '_soil_filled_rf_ws'
        site_name_val = 'GRDR'

        # --- ROBUST CSV LOADING ---
        df = pd.read_csv(file_path, header=0)
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from {file_path} due to unparseable timestamps.")
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        print(f"DEBUG_LOAD: Successfully loaded {file_path}. Index name: {df.index.name}, Index type: {type(df.index)}")
        # --- END ROBUST CSV LOADING ---

        add_labels_to_dataframe(df, site_name_val, data_type_suffix,
                                 file_path.replace('.csv', ''), DATETIME_COLUMN)
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found. Skipping. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {site_name_val} ({data_type_suffix.strip('_')}): {e}")
    
    print("\n" + "="*50 + "\n")

    # --- Process WOOD (With Soil - Standard) ---
    try:
        file_path = WOOD_FILLED_STANDARD_WS_PATH
        data_type_suffix = '_soil_filled_rf_ws'
        site_name_val = 'WOOD'

        # --- ROBUST CSV LOADING ---
        df = pd.read_csv(file_path, header=0)
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from {file_path} due to unparseable timestamps.")
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        print(f"DEBUG_LOAD: Successfully loaded {file_path}. Index name: {df.index.name}, Index type: {type(df.index)}")
        # --- END ROBUST CSV LOADING ---

        add_labels_to_dataframe(df, site_name_val, data_type_suffix,
                                 file_path.replace('.csv', ''), DATETIME_COLUMN)
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found. Skipping. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {site_name_val} ({data_type_suffix.strip('_')}): {e}")
    
    print("\n" + "="*50 + "\n")

    # --- Process WOOD (With Soil - Soil) ---
    try:
        file_path = WOOD_FILLED_SOIL_WS_PATH
        data_type_suffix = '_soil_filled_rf_ws'
        site_name_val = 'WOOD'

        # --- ROBUST CSV LOADING ---
        df = pd.read_csv(file_path, header=0)
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from {file_path} due to unparseable timestamps.")
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        print(f"DEBUG_LOAD: Successfully loaded {file_path}. Index name: {df.index.name}, Index type: {type(df.index)}")
        # --- END ROBUST CSV LOADING ---

        add_labels_to_dataframe(df, site_name_val, data_type_suffix,
                                 file_path.replace('.csv', ''), DATETIME_COLUMN)
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found. Skipping. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {site_name_val} ({data_type_suffix.strip('_')}): {e}")
    
    print("\n" + "="*50 + "\n")

    # --- Process GRDR (No Soil) ---
    try:
        file_path = GRDR_FILLED_NO_SOIL_NS_PATH
        data_type_suffix = '_soil_filled_rf_ws'
        site_name_val = 'GRDR'

        # --- ROBUST CSV LOADING ---
        df = pd.read_csv(file_path, header=0)
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from {file_path} due to unparseable timestamps.")
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        print(f"DEBUG_LOAD: Successfully loaded {file_path}. Index name: {df.index.name}, Index type: {type(df.index)}")
        # --- END ROBUST CSV LOADING ---

        add_labels_to_dataframe(df, site_name_val, data_type_suffix,
                                 file_path.replace('.csv', ''), DATETIME_COLUMN)
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found. Skipping. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {site_name_val} ({data_type_suffix.strip('_')}): {e}")
    
    print("\n" + "="*50 + "\n")

    # --- Process WOOD (No Soil) ---
    try:
        file_path = WOOD_FILLED_NO_SOIL_NS_PATH
        data_type_suffix = '_soil_filled_rf_ws'
        site_name_val = 'WOOD'

        # --- ROBUST CSV LOADING ---
        df = pd.read_csv(file_path, header=0)
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from {file_path} due to unparseable timestamps.")
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)
        print(f"DEBUG_LOAD: Successfully loaded {file_path}. Index name: {df.index.name}, Index type: {type(df.index)}")
        # --- END ROBUST CSV LOADING ---

        add_labels_to_dataframe(df, site_name_val, data_type_suffix,
                                 file_path.replace('.csv', ''), DATETIME_COLUMN)
    except FileNotFoundError as e:
        print(f"Error: {file_path} not found. Skipping. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {site_name_val} ({data_type_suffix.strip('_')}): {e}")
    
    print("\n" + "="*50 + "\n")