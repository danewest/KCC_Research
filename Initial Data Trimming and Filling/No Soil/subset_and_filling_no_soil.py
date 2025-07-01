import pandas as pd
import numpy as np

# --- Configuration based on your provided data format ---
DATETIME_COLUMN = 'UTCTimestampCollected'
# List of critical variables to check for non-empty start and for labels.
# SM02 has been removed from this list as requested.
CRITICAL_VARS_NO_SOIL = ['TAIR', 'PRES', 'VT90', 'VT20']

# --- Helper function for gap analysis ---
def analyze_data_gaps(df, analysis_vars, expected_freq_minutes, site_name, analysis_type_suffix=""):
    """
    Analyzes gaps in specified columns of a DataFrame based on an expected frequency.

    Args:
        df (pd.DataFrame): The DataFrame to analyze (should be the trimmed DataFrame).
        analysis_vars (list): List of column names to check for gaps.
        expected_freq_minutes (int): The expected data frequency in minutes (e.g., 5, 30).
        site_name (str): The name of the site (e.g., 'GRDR', 'WOOD').
        analysis_type_suffix (str): An additional suffix for output filenames (e.g., '_soil', '_standard').
    """
    print(f"\n--- Analyzing Gaps for {site_name} ({analysis_type_suffix.strip('_').replace('_', ' ')} data at {expected_freq_minutes}-min intervals) ---")

    if df is None or df.empty:
        print(f"Skipping gap analysis for {site_name} ({analysis_type_suffix}) as DataFrame is empty or None.")
        return

    # --- DEBUGGING PRINTS ---
    print(f"DEBUG: Type of df.index: {type(df.index)}")
    print(f"DEBUG: Dtype of df.index: {df.index.dtype}")
    
    # Check if the index is a DatetimeIndex, if not, attempt conversion
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"ERROR: df.index is NOT a DatetimeIndex. It is: {type(df.index)}")
        print(f"First 5 elements of index: {df.index[:5].tolist()}")
        print(f"Min index value before conversion attempt: {df.index.min()}")
        print(f"Max index value before conversion attempt: {df.index.max()}")
        try:
            # Attempt conversion, coercing errors to NaT
            # Use 'mixed' format to handle potential variations, and errors='coerce' to turn unparseable strings into NaT
            original_index_name = df.index.name # Store index name if it exists
            df.index = pd.to_datetime(df.index, errors='coerce', format='mixed')
            df.index.name = original_index_name # Restore index name

            print(f"DEBUG: Successfully attempted conversion of index to DatetimeIndex with error coercion.")
            
            # After conversion, check for and remove any NaT values introduced in the index
            initial_len = len(df)
            df.dropna(axis=0, subset=[df.index.name] if df.index.name else None, inplace=True) # Drops rows where index is NaT
            if len(df) < initial_len:
                print(f"DEBUG: Removed {initial_len - len(df)} rows due to unparseable datetimes in index (converted to NaT).")
            
            # Re-check if index is still DatetimeIndex after NaT removal
            if not isinstance(df.index, pd.DatetimeIndex):
                 print(f"FATAL ERROR: Index is still not DatetimeIndex after conversion and NaN removal. Current type: {type(df.index)}. Exiting.")
                 return # Cannot proceed if index is fundamentally broken

        except Exception as e:
            print(f"FATAL ERROR: Failed to convert index to DatetimeIndex even with error coercion: {e}")
            print("Cannot proceed with date range generation. Please ensure 'UTCTimestampCollected' is correctly parsed as datetime upon loading.")
            return # Exit the function if conversion still fails

    # Now that we've ensured (or tried to ensure) it's a DatetimeIndex and cleaned NaT from index
    # It's important to re-evaluate min/max AFTER potential conversion and NaN removal
    if df.empty: # Check if dropping NaT made it empty
        print(f"DEBUG: DataFrame became empty after cleaning index. Skipping gap analysis.")
        return

    print(f"DEBUG: Value of df.index.min() (after potential conversion/cleaning): {df.index.min()}")
    print(f"DEBUG: Type of df.index.min() (after potential conversion/cleaning): {type(df.index.min())}")
    print(f"DEBUG: Value of df.index.max() (after potential conversion/cleaning): {df.index.max()}")
    print(f"DEBUG: Type of df.index.max() (after potential conversion/cleaning): {type(df.index.max())}")

    # Create a complete datetime index based on the expected frequency
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=f'{expected_freq_minutes}min')
    
    # Reindex the DataFrame to fill in explicit NaNs for missing timestamps
    df_reindexed = df.reindex(full_time_range)

    all_gaps = []

    for col in analysis_vars:
        if col not in df_reindexed.columns:
            print(f"Warning: Column '{col}' not found in reindexed data for {site_name}. Skipping gap analysis for this column.")
            continue

        # Identify where values are NaN
        is_nan = df_reindexed[col].isna()

        # Find starts and ends of NaN sequences
        # Shifted boolean array to find transitions from non-NaN to NaN (gap start)
        # and from NaN to non-NaN (gap end)
        gap_starts = df_reindexed.index[is_nan & ~is_nan.shift(1, fill_value=False)]
        gap_ends = df_reindexed.index[is_nan & ~is_nan.shift(-1, fill_value=False)]

        # Ensure we have corresponding starts and ends
        # Handle cases where gap starts at beginning or ends at end of data
        if len(gap_starts) > len(gap_ends):
            gap_ends = gap_ends.tolist() + [df_reindexed.index.max()]
        elif len(gap_ends) > len(gap_starts):
            gap_starts = [df_reindexed.index.min()] + gap_starts.tolist()

        for start_dt, end_dt in zip(gap_starts, gap_ends):
            duration = end_dt - start_dt
        
            # --- IMPORTANT CHANGE: New gap categorization ---
            RF_MAX_GAP_HOURS = 3 * 30 * 24 # Example: 3 months * 30 days/month * 24 hours/day (adjust as needed)
                                        # or simply 2160 hours for 3 months
        
            if duration <= pd.Timedelta(hours=1):
                gap_type = 'Short Gap (<= 1 hour)'
            elif duration <= pd.Timedelta(hours=RF_MAX_GAP_HOURS): # Gaps that RF might fill
                gap_type = f'Long Gap (> 1 hour, <= {RF_MAX_GAP_HOURS} hours)'
            else: # Gaps that are too long to fill with RF
                gap_type = f'Very Long Gap (> {RF_MAX_GAP_HOURS} hours - UNFILLABLE)'
            # --- END IMPORTANT CHANGE ---
        
        if pd.isna(df_reindexed.loc[start_dt, col]):
            all_gaps.append({
                'Site': site_name,
                'Variable': col,
                'Gap_Start': start_dt,
                'Gap_End': end_dt,
                'Duration': str(duration),
                'Duration_Minutes': duration.total_seconds() / 60,
                'Gap_Type': gap_type
            })
    
    if all_gaps:
        gaps_df = pd.DataFrame(all_gaps)
        gaps_df.sort_values(by=['Gap_Start', 'Variable'], inplace=True)
        output_path = f"{site_name}_gaps_report{analysis_type_suffix}.csv"
        gaps_df.to_csv(output_path, index=False)
        print(f"Gap analysis report saved to: {output_path}")
    else:
        print(f"No notable gaps found for specified variables in {site_name} ({analysis_type_suffix}).")

# --- Function to process a single site ---
def process_site_data_no_soil(file_path, site_name):
    print(f"\n--- Processing {site_name} Data (Excluding Soil Observations) ---")

    # 1. Load Data - APPLYING ROBUST DATETIME HANDLING
    try:
        # Load the CSV without initially parsing dates for the timestamp column
        df = pd.read_csv(file_path, header=0)
        
        # Explicitly convert the timestamp column to datetime, coercing errors to NaT
        # 'format='mixed'' allows pandas to try various formats if they're inconsistent
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], errors='coerce', format='mixed')
        
        # Drop any rows where the timestamp could not be parsed (resulted in NaT)
        initial_rows = len(df)
        df.dropna(subset=[DATETIME_COLUMN], inplace=True)
        if len(df) < initial_rows:
            print(f"DEBUG: Removed {initial_rows - len(df)} rows from original data due to unparseable timestamps in '{DATETIME_COLUMN}'.")

        # Now set the cleaned datetime column as the index
        df.set_index(DATETIME_COLUMN, inplace=True)
        df.sort_index(inplace=True)

        # Ensure critical variables for NO SOIL are numeric
        for col in CRITICAL_VARS_NO_SOIL:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Warning: Critical column '{col}' not found in {site_name} data.")

        # Ensure SM02, SM04, ST02, and ST04 are also numeric if they exist, even if not critical for start date
        # These are kept so they are still present in the trimmed output CSV if needed for other purposes
        if 'SM02' in df.columns:
            df['SM02'] = pd.to_numeric(df['SM02'], errors='coerce')
        if 'SM04' in df.columns:
            df['SM04'] = pd.to_numeric(df['SM04'], errors='coerce')
        if 'ST02' in df.columns:
            df['ST02'] = pd.to_numeric(df['ST02'], errors='coerce')
        if 'ST04' in df.columns:
            df['ST04'] = pd.to_numeric(df['ST04'], errors='coerce')

        print(f"Successfully loaded {site_name} data. Initial shape: {df.shape}")
        print(f"Initial data types for {site_name} (relevant vars):\\n{df[CRITICAL_VARS_NO_SOIL].dtypes}")
        print(f"Initial missing values (NaN) in critical vars (no soil) for {site_name}:\\n{df[CRITICAL_VARS_NO_SOIL].isnull().sum()}")

    except Exception as e:
        print(f"Error loading {site_name} data: {e}")
        return None

    # 2. Find the date where critical variables (excluding soil) all start to have non-empty values
    start_dates = {}
    for var in CRITICAL_VARS_NO_SOIL:
        if var in df.columns:
            first_valid_idx = df[var].first_valid_index()
            if first_valid_idx:
                start_dates[var] = first_valid_idx
            else:
                print(f"Warning: {var} has no valid data in {site_name} for first valid index check.")
        else:
            print(f"Error: Critical variable '{var}' not found in {site_name} data during start date check.")

    df_trimmed = None
    if start_dates and len(start_dates) == len(CRITICAL_VARS_NO_SOIL):
        common_start_date = max(start_dates.values())
        print(f"Common start date for all critical variables (no soil) in {site_name}: {common_start_date}")
        
        df_trimmed = df[df.index >= common_start_date].copy()
        print(f"DataFrame trimmed from {common_start_date}. New shape: {df_trimmed.shape}")
        print(f"**Earliest datetime in df_trimmed for {site_name}: {df_trimmed.index.min()}**") # Added verification print
        print(f"Missing values after trimming in critical vars (no soil) for {site_name}:\\n{df_trimmed[CRITICAL_VARS_NO_SOIL].isnull().sum()}")

        # --- Save the trimmed DataFrame to a new CSV ---
        trimmed_output_path = f"{site_name}_trimmed_data_no_soil.csv" # New filename for clarity
        df_trimmed.to_csv(trimmed_output_path, index=True)
        print(f"Trimmed data (excluding soil criteria) saved to: {trimmed_output_path}")

        # --- Identify and record missing values specifically in the TRIMMED data (no soil) ---
        missing_details = []
        # Get all datetimes where ANY of the critical vars are missing in the TRIMMED data
        # THIS IS PERFORMED ON df_trimmed, which already starts AFTER common_start_date
        missing_datetimes_in_trimmed = df_trimmed.index[df_trimmed[CRITICAL_VARS_NO_SOIL].isnull().any(axis=1)].tolist()

        print(f"**Analyzing missing datetimes for {site_name} from {len(missing_datetimes_in_trimmed)} potential points in TRIMMED data (no soil criteria).**") # Added verification print
        if missing_datetimes_in_trimmed:
            print(f"**First 5 missing datetimes found in TRIMMED data (no soil criteria) for {site_name}: {missing_datetimes_in_trimmed[:5]}**") # Added verification print
            
        for dt in missing_datetimes_in_trimmed:
            row = df_trimmed.loc[dt]
            missing_in_row = []
            for var in CRITICAL_VARS_NO_SOIL:
                if var in row.index and pd.isna(row[var]):
                    missing_in_row.append(var)
            if missing_in_row: # Only add if there were actual missing values
                missing_details.append({
                    DATETIME_COLUMN: dt,
                    'Missing_Variables': ', '.join(missing_in_row)
                })

        df_missing_details = pd.DataFrame(missing_details)
        missing_details_output_path = f"{site_name}_missing_details_trimmed_no_soil.csv" # New filename for clarity
        if not df_missing_details.empty:
            df_missing_details.to_csv(missing_details_output_path, index=False)
            print(f"Detailed missing value information for trimmed data (no soil criteria) saved to: {missing_details_output_path}")
            print(f"**Earliest datetime in {missing_details_output_path} for {site_name}: {df_missing_details[DATETIME_COLUMN].min()}**") # Added verification print
        else:
            print(f"No specific missing values found in critical variables (no soil criteria) for the trimmed data in {site_name}. No '{missing_details_output_path}' file generated.")

    else:
        print(f"Skipping trimming for {site_name} as common start date could not be determined for all critical variables (excluding soil).")

    # 3. Determine dates that provide continuous data for gap filling and forecasting
    print("\n--- Next Steps for Continuous Data Determination ---")
    print(f"Review the trimmed '{site_name}' data (if available and saved to {site_name}_trimmed_data.csv).")
    print("Visually inspect or programmatically identify long continuous segments suitable for gap filling and forecasting.")
    print("Consider what defines a 'continuous' segment for your analysis (e.g., maximum allowed gap size).")
    print("Record these specific date ranges in your notes document.")

    return df_trimmed

# --- Main execution for both sites (for 'no soil' analysis) ---

# --- Helper function to ensure DatetimeIndex robustness for DataFrames returned by process_site_data ---
def _ensure_datetime_index_robust(df_input, site_name_for_debug, datetime_column_name):
    if df_input is None or df_input.empty:
        print(f"DEBUG (Main): Input DataFrame for {site_name_for_debug} is None or empty. Skipping index check.")
        return None
    
    # Create a mutable copy to avoid potential SettingWithCopyWarning issues if inplace operations are done later
    df_temp = df_input.copy() 

    # If the index is not already a DatetimeIndex or is an object dtype (which indicates parsing issues)
    if not isinstance(df_temp.index, pd.DatetimeIndex) or df_temp.index.dtype == object:
        print(f"DEBUG (Main): Index of {site_name_for_debug} is NOT DatetimeIndex upon return from process_site_data_no_soil. Attempting robust conversion.")
        original_index_name = df_temp.index.name
        try:
            # Attempt to convert the index to DatetimeIndex, coercing errors to NaT
            temp_index = pd.to_datetime(df_temp.index, errors='coerce', format='mixed')
            
            # Check for and remove any NaT values introduced in the index
            initial_len = len(df_temp)
            # Filter rows where the index itself is not NaT
            df_temp = df_temp[temp_index.notna()]
            df_temp.index = temp_index[temp_index.notna()] # Assign the cleaned DatetimeIndex to the DataFrame
            df_temp.index.name = original_index_name # Restore index name

            if len(df_temp) < initial_len:
                print(f"DEBUG (Main): Removed {initial_len - len(df_temp)} rows in main block due to unparseable datetimes in index (converted to NaT).")
            
            if not isinstance(df_temp.index, pd.DatetimeIndex):
                print(f"FATAL ERROR (Main): Index conversion failed even in main block after robust NaT filtering. {site_name_for_debug} cannot be processed.")
                return None
            print(f"DEBUG (Main): {site_name_for_debug} index successfully converted to DatetimeIndex in main block.")
            return df_temp
        except Exception as e:
            print(f"FATAL ERROR (Main): Unexpected error during robust index conversion in main block for {site_name_for_debug}: {e}")
            return None
    else:
        print(f"DEBUG (Main): {site_name_for_debug} index is already DatetimeIndex upon return from process_site_data_no_soil.")
        return df_temp

# Process site data first
grdr_df_trimmed_no_soil = process_site_data_no_soil('GRDR.csv', 'GRDR')
wood_df_trimmed_no_soil = process_site_data_no_soil('WOOD.csv', 'WOOD')

# Apply the robust index conversion after processing, before gap analysis
grdr_df_trimmed_no_soil = _ensure_datetime_index_robust(grdr_df_trimmed_no_soil, 'GRDR', DATETIME_COLUMN)
wood_df_trimmed_no_soil = _ensure_datetime_index_robust(wood_df_trimmed_no_soil, 'WOOD', DATETIME_COLUMN)


# --- Define variables for gap analysis (all 5-minute in this script) ---
ALL_CRITICAL_VARS_NO_SOIL_FOR_GAP_ANALYSIS = ['TAIR', 'PRES', 'VT90', 'VT20']

# --- Perform gap analysis for GRDR (using the universal analyze_data_gaps) ---
if grdr_df_trimmed_no_soil is not None:
    analyze_data_gaps(grdr_df_trimmed_no_soil, ALL_CRITICAL_VARS_NO_SOIL_FOR_GAP_ANALYSIS, 5, 'GRDR', '_no_soil_gaps')

# --- Perform gap analysis for WOOD (using the universal analyze_data_gaps) ---
if wood_df_trimmed_no_soil is not None:
    analyze_data_gaps(wood_df_trimmed_no_soil, ALL_CRITICAL_VARS_NO_SOIL_FOR_GAP_ANALYSIS, 5, 'WOOD', '_no_soil_gaps')


# --- Notes Document Content Guidance ---
print("\n--- Content for your Notes Document (for no soil criteria analysis) ---")
# ... (the rest of your notes guidance for gap filling and feature engineering) ...