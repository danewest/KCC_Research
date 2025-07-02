import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler # Added by Claude's script, might not be used directly for RF filling here
from sklearn.model_selection import cross_val_score # Added by Claude's script
import warnings
warnings.filterwarnings('ignore') # To suppress warnings as suggested by Claude

# --- Configuration (ensure these match your other scripts) ---
DATETIME_COLUMN = 'UTCTimestampCollected'

def create_time_features(df_index):
    """
    Create time-based features from datetime index for Random Forest.
    
    Args:
        df_index (pd.DatetimeIndex): Datetime index
    
    Returns:
        pd.DataFrame: DataFrame with time-based features
    """
    time_features = pd.DataFrame(index=df_index)
    
    # Basic time features
    time_features['hour'] = df_index.hour
    time_features['day_of_year'] = df_index.dayofyear
    time_features['month'] = df_index.month
    time_features['day_of_week'] = df_index.dayofweek
    
    # Cyclical encoding for better RF performance
    time_features['hour_sin'] = np.sin(2 * np.pi * df_index.hour / 24)
    time_features['hour_cos'] = np.cos(2 * np.pi * df_index.hour / 24)
    time_features['day_sin'] = np.sin(2 * np.pi * df_index.dayofyear / 365.25)
    time_features['day_cos'] = np.cos(2 * np.pi * df_index.dayofyear / 365.25)
    time_features['month_sin'] = np.sin(2 * np.pi * df_index.month / 12)
    time_features['month_cos'] = np.cos(2 * np.pi * df_index.month / 12)
    
    return time_features

def create_lagged_features(df, target_var, lags=[1, 2, 3, 6, 12, 24], window_stats=[6, 12, 24]):
    """
    Create lagged and rolling statistical features for a target variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_var (str): Target variable name
        lags (list): List of lag periods to create
        window_stats (list): List of window sizes for rolling statistics
    
    Returns:
        pd.DataFrame: DataFrame with lagged features
    """
    lagged_features = pd.DataFrame(index=df.index)
    
    if target_var not in df.columns:
        return lagged_features
    
    # Simple lags
    for lag in lags:
        lagged_features[f'{target_var}_lag_{lag}'] = df[target_var].shift(lag)
    
    # Rolling statistics
    for window in window_stats:
        lagged_features[f'{target_var}_mean_{window}'] = df[target_var].rolling(window=window, min_periods=1).mean()
        lagged_features[f'{target_var}_std_{window}'] = df[target_var].rolling(window=window, min_periods=1).std()
        lagged_features[f'{target_var}_min_{window}'] = df[target_var].rolling(window=window, min_periods=1).min()
        lagged_features[f'{target_var}_max_{window}'] = df[target_var].rolling(window=window, min_periods=1).max()
    
    return lagged_features

def get_correlated_variables(df, target_var, correlation_threshold=0.3):
    """
    Find variables that are correlated with the target variable.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_var (str): Target variable name
        correlation_threshold (float): Minimum correlation to include variable
    
    Returns:
        list: List of correlated variable names
    """
    if target_var not in df.columns:
        return []
    
    # Calculate correlations
    correlations = df.corr()[target_var].abs()
    
    # Filter variables above threshold (excluding the target itself)
    correlated_vars = correlations[
        (correlations >= correlation_threshold) & 
        (correlations.index != target_var)
    ].index.tolist()
    
    return correlated_vars

# --- Helper function for Random Forest gap filling (Claude's code with crucial modifications) ---
def fill_gaps_with_random_forest(df, target_var, gap_mask, max_gap_hours=72, 
                                 min_training_points=100, n_estimators=100, 
                                 random_state=42):
    """
    Fill gaps in a single variable using Random Forest.
    
    Args:
        df (pd.DataFrame): Input DataFrame with datetime index
        target_var (str): Variable to fill gaps for
        gap_mask (pd.Series): Boolean mask indicating gap locations (for the gap itself)
        max_gap_hours (int): Maximum gap size to attempt filling (hours)
        min_training_points (int): Minimum points needed for training
        n_estimators (int): Number of trees in Random Forest
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (filled_series, n_filled, success_flag)
    """
    print(f"      Attempting Random Forest filling for {target_var}...")
    
    if target_var not in df.columns:
        print(f"        Error: {target_var} not found in DataFrame")
        return df[target_var] if target_var in df.columns else pd.Series(index=df.index), 0, False
    
    filled_series = df[target_var].copy() # This series will be updated
    
    # Check if gap is too large (redundant with outside logic but good safeguard)
    # The gap_mask is for a single gap. Need to ensure it's not empty before min/max.
    if gap_mask.any():
        gap_duration_sec = (df.index[gap_mask].max() - df.index[gap_mask].min()).total_seconds()
        gap_duration_hours = gap_duration_sec / 3600
    else: # If gap_mask is empty, it's not a real gap to process
        return filled_series, 0, False

    if gap_duration_hours > max_gap_hours:
        print(f"        Gap too large ({gap_duration_hours:.1f} hours > {max_gap_hours} hours), skipping RF filling")
        return filled_series, 0, False
    
    # --- Feature Engineering ---
    print("        Creating time-based features...")
    time_features = create_time_features(df.index) # From entire df index

    print("        Creating lagged features...")
    lagged_features = create_lagged_features(df, target_var) # From entire df

    print("        Finding correlated variables...")
    # Select only numeric columns for correlation calculation
    numeric_df_for_corr = df.select_dtypes(include=np.number).drop(columns=[target_var], errors='ignore')
    # Exclude any known identifier columns that might have numeric-like values
    exclude_ids = [col for col in ['Site', 'site_name', 'SomeOtherID'] if col in numeric_df_for_corr.columns] # 'SomeOtherID' is a placeholder
    numeric_df_for_corr = numeric_df_for_corr.drop(columns=exclude_ids, errors='ignore')
    
    correlated_vars = get_correlated_variables(numeric_df_for_corr, target_var) # Pass cleaned numeric df
    
    # Combine all features into a single DataFrame `all_features_df`
    all_features_df = pd.DataFrame(index=df.index)
    all_features_df = pd.concat([all_features_df, time_features], axis=1) # Time features are always good
    
    if not lagged_features.empty:
        all_features_df = pd.concat([all_features_df, lagged_features], axis=1)

    for var_corr in correlated_vars:
        if var_corr in df.columns and pd.api.types.is_numeric_dtype(df[var_corr]):
            all_features_df[var_corr] = df[var_corr] # Ensure feature is numeric
    
    feature_columns = all_features_df.columns.tolist() # All columns in all_features_df are potential features
    print(f"        Total features created: {len(feature_columns)}")

    # --- Prepare Training Data (non-gap, non-NaN target variable) ---
    # Training mask applies to the target variable
    training_mask = ~gap_mask & filled_series.notna()
    
    # Get raw training features
    X_train_raw = all_features_df.loc[training_mask, feature_columns]
    y_train = filled_series.loc[training_mask]

    # Drop rows from X_train if they have any NaN in features for robust training
    train_complete_mask = X_train_raw.notna().all(axis=1)
    X_train = X_train_raw.loc[train_complete_mask]
    y_train = y_train.loc[train_complete_mask] # Align y_train with X_train

    if len(X_train) < min_training_points:
        print(f"        Insufficient complete training data after NaN removal ({len(X_train)} < {min_training_points}). Skipping RF filling.")
        return filled_series, 0, False

    # --- Prepare Prediction Data (gap locations) ---
    X_pred_raw = all_features_df.loc[gap_mask, feature_columns]

    # CRITICAL FIX: Impute NaNs in prediction features (X_pred_raw)
    # Random Forest cannot handle NaNs in input features (X).
    # Use ffill/bfill to impute NaNs within the feature columns for prediction.
    fill_limit_for_features = int(gap_duration_hours * 60 / df.index.freq.n) if df.index.freq else 12 
    if fill_limit_for_features == 0 and gap_duration_hours > 0:
        fill_limit_for_features = 1 

    X_pred_imputed = X_pred_raw.fillna(method='ffill', limit=fill_limit_for_features).fillna(method='bfill', limit=fill_limit_for_features)
    
    # After imputation, check for remaining NaNs
    pred_complete_mask = X_pred_imputed.notna().all(axis=1)
    
    if not pred_complete_mask.any():
        print("        No complete feature vectors available for prediction after imputation. Skipping RF fill.")
        return filled_series, 0, False

    X_pred_final = X_pred_imputed.loc[pred_complete_mask] # Use only rows with complete features

    # --- IMPORTANT: ENSURE THIS 'try' BLOCK IS PRESENT ---
    try: 
        # --- Train Random Forest model ---
        print(f"        Training Random Forest with {len(X_train)} samples...")
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None, # Allow trees to grow deeper
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1 # Use all available CPU cores
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate model performance (quick cross-validation on subset)
        if len(X_train) > 1000:
            sample_indices = np.random.choice(len(X_train), size=1000, replace=False)
            X_sample = X_train.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]
        else:
            X_sample = X_train
            y_sample = y_train
            
        cv_scores = cross_val_score(rf_model, X_sample, y_sample, cv=3, scoring='r2', n_jobs=-1)
        print(f"        Model R² (CV): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # --- Make Predictions and Fill Gaps ---
        predictions = rf_model.predict(X_pred_final)
        
        # Fill the gaps in the 'filled_series' copy
        original_nans_in_predicted_range_mask = filled_series.loc[X_pred_final.index].isna()
        
        if original_nans_in_predicted_range_mask.any():
            filled_series.loc[X_pred_final.index[original_nans_in_predicted_range_mask]] = predictions[original_nans_in_predicted_range_mask.values]
            n_filled = original_nans_in_predicted_range_mask.sum()
            print(f"        Successfully filled {n_filled} data points using Random Forest")
        else:
            n_filled = 0
            print("        No NaNs found in target variable within the prediction range to fill. Skipping RF fill.")

        return filled_series, n_filled, True
        
    # --- THIS 'except' BLOCK SHOULD BE INDENTED TO MATCH THE 'try' ABOVE ---
    except Exception as e:
        print(f"        Error during Random Forest training/prediction: {e}")
        return filled_series, 0, False

def fill_long_gaps_with_rf(df_filled, gap_report_path, site_name, 
                          max_gap_hours=2160, min_training_points=200,
                          n_estimators=100, random_state=42):
    """
    Fill long gaps using Random Forest based on gap report.
    
    Args:
        df_filled (pd.DataFrame): DataFrame that has already been processed for short gaps
        gap_report_path (str): Path to the gap report CSV
        site_name (str): Site name for logging
        max_gap_hours (int): Maximum gap size to attempt filling (default: 3 months)
        min_training_points (int): Minimum training data points required
        n_estimators (int): Number of trees in Random Forest
        random_state (int): Random seed
    
    Returns:
        tuple: (df_rf_filled, total_filled_points, filled_variables)
    """
    print(f"\n--- Random Forest Gap Filling for {site_name} ---")
    
    # Load gap report
    try:
        gaps_df = pd.read_csv(gap_report_path)
        gaps_df['Gap_Start'] = pd.to_datetime(gaps_df['Gap_Start'])
        gaps_df['Gap_End'] = pd.to_datetime(gaps_df['Gap_End'])
        print(f"Loaded gap report with {len(gaps_df)} gaps")
    except Exception as e:
        print(f"Error loading gap report: {e}")
        return df_filled, 0, set()
    
    # Filter for long gaps only
    long_gaps = gaps_df[gaps_df['Gap_Type'].str.contains('Long Gap', na=False)].copy()
    print(f"Found {len(long_gaps)} long gaps to process with Random Forest")
    
    if len(long_gaps) == 0:
        print("No long gaps found for Random Forest filling")
        return df_filled, 0, set()
    
    df_rf_filled = df_filled.copy()
    total_filled = 0
    filled_vars = set()
    
    # Process each long gap
    for idx, gap_row in long_gaps.iterrows():
        var = gap_row['Variable']
        gap_start = gap_row['Gap_Start']
        gap_end = gap_row['Gap_End']
        duration_hours = gap_row['Duration_Minutes'] / 60
        
        print(f"\nProcessing long gap for {var}: {gap_start} to {gap_end} ({duration_hours:.1f} hours)")
        
        if var not in df_rf_filled.columns:
            print(f"  Warning: Variable {var} not found in DataFrame")
            continue
        
        # Create gap mask
        gap_mask = (df_rf_filled.index >= gap_start) & (df_rf_filled.index <= gap_end)
        
        # Apply Random Forest filling
        filled_series, n_filled, success = fill_gaps_with_random_forest(
            df_rf_filled, var, gap_mask, max_gap_hours, 
            min_training_points, n_estimators, random_state
        )
        
        if success and n_filled > 0:
            df_rf_filled[var] = filled_series
            total_filled += n_filled
            filled_vars.add(var)
            print(f"  ✓ Filled {n_filled} points for {var}")
        else:
            print(f"  ✗ Failed to fill gap for {var}")
    
    print(f"\n--- Random Forest Summary for {site_name} ---")
    print(f"Total points filled: {total_filled}")
    print(f"Variables successfully filled: {len(filled_vars)}")
    print(f"Variables: {', '.join(sorted(filled_vars))}")
    
    return df_rf_filled, total_filled, filled_vars

# Integration function to add RF filling to your existing pipeline
def enhanced_fill_gaps_with_rf(df_original_trimmed, gap_report_path, site_name, 
                              output_suffix="", dataframe_reindex_freq_minutes=None, 
                              datetime_column_name='UTCTimestampCollected',
                              enable_rf_filling=True, max_rf_gap_hours=2160):
    """
    Enhanced version of your fill_gaps_based_on_report function that includes Random Forest.
    
    This function first applies linear interpolation for short gaps, then Random Forest for long gaps.
    """
    print(f"\n--- Enhanced Gap Filling (Linear + RF) for {site_name} ---")
    
    if df_original_trimmed is None or df_original_trimmed.empty:
        print(f"Skipping gap filling for {site_name} as DataFrame is empty or None.")
        return None
    
    # First, apply your existing linear interpolation logic for short gaps
    # (Copy the logic from your fill_gaps_based_on_report function)
    
    # Load gap report
    try:
        gaps_df = pd.read_csv(gap_report_path)
        gaps_df['Gap_Start'] = pd.to_datetime(gaps_df['Gap_Start'])
        gaps_df['Gap_End'] = pd.to_datetime(gaps_df['Gap_End'])
        print(f"Loaded gap report with {len(gaps_df)} gaps from {gap_report_path}.")
    except FileNotFoundError:
        print(f"Error: Gap report file not found at {gap_report_path}.")
        return df_original_trimmed
    except Exception as e:
        print(f"Error loading gap report: {e}")
        return df_original_trimmed
    
    # Make a copy and reindex
    df_filled = df_original_trimmed.copy()
    
    if dataframe_reindex_freq_minutes is None:
        raise ValueError("dataframe_reindex_freq_minutes must be provided.")
    
    full_time_range = pd.date_range(start=df_filled.index.min(), 
                                  end=df_filled.index.max(), 
                                  freq=f'{dataframe_reindex_freq_minutes}min')
    df_filled = df_filled.reindex(full_time_range)
    print(f"DataFrame reindexed to {dataframe_reindex_freq_minutes}-min frequency.")
    
    # Apply linear interpolation for short gaps
    filled_count_short = 0
    filled_vars_short = set()
    
    short_gaps = gaps_df[gaps_df['Gap_Type'].str.contains('Short Gap', na=False)]
    
    for index, row in short_gaps.iterrows():
        var = row['Variable']
        gap_start = row['Gap_Start']
        gap_end = row['Gap_End']
        
        if var not in df_filled.columns:
            continue
            
        gap_mask = (df_filled.index >= gap_start) & (df_filled.index <= gap_end)
        
        # Apply linear interpolation (simplified version of your logic)
        original_nan_count = df_filled.loc[gap_mask, var].isna().sum()
        
        # Create extended window for interpolation
        start_idx = max(0, df_filled.index.get_indexer([gap_start], method='pad')[0] - 1)
        end_idx = min(len(df_filled) - 1, df_filled.index.get_indexer([gap_end], method='backfill')[0] + 1)
        
        temp_series = df_filled.iloc[start_idx:end_idx + 1][var]
        filled_temp_series = temp_series.interpolate(method='linear', limit_direction='both')
        df_filled.loc[filled_temp_series.index, var] = filled_temp_series
        
        current_nan_count = df_filled.loc[gap_mask, var].isna().sum()
        filled_this_gap = original_nan_count - current_nan_count
        
        if filled_this_gap > 0:
            filled_count_short += filled_this_gap
            filled_vars_short.add(var)
    
    print(f"Linear interpolation filled {filled_count_short} points across {len(filled_vars_short)} variables")
    
    # Apply Random Forest for long gaps
    if enable_rf_filling:
        df_filled, rf_filled_count, rf_filled_vars = fill_long_gaps_with_rf(
            df_filled, gap_report_path, site_name, max_rf_gap_hours
        )
        
        print(f"Random Forest filled {rf_filled_count} additional points across {len(rf_filled_vars)} variables")
    else:
        rf_filled_count = 0
        rf_filled_vars = set()
    
    # Save results
    output_filled_path = f"{site_name}_data_gap_filled{output_suffix}.csv"
    
    # --- FIX: Ensure index name is explicitly set before saving ---
    df_filled.index.name = datetime_column_name # `datetime_column_name` is already a parameter in this function
    # --- END FIX ---
    
    df_filled.to_csv(output_filled_path, index=True)
    print(f"Enhanced gap-filled data saved to: {output_filled_path}")
    
    # Summary
    total_filled = filled_count_short + rf_filled_count
    total_vars = len(filled_vars_short | rf_filled_vars)
    print(f"\n--- Final Summary for {site_name} ---")
    print(f"Total data points filled: {total_filled}")
    print(f"Variables processed: {total_vars}")
    print(f"Linear interpolation: {filled_count_short} points, {len(filled_vars_short)} variables")
    print(f"Random Forest: {rf_filled_count} points, {len(rf_filled_vars)} variables")
    
    return df_filled

# --- Main execution block for the gap filling script ---
if __name__ == "__main__":
    print("--- Starting Enhanced Gap Filling Process (Linear + Random Forest) ---")

    # --- Configuration for input/output files ---
    # These paths should point to the output of your 'subset_and_filling.py' (or 'subset_and_filling_no_soil.py')
    # Assuming your scripts are in the same parent directory structure as before,
    # and you run this 'fill_gaps.py' from the same level as the 'With Soil'/'No Soil' folders.

    # Paths for 'With Soil' data and reports
    GRDR_TRIMMED_WITH_SOIL_PATH = 'With Soil/Filtered CSVs/GRDR_trimmed_data.csv'
    WOOD_TRIMMED_WITH_SOIL_PATH = 'With Soil/Filtered CSVs/WOOD_trimmed_data.csv'
    GRDR_STANDARD_GAPS_REPORT = 'With Soil/Gap Reports/GRDR_gaps_report_standard_gaps.csv'
    GRDR_SOIL_GAPS_REPORT = 'With Soil/Gap Reports/GRDR_gaps_report_soil_gaps.csv'
    WOOD_STANDARD_GAPS_REPORT = 'With Soil/Gap Reports/WOOD_gaps_report_standard_gaps.csv'
    WOOD_SOIL_GAPS_REPORT = 'With Soil/Gap Reports/WOOD_gaps_report_soil_gaps.csv'

    # Paths for 'No Soil' data and reports
    GRDR_TRIMMED_NO_SOIL_PATH = 'No Soil/Filtered CSVs/GRDR_trimmed_data_no_soil.csv'
    WOOD_TRIMMED_NO_SOIL_PATH = 'No Soil/Filtered CSVs/WOOD_trimmed_data_no_soil.csv'
    GRDR_NO_SOIL_GAPS_REPORT = 'No Soil/Gap Reports/GRDR_gaps_report_no_soil_gaps.csv'
    WOOD_NO_SOIL_GAPS_REPORT = 'No Soil/Gap Reports/WOOD_gaps_report_no_soil_gaps.csv'

    # --- Random Forest Specific Configuration ---
    # Max gap size in hours for Random Forest to attempt filling (e.g., 3 months)
    RF_MAX_GAP_HOURS = 3 * 30 * 24 # 3 months * 30 days/month * 24 hours/day = 2160 hours
    ENABLE_RF_FILLING = True # Set to False if you only want linear interpolation

    # --- Process GRDR (With Soil) ---
    try:
        grdr_trimmed_ws = pd.read_csv(GRDR_TRIMMED_WITH_SOIL_PATH, index_col=DATETIME_COLUMN, parse_dates=True)
        print(f"Loaded GRDR (With Soil) trimmed data. Shape: {grdr_trimmed_ws.shape}")
        
        # Apply enhanced filling for standard variables (linear + RF)
        # Note: 'enhanced_fill_gaps_with_rf' will internally use 'all_feature_variables' for RF,
        # which it derives from the input df.
        
        # For standard frequency variables (TAIR, PRES, VT90, VT20)
        # The 'output_suffix' will be combined with the base filename to form the final output name.
        grdr_rf_filled_ws_standard = enhanced_fill_gaps_with_rf(
            grdr_trimmed_ws.copy(), 
            GRDR_STANDARD_GAPS_REPORT, 
            'GRDR', 
            output_suffix='_standard_filled_rf_ws',
            dataframe_reindex_freq_minutes=5, # Standard data is 5-min
            enable_rf_filling=ENABLE_RF_FILLING,
            max_rf_gap_hours=RF_MAX_GAP_HOURS
        )
        
        # For soil frequency variables (SM02, SM04, ST02, ST04)
        grdr_rf_filled_ws_soil = enhanced_fill_gaps_with_rf(
            grdr_trimmed_ws.copy(), 
            GRDR_SOIL_GAPS_REPORT, 
            'GRDR', 
            output_suffix='_soil_filled_rf_ws',
            dataframe_reindex_freq_minutes=30, # Soil data is 30-min
            enable_rf_filling=ENABLE_RF_FILLING,
            max_rf_gap_hours=RF_MAX_GAP_HOURS
        )

    except FileNotFoundError as e:
        print(f"Error loading GRDR (With Soil) trimmed data or reports: {e}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred processing GRDR (With Soil): {e}")

    print("\n" + "="*50 + "\n") # Separator for clarity

    # --- Process WOOD (With Soil) ---
    try:
        wood_trimmed_ws = pd.read_csv(WOOD_TRIMMED_WITH_SOIL_PATH, index_col=DATETIME_COLUMN, parse_dates=True)
        print(f"Loaded WOOD (With Soil) trimmed data. Shape: {wood_trimmed_ws.shape}")

        wood_rf_filled_ws_standard = enhanced_fill_gaps_with_rf(
            wood_trimmed_ws.copy(), 
            WOOD_STANDARD_GAPS_REPORT, 
            'WOOD', 
            output_suffix='_standard_filled_rf_ws',
            dataframe_reindex_freq_minutes=5, 
            enable_rf_filling=ENABLE_RF_FILLING,
            max_rf_gap_hours=RF_MAX_GAP_HOURS
        )
        wood_rf_filled_ws_soil = enhanced_fill_gaps_with_rf(
            wood_trimmed_ws.copy(), 
            WOOD_SOIL_GAPS_REPORT, 
            'WOOD', 
            output_suffix='_soil_filled_rf_ws',
            dataframe_reindex_freq_minutes=30, 
            enable_rf_filling=ENABLE_RF_FILLING,
            max_rf_gap_hours=RF_MAX_GAP_HOURS
        )

    except FileNotFoundError as e:
        print(f"Error loading WOOD (With Soil) trimmed data or reports: {e}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred processing WOOD (With Soil): {e}")

    print("\n" + "="*50 + "\n") # Separator for clarity

    # --- Process GRDR (No Soil) ---
    try:
        grdr_trimmed_ns = pd.read_csv(GRDR_TRIMMED_NO_SOIL_PATH, index_col=DATETIME_COLUMN, parse_dates=True)
        print(f"Loaded GRDR (No Soil) trimmed data. Shape: {grdr_trimmed_ns.shape}")

        grdr_rf_filled_ns = enhanced_fill_gaps_with_rf(
            grdr_trimmed_ns.copy(), 
            GRDR_NO_SOIL_GAPS_REPORT, 
            'GRDR', 
            output_suffix='_no_soil_filled_rf_ns',
            dataframe_reindex_freq_minutes=5, 
            enable_rf_filling=ENABLE_RF_FILLING,
            max_rf_gap_hours=RF_MAX_GAP_HOURS
        )

    except FileNotFoundError as e:
        print(f"Error loading GRDR (No Soil) trimmed data or reports: {e}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred processing GRDR (No Soil): {e}")
    
    print("\n" + "="*50 + "\n") # Separator for clarity

    # --- Process WOOD (No Soil) ---
    try:
        wood_trimmed_ns = pd.read_csv(WOOD_TRIMMED_NO_SOIL_PATH, index_col=DATETIME_COLUMN, parse_dates=True)
        print(f"Loaded WOOD (No Soil) trimmed data. Shape: {wood_trimmed_ns.shape}")

        wood_rf_filled_ns = enhanced_fill_gaps_with_rf(
            wood_trimmed_ns.copy(), 
            WOOD_NO_SOIL_GAPS_REPORT, 
            'WOOD', 
            output_suffix='_no_soil_filled_rf_ns',
            dataframe_reindex_freq_minutes=5, 
            enable_rf_filling=ENABLE_RF_FILLING,
            max_rf_gap_hours=RF_MAX_GAP_HOURS
        )

    except FileNotFoundError as e:
        print(f"Error loading WOOD (No Soil) trimmed data or reports: {e}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred processing WOOD (No Soil): {e}")

    print("\n--- Enhanced Gap Filling Process Completed ---")