import pandas as pd
import numpy as np
import os

# --- Helper Function: Consolidate Overlapping Time Spans ---
def consolidate_spans(spans):
    """
    Merges overlapping or adjacent time spans (start, end) into a minimal set of merged spans.
    Assumes spans are already sorted by start time.
    """
    if not spans:
        return []

    merged = []
    current_start, current_end = spans[0]

    for next_start, next_end in spans[1:]:
        # If the next span overlaps or is adjacent to the current merged span
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end)) # Add the last merged span
    return merged

# --- Function: Find Major Gaps based on Variable NaNs ---
def find_major_variable_gaps(df, critical_vars, gap_threshold_months=2, data_freq='5min'):
    """
    Identifies major gaps (consecutive NaNs longer than threshold) within specified
    critical variables and consolidates them into a list of overall major gap spans.

    Args:
        df (pd.DataFrame): The input DataFrame.
        critical_vars (list): List of column names to check for NaN blocks.
        gap_threshold_months (int): Duration in months for a NaN block to be considered major.
        data_freq (str): Expected frequency of the data (e.g., '5min') to calculate NaN block durations correctly.

    Returns:
        list: A list of (start_timestamp, end_timestamp) tuples for consolidated major gaps.
    """
    all_major_gap_spans = []
    # FIX: Use days for Timedelta, as 'months' is not directly supported
    major_gap_threshold = pd.Timedelta(days=gap_threshold_months * 30.44) # Approx. 30.44 days per month
    expected_step = pd.to_timedelta(data_freq)

    print(f"  Analyzing individual variable gaps (threshold > {gap_threshold_months} months = {major_gap_threshold})...")

    for var in critical_vars:
        if var not in df.columns:
            print(f"  Warning: Variable '{var}' not found in DataFrame for gap analysis. Skipping.")
            continue

        is_nan = df[var].isnull()
        if not is_nan.any(): # No NaNs for this variable
            continue

        # Identify where NaN blocks start and end
        nan_block_starts_mask = is_nan & ~is_nan.shift(1).fillna(False)
        nan_block_ends_mask = is_nan & ~is_nan.shift(-1).fillna(False)

        nan_starts = df.index[nan_block_starts_mask].tolist()
        nan_ends = df.index[nan_block_ends_mask].tolist()

        if len(nan_starts) != len(nan_ends):
            print(f"  Warning: Mismatch in NaN block starts/ends for {var}. This should not happen with proper mask logic.")
            continue

        for i in range(len(nan_starts)):
            block_start = nan_starts[i]
            block_end = nan_ends[i]
            
            # Calculate duration including the block_end timestamp itself
            duration = block_end - block_start + expected_step
            
            if duration > major_gap_threshold:
                all_major_gap_spans.append((block_start, block_end))
                print(f"    Major gap identified for '{var}': {block_start} to {block_end} (Duration: {duration})")

    # Sort all identified major gap spans by start time before consolidating
    all_major_gap_spans.sort(key=lambda x: x[0])
    consolidated_major_gaps = consolidate_spans(all_major_gap_spans)
    
    print(f"  Total {len(consolidated_major_gaps)} consolidated major variable gaps found.")
    return consolidated_major_gaps

# --- Function: Split DataFrame by a List of Gap Spans ---
def split_df_by_spans(df, major_gap_spans, data_freq='5min'):
    """
    Splits a DataFrame into continuous segments based on a list of major gap spans.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        major_gap_spans (list): A list of (start_timestamp, end_timestamp) tuples
                                 representing the major gaps to split by.
        data_freq (str): The expected frequency of the data (e.g., '5min').
                         Used to ensure correct segment boundaries relative to existing data points.

    Returns:
        list: A list of pd.DataFrames, each representing a continuous segment.
    """
    segments = []
    current_segment_start_time = df.index.min()
    expected_step = pd.to_timedelta(data_freq)
    
    # Ensure major_gap_spans are sorted by start time (already done by find_major_variable_gaps)

    print("\n  Splitting DataFrame into segments...")
    for i, (gap_start, gap_end) in enumerate(major_gap_spans):
        # Determine the last timestamp to include in the current segment
        # This should be the last actual timestamp in df that is before gap_start
        segment_end_time = df.index[df.index < gap_start].max()

        if pd.isna(segment_end_time) and current_segment_start_time < gap_start: # Case where first segment might be empty or just starts before gap
            # This happens if gap_start is at or very near the df.index.min()
            pass # No segment before this gap, or it was already covered by a previous gap
        elif current_segment_start_time <= segment_end_time: # Ensure the segment has a valid range
            segment = df.loc[current_segment_start_time : segment_end_time].copy()
            if not segment.empty:
                segments.append(segment)
                print(f"    Segment {len(segments)}: {segment.index.min()} to {segment.index.max()} (before gap {gap_start})")
            else:
                print(f"    Warning: Empty segment generated before gap {gap_start}. Range: {current_segment_start_time} to {segment_end_time}")

        # Update the start time for the next segment to be just after the gap ends
        # This should be the first actual timestamp in df that is after gap_end
        current_segment_start_time = df.index[df.index > gap_end].min()
        
        if pd.isna(current_segment_start_time):
            # If there's no data after this gap, then we're done with segments
            current_segment_start_time = df.index.max() + expected_step # Set to a value beyond the data to stop further segment creation

    # Add the final segment (from the last gap end to the end of the DataFrame)
    if current_segment_start_time <= df.index.max(): # Check if there's still data left after the last gap
        final_segment = df.loc[current_segment_start_time : df.index.max()].copy()
        if not final_segment.empty:
            segments.append(final_segment)
            print(f"    Final Segment {len(segments)}: {final_segment.index.min()} to {final_segment.index.max()}")
        else:
            print(f"    Warning: Final segment is empty. Range: {current_segment_start_time} to {df.index.max()}")
    else:
        print("    No data left for a final segment after the last major gap.")


    return segments

# --- Function to Report Remaining Gaps in Critical Variables for a Segment (unchanged, but added to this file) ---
def report_gaps_in_critical_variables_for_segment(file_path, segment_name, critical_vars, output_report_base_dir):
    """
    Loads a data segment file, identifies and reports the number of missing values (NaNs)
    for each specified critical variable. Also saves a CSV listing all datetimes
    where any of these critical variables have missing data for this specific segment.

    Args:
        file_path (str): Path to the input CSV file for a segment.
        segment_name (str): A descriptive name for the segment (e.g., 'GRDR_segment_1_20220101_to_20220301').
        critical_vars (list): A list of column names to check for missing values.
        output_report_base_dir (str): The base directory where reports for this segment should be saved.
    """
    print(f"\n--- Reporting Gaps for Critical Variables in {segment_name} ---")

    try:
        df = pd.read_csv(file_path, index_col='UTCTimestampCollected', parse_dates=True)
        df.sort_index(inplace=True)

        if df.empty:
            print(f"Segment {segment_name} is empty. No variables to check.")
            return

        # print(f"Loaded {segment_name}. Shape: {df.shape}. Time range: {df.index.min()} to {df.index.max()}")

        # Ensure all critical variables exist in the DataFrame
        missing_critical_vars_in_df = [var for var in critical_vars if var not in df.columns]
        if missing_critical_vars_in_df:
            print(f"Warning: The following critical variables were not found in {segment_name}: {missing_critical_vars_in_df}")
            critical_vars_present = [var for var in critical_vars if var in df.columns]
        else:
            critical_vars_present = critical_vars

        if not critical_vars_present:
            print(f"No critical variables found in {segment_name} to check for gaps.")
            return

        print("Missing values per critical variable for this segment:")
        missing_counts = df[critical_vars_present].isnull().sum()
        print(missing_counts)

        # Collect all datetimes where ANY critical variable has a missing value
        all_missing_datetimes_set = set()
        for col in critical_vars_present:
            missing_indices_col = df[df[col].isna()].index
            all_missing_datetimes_set.update(missing_indices_col)

        all_missing_datetimes_list = sorted(list(all_missing_datetimes_set))

        # Create segment-specific report directory within the output_report_base_dir
        # This keeps the reports organized by segment
        segment_report_dir = os.path.join(output_report_base_dir, f"{segment_name}_variable_gaps_reports")
        os.makedirs(segment_report_dir, exist_ok=True) # Ensure directory exists

        output_report_file = os.path.join(segment_report_dir, f"{segment_name}_remaining_variable_gaps.csv")
        if all_missing_datetimes_list:
            missing_df_report = pd.DataFrame({'UTCTimestampCollected': all_missing_datetimes_list})
            missing_df_report.to_csv(output_report_file, index=False)
            print(f"Datetimes where critical variables still have gaps for {segment_name} saved to: {output_report_file}")
        else:
            print(f"No remaining gaps found in any critical variables for {segment_name}.")

    except Exception as e:
        print(f"An error occurred while reporting gaps for {segment_name}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the critical variables to check for remaining gaps
    CRITICAL_VARIABLES = ['TAIR', 'PRES', 'SM02', 'VT90', 'VT20', 'ST02', 'SM04', 'ST04']

    # Paths to your final combined gap-filled data files
    GRDR_FILE = 'GRDR_data_gap_filled_combined_ws.csv'
    WOOD_FILE = 'WOOD_data_gap_filled_combined_ws.csv'

    # --- Processing Function for a Site ---
    def process_site_comprehensively(file_path, site_name, critical_vars, gap_threshold_months=2, data_freq='5min'):
        print(f"\n" + "="*70)
        print(f"Starting Comprehensive Processing for {site_name} Data")
        print(f"="*70)

        try:
            # Load the original combined gap-filled data
            main_df = pd.read_csv(file_path, index_col='UTCTimestampCollected', parse_dates=True)
            main_df.sort_index(inplace=True)

            if main_df.empty:
                print(f"Main DataFrame for {site_name} is empty. Skipping processing.")
                return

            print(f"Loaded {site_name} data for gap analysis. Shape: {main_df.shape}. Time range: {main_df.index.min()} to {main_df.index.max()}")

            # Step 1: Find major variable-specific gaps and consolidate them
            major_gap_spans = find_major_variable_gaps(main_df, critical_vars, gap_threshold_months, data_freq)

            # Step 2: Split the main DataFrame into continuous segments based on these consolidated gaps
            print("\n--- Splitting Data into Continuous Segments ---")
            continuous_segments_list = split_df_by_spans(main_df, major_gap_spans, data_freq)
            
            output_segments_dir = f"{site_name}_continuous_segments_by_variable_gaps"
            os.makedirs(output_segments_dir, exist_ok=True)
            print(f"  Created output directory: {output_segments_dir}")

            if not continuous_segments_list:
                print(f"No continuous segments generated for {site_name}. This may indicate that the entire dataset is one big gap.")
                # If no splits or segments, try to save the original df as one segment if not empty
                if not main_df.empty:
                    start_ts = main_df.index.min().strftime('%Y%m%d_%H%M')
                    end_ts = main_df.index.max().strftime('%Y%m%d_%H%M')
                    output_file = os.path.join(output_segments_dir, f"{site_name}_segment_1_{start_ts}_to_{end_ts}.csv")
                    main_df.to_csv(output_file)
                    print(f"  Saved main dataset as segment 1 to: {output_file} (no major variable gaps found).")
                    # No need to report on this single segment if no gaps, as this process is about removing major gaps.
                else:
                    print(f"  No data to save for {site_name}.")
                return # Exit the function if no segments generated or saved

            else:
                # Save each generated segment to a CSV file
                print("\n--- Saving Continuous Segments ---")
                # Need to iterate through the actual files saved to ensure correct file paths for reporting
                saved_segment_files = []
                for i, segment_df in enumerate(continuous_segments_list):
                    if not segment_df.empty:
                        start_ts = segment_df.index.min().strftime('%Y%m%d_%H%M')
                        end_ts = segment_df.index.max().strftime('%Y%m%d_%H%M')
                        output_file = os.path.join(output_segments_dir, f"{site_name}_segment_{i+1}_{start_ts}_to_{end_ts}.csv")
                        segment_df.to_csv(output_file)
                        print(f"  Saved segment {i+1} ({start_ts} to {end_ts}) to: {output_file}")
                        saved_segment_files.append(output_file)
                    else:
                        print(f"  Warning: An empty segment was generated. Skipping save for segment {i+1}.")
                
                # Step 3: Report remaining gaps for each new segment
                print(f"\n--- Reporting Remaining Gaps for {site_name} Segments ---")
                for segment_file in sorted(saved_segment_files): # Sort files for consistent processing order
                    segment_name_for_report = os.path.basename(segment_file).replace('.csv', '')
                    report_gaps_in_critical_variables_for_segment(segment_file, segment_name_for_report, critical_vars, output_segments_dir)


        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Please ensure the path is correct and the file exists.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {site_name}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    # --- Run Processing for Both Sites ---
    # NOTE: Ensure data_freq ('5min') matches the actual frequency of your data for accurate gap detection.
    process_site_comprehensively(GRDR_FILE, 'GRDR', CRITICAL_VARIABLES, gap_threshold_months=2, data_freq='5min')
    process_site_comprehensively(WOOD_FILE, 'WOOD', CRITICAL_VARIABLES, gap_threshold_months=2, data_freq='5min')