import pandas as pd
from herbie import Herbie
import xarray as xr
import os
import numpy as np
import datetime as dt
import warnings
import cfgrib

# Disable the specific FutureWarning from xarray/cfgrib
warnings.filterwarnings(
    "ignore",
    message="In a future version, xarray will not decode timedelta values.*",
    category=FutureWarning,
)

# --- Configuration ---

SITE_CONFIGS = {
    "GRDR": {
        "latitude": 36.80221,
        "longitude": -85.43101,
        "start_date": '2021-01-01',
        "end_date": '2021-01-03'
    },
    "WOOD": {
        "latitude": 36.99315,
        "longitude": -84.96609,
        "start_date": '2024-01-01',
        "end_date": '2024-01-03'
    }
}

# List of HRRR variables you want to download
HRRR_VARIABLES = [
    'TMP:2 m above ground',      # 2-meter air temperature (Kelvin)
    'PBLH:0.000',                # Planetary Boundary Layer Height (meters)
    'UGRD:10 m above ground',    # U-component of 10-m wind (m/s)
    'VGRD:10 m above ground',    # V-component of 10-m wind (m/s)
    'TCDC:entire atmosphere',    # Total Cloud Cover (%)
    'DLWRF:surface',             # Downward Longwave Radiation Flux (W/m^2)
    'DSWRF:surface',             # Downward Shortwave Radiation Flux (W/m^2)
    'HGT:surface',               # Geopotential Height at surface (m)
]

# Forecast hours to download (fxx=00 is analysis, fxx=01 is 1-hour forecast, etc.)
# For historical analysis/reanalysis, fxx=00 is usually the closest to observations.
FORECAST_HOURS = 0 # Only download the analysis field (f00)

# Directory to save downloaded CSVs
OUTPUT_DATA_DIR = "./HRRR_Downloaded_Data"

# --- Create Output Directory ---
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

print("Starting HRRR data download...")

# Ensure cfgrib_short_names map is defined as I provided in the previous step
cfgrib_short_names = {
    'TMP:2 m above ground':   't2m',
    'PBLH:0.000':             'pblh',
    'UGRD:10 m above ground': 'u10',
    'VGRD:10 m above ground': 'v10',
    'TCDC:entire atmosphere': 'tcc',
    'DLWRF:surface':          'dlwrf',
    'DSWRF:surface':          'dswrf',
    'HGT:surface':            'gh',
}
cfgrib_to_hrrr_vars_map = {v: k.split(':')[0] for k, v in cfgrib_short_names.items()}
cfgrib_to_hrrr_vars_map['t2m'] = 'TMP_C'
cfgrib_to_hrrr_vars_map['u10'] = 'UGRD'
cfgrib_to_hrrr_vars_map['v10'] = 'VGRD'
cfgrib_to_hrrr_vars_map['tcc'] = 'TCDC'
cfgrib_to_hrrr_vars_map['gh'] = 'HGT'


# --- Loop through each site's configuration ---
for site_name, config in SITE_CONFIGS.items():
    site_lat = config["latitude"]
    site_lon = config["longitude"]
    start_date_str = config["start_date"]
    end_date_str = config["end_date"]
    
    date_range = pd.to_datetime(pd.date_range(start=start_date_str, end=end_date_str, freq='D'))
    
    print(f"\n--- Downloading HRRR data for {site_name} (Lat: {site_lat}, Lon: {site_lon}) ---")
    print(f"    Date range: {start_date_str} to {end_date_str}")
    
    all_site_data = []

    # Use the short names for Herbie's internal filtering for download.
    hrrr_search_string = "|".join(cfgrib_short_names.values())
    

    for date in date_range:
        for hr_utc in range(24): 
            model_run_time = date + dt.timedelta(hours=hr_utc)
            
            H = Herbie(
                model_run_time.strftime("%Y-%m-%d %H:%M"),
                model='hrrr',
                product='sfc',
                fxx=FORECAST_HOURS,
                verbose=True, # Keep verbose for Herbie's info
                searchString=hrrr_search_string # Use the short names for search
            )

            try:
                # This is the key: Herbie's xarray() method has an xarray_kwargs argument
                # You can pass arguments to xarray.open_dataset here.
                # 'combine="by_code"' is the default behavior that leads to multiple datasets.
                # We want to force it to combine them.
                # However, the note "multiple hypercubes" implies `cfgrib` is returning
                # a list of distinct datasets for each GRIB message.
                
                # Let's try to get a single xarray.Dataset where lat/lon are proper coordinates
                # using the `filter_by_keys` within `backend_kwargs` in H.xarray().
                # This is the cleanest way with Herbie to get a single, combined dataset.

                ds = H.xarray(
                    # This will pass these arguments to cfgrib.open_datasets or xarray.open_dataset
                    # These arguments tell cfgrib how to interpret the GRIB file
                    # 'merge=True' combines the messages into a single dataset.
                    # 'filter_by_keys' limits variables read from the file.
                    # 'index_by_keys' tells it to make specific GRIB keys into xarray coordinates.
                    xarray_kwargs={
                        'engine': 'cfgrib',
                        'backend_kwargs': {
                            'filter_by_keys': {'shortName': list(cfgrib_short_names.values())},
                            'index_by_keys': ['latitude', 'longitude', 'time', 'step', 'valid_time']
                        }
                    },
                    # Ensure it only downloads the specific part of the GRIB file if possible
                    # This also ensures lat/lon for point selection.
                    # xarray_sel_kwargs={'latitude': site_lat, 'longitude': site_lon, 'method': 'nearest'}
                    # No, let's keep the .sel() outside, just ensure ds is correctly formed.
                )
                
                if ds is None:
                    print(f"  No data (ds is None) found for {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')}. Skipping.")
                    continue
                
                # If Herbie's xarray() still returns a list, try to combine it explicitly
                # This fallback ensures we have a single dataset to work with.
                if isinstance(ds, list):
                    ds = [d for d in ds if d is not None and len(d.data_vars) > 0]
                    if not ds:
                        print(f"  No valid datasets in list for {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')}. Skipping.")
                        continue
                    try:
                        # Use combine_by_coords as it's more robust, assuming some common coords might exist.
                        # It will attempt to align along shared coordinates.
                        ds = xr.combine_by_coords(ds, compat='override', combine_attrs='drop')
                    except Exception as combine_e:
                        print(f"  Error combining datasets for {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')}: {combine_e}. Skipping.")
                        continue


                # Now, ds should be a single xarray.Dataset.
                # Re-verify that latitude is a coordinate after this step.
                if 'latitude' not in ds.coords:
                    # If it's a data variable, set it as a coordinate.
                    if 'latitude' in ds.data_vars and 'longitude' in ds.data_vars:
                        ds = ds.set_coords(['latitude', 'longitude'])
                    else:
                        print(f"  Error: Combined Dataset for {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')} still has no 'latitude' coordinate or is malformed. DS content:\n{ds}\n Skipping.")
                        continue
                
                # --- DEBUGGING: Print info about the combined dataset ---
                print(f"\n--- DEBUG: Final Dataset structure for {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')} (after combining and set_coords) ---")
                print(ds) # Print full dataset structure
                print(f"--- END DEBUG ---\n")
                # --- END DEBUGGING ---

                # Perform point selection
                ds_point = ds.sel(latitude=site_lat, longitude=site_lon, method='nearest')

                row_data = {
                    'UTCTimestamp': ds_point['valid_time'].item(),
                    'HRRR_RunTime': model_run_time,
                    'HRRR_NearestLat': ds_point['latitude'].item(),
                    'HRRR_NearestLon': ds_point['longitude'].item(),
                }
                
                found_any_data = False
                for hrrr_var_full_name in HRRR_VARIABLES:
                    var_cfgrib_name = cfgrib_short_names[hrrr_var_full_name]
                    output_col_name = cfgrib_to_hrrr_vars_map[var_cfgrib_name]

                    if var_cfgrib_name in ds_point.data_vars:
                        value = ds_point[var_cfgrib_name].item()
                        
                        if var_cfgrib_name == 't2m':
                            row_data[output_col_name] = value - 273.15
                        elif var_cfgrib_name in ['u10', 'v10']:
                            row_data[output_col_name] = value
                        else:
                            row_data[output_col_name] = value
                        found_any_data = True
                    else:
                        row_data[output_col_name] = np.nan

                if not found_any_data:
                    print(f"  No extractable data for any requested variable for {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')}. Skipping.")
                    continue

                all_site_data.append(row_data)

            except Exception as e:
                print(f"  General Error during processing for HRRR run {model_run_time.strftime('%Y-%m-%d %H:%M UTCTZ')}: {e}")
                continue
                
    # --- Post-processing for each site ---
    if all_site_data:
        df_site_hrrr = pd.DataFrame(all_site_data)
        
        if 'UGRD' in df_site_hrrr.columns and 'VGRD' in df_site_hrrr.columns:
            df_site_hrrr['wind_speed_10m_mps'] = np.sqrt(df_site_hrrr['UGRD']**2 + df_site_hrrr['VGRD']**2)
            df_site_hrrr['wind_dir_10m_deg_from'] = (270 - np.degrees(np.arctan2(df_site_hrrr['UGRD'], df_site_hrrr['VGRD']))) % 360
            df_site_hrrr.drop(columns=['UGRD', 'VGRD'], inplace=True)
        
        df_site_hrrr['UTCTimestamp'] = pd.to_datetime(df_site_hrrr['UTCTimestamp'])
        df_site_hrrr.set_index('UTCTimestamp', inplace=True)
        df_site_hrrr.sort_index(inplace=True)
        
        output_csv_path = os.path.join(OUTPUT_DATA_DIR, f"{site_name}_HRRR_surface_data.csv")
        df_site_hrrr.to_csv(output_csv_path)
        print(f"Successfully downloaded and saved HRRR data for {site_name} to {output_csv_path}")
    else:
        print(f"No HRRR data successfully downloaded for {site_name}.")

print("\nHRRR data download process complete.")