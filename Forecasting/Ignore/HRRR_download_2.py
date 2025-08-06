from herbie import Herbie
from datetime import datetime, timedelta
import os

# Define your site locations and year
sites = {
    "GRDR": {"lat": 36.80221, "lon": -85.43101, "year": 2021},
    "WOOD": {"lat": 36.99315, "lon": -84.96609, "year": 2024},
}

# Herbie configuration
product = "sfc"  # HRRR surface data
model = "hrrr"
fxx = 0  # 0-hr forecast (analysis)
save_dir = "./hrrr_data"

# Loop through each site and download data for their respective year
for site, info in sites.items():
    lat, lon, year = info["lat"], info["lon"], info["year"]
    print(f"\nStarting download for {site} ({year})")

    start_date = datetime(year, 1, 1, 0)
    end_date = datetime(year, 1, 1, 3)  # up to last hour
    dt = start_date

    site_dir = os.path.join(save_dir, site)
    os.makedirs(site_dir, exist_ok=True)

    while dt <= end_date:
        try:
            H = Herbie(
                dt,
                model=model,
                product=product,
                fxx=fxx,
                save_dir=site_dir,
                verbose=False,
            )

            ds = H.xarray()
            ds_point = ds.sel(lat=lat, lon=lon, method="nearest")

            output_path = os.path.join(site_dir, f"{site}_{dt:%Y%m%d%H}.nc")
            ds_point.to_netcdf(output_path)
            print(f"Downloaded {output_path}")

        except Exception as e:
            print(f"Skipping {dt:%Y-%m-%d %H}: {e}")