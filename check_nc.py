import os
from glob import glob
import xarray as xr
import numpy as np
import pandas as pd
import calendar

def check_file(ds, filename, year):
    n_stations = len(ds.get("station_id", []))
    n_times = len(ds.get("time", []))
    
    if "time" in ds.coords:
        start_time = pd.to_datetime(ds["time"].values[0])
        end_time = pd.to_datetime(ds["time"].values[-1])
    else:
        start_time = end_time = None

    numeric_cols = [var for var in ds.data_vars if np.issubdtype(ds[var].dtype, np.number)]
    all_numeric = all(np.issubdtype(ds[var].dtype, np.number) for var in numeric_cols)

    if numeric_cols:
        df_numeric = ds[numeric_cols].to_dataframe()
        total_values = df_numeric.size
        nan_values = df_numeric.isna().sum().sum()
        nan_percentage = nan_values / total_values if total_values > 0 else 0
    else:
        nan_percentage = 0.0

    days_in_year = 366 if calendar.isleap(year) else 365
    expected_rows = n_stations * 6 * 24 * days_in_year
    data_ratio = len(ds.to_dataframe()) / expected_rows if expected_rows else 0
    enough_data = data_ratio >= 0.8

    print(
        f"{os.path.basename(filename)} | "
        f"stations:{n_stations} | "
        f"time:{start_time}->{end_time} | "
        f"numeric_ok:{all_numeric} | "
        f"enough_data:{enough_data:.1%} | "
        f"nan_percentage:{nan_percentage:.1%}"
    )

def main():
    nc_files = glob("data/datasetv1/*.nc")
    if not nc_files:
        print("No NetCDF files found in data/datasetv1/")
        return

    for f in sorted(nc_files):
        year = int(os.path.basename(f).split(".")[0])
        ds = xr.open_dataset(f)
        check_file(ds, f, year)
        ds.close()

if __name__ == "__main__":
    print("Starting QC for NetCDF dataset files...\n")
    main()
    print("\nQC complete.")