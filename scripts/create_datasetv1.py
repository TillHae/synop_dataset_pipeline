import os
from glob import glob
from multiprocessing import Pool, cpu_count
import calendar
from datetime import datetime

import pandas as pd
import numpy as np

def process_file(file_path, year):
    with open(file_path, "rb") as station:
        header_line = station.readline().rstrip(b"\n").decode("latin1")
        header = [col.strip() for col in header_line.split(";")]

        first_line = station.readline().rstrip(b"\n").decode("latin1")
        if not first_line:
            return None

        if os.path.getsize(file_path) > 1024:
            station.seek(-1024, 2)
        lines = station.read().splitlines()
        last_line = first_line if not lines else lines[-1].decode("latin1")

    first_date = pd.to_datetime(first_line.split(";")[1])
    last_date = pd.to_datetime(last_line.split(";")[1])

    if last_date.year < year or first_date.year > year:
        return None

    lines_per_day = 6 * 24
    lines_in_year = lines_per_day * (366 if calendar.isleap(year) else 365)
    skip_rows = (datetime(year, 1, 1) - first_date).days * lines_per_day

    df = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        names=header,
        engine="python",
        encoding="latin1",
        skiprows=skip_rows + 1 if skip_rows >= 0 else 1,
        nrows=lines_in_year if skip_rows >= 0 else lines_in_year + skip_rows,
        on_bad_lines="skip"
    )

    if df.empty:
        return None

    df = df.rename(columns={"STATIONS_ID": "station_id"})
    df["MESS_DATUM"] = df["MESS_DATUM"].astype(str).str.zfill(12)
    df["time"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H%M", errors="coerce")
    df = df.drop(columns=["MESS_DATUM"])

    df = df[df["time"].dt.year == year]

    return df

def process_year(year, metadata):
    data_files = glob("data/*/concatenated/*.txt")
    if not data_files:
        print(f"No data files found for year {year}")
        return

    num_workers = min(72, int(cpu_count() * 0.9))
    print(f"Processing year {year} with {num_workers}/{cpu_count()} workers...")

    with Pool(num_workers) as pool:
        dfs = pool.starmap(process_file, [(f, year) for f in data_files])

    dfs = [df for df in dfs if df is not None]
    if not dfs:
        print(f"No valid data for year {year}")
        return

    combined = pd.concat(dfs, axis=0)
    combined = combined.groupby(["station_id", "time"], as_index=False).first()

    merged = pd.merge(combined, metadata, on="station_id", how="left")

    cols_to_check = [c for c in merged.columns if c not in ["station_id", "time"]]
    merged[cols_to_check] = merged[cols_to_check].replace(-999, np.nan)
    for col in cols_to_check:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    ds = merged.set_index(["station_id", "time"]).to_xarray()

    os.makedirs("data/datasetv1", exist_ok=True)
    ds.to_netcdf(f"data/datasetv1/{year}.nc")
    print(f"Saved {year}.nc ({len(merged)} rows)")

def create_dataset():
    metadata_files = glob("data/metadata/unzips/*.txt")
    meta_dfs = []
    columns_needed = ["Stations_id", "Geogr.Breite", "Geogr.Laenge", "Stationshoehe"]
    for f in metadata_files:
        df = pd.read_csv(f, sep=";", engine="python", encoding="latin1", usecols=columns_needed)
        meta_dfs.append(df.iloc[[-1]])

    metadata = pd.concat(meta_dfs, ignore_index=True)
    metadata = metadata.rename(columns={
        "Stations_id": "station_id",
        "Geogr.Breite": "latitude",
        "Geogr.Laenge": "longitude",
        "Stationshoehe": "height"
    })
    print("--- Finished reading metadata ---")

    for year in range(1990, 2025):
        process_year(year, metadata)

if __name__ == "__main__":
    print("---------- Starting Dataset Creation Workflow ----------\n")
    create_dataset()
    print("\n---------- Workflow Complete ----------")