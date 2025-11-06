import os
from glob import glob
import calendar
from functools import reduce

import pandas as pd
import xarray as xr

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

    print("--- finished reading metadata ---")

    years = [i for i in range(1990, 2025)]
    for year in years:
        data_files = glob("data/*/concatenated/*.txt")
        data_list = []

        lines_per_day = 6 * 24
        lines_in_year = lines_per_day * (366 if calendar.isleap(year) else 365)
        for f in data_files:
            with open(f, "rb") as station:
                header_line = station.readline().rstrip(b"\n").decode("latin1")
                header = [col.strip() for col in header_line.split(";")]

                first_line = station.readline().rstrip(b"\n").decode("latin1")
                if not first_line:
                    print(f"Skipping empty file {f}")
                    continue

                if os.path.getsize(f) > 1024:
                    station.seek(-1024, 2)

                lines = station.read().splitlines()
                if not lines:
                    last_line = first_line
                else:
                    last_line = lines[-1].decode("latin1")

            first_date = pd.to_datetime(first_line.split(";")[1])
            last_date = pd.to_datetime(last_line.split(";")[1])

            if last_date.year < year or first_date.year > year:
                continue

            skip_rows = int((pd.Timestamp(year=year, month=1, day=1, hour=0, minute=0) - first_date).total_seconds() / 600)

            df = pd.read_csv(
                f,
                sep=";",
                header=None,
                names=header,
                engine="python",
                encoding="latin1",
                skiprows=skip_rows + 1 if skip_rows >= 0 else 1,
                nrows=lines_in_year if skip_rows >= 0 else lines_in_year + skip_rows
            )

            df = df.rename(columns={"STATIONS_ID": "station_id"})
            df["MESS_DATUM"] = df["MESS_DATUM"].astype(str).str.zfill(12)
            df["time"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H%M", errors="coerce")
            df = df.drop(columns=["MESS_DATUM"])

            # print(df["time"].iloc[0], df["time"].iloc[-1])

            if not df.empty:
                df = df.set_index(["station_id", "time"])
                if "QN" in df.columns:
                    df = df.drop(columns=["QN"])

                df = df.reset_index()
                df = df.drop_duplicates(subset=["station_id", "time"])
                data_list.append(df)

        print(f"--- finished reading data_files for {year} ---")

        if not data_list:
            print(f"No data for year {year}")
            return

        combined = pd.concat(data_list, axis=0)
        combined = combined.groupby(["station_id", "time"], as_index=False).first()

        merged = pd.merge(combined, metadata, on="station_id", how="left")
        merged["time"] = combined["time"]

        ds = merged.set_index(["station_id", "time"]).to_xarray()

        os.makedirs("data/datasetv1", exist_ok=True)
        ds.to_netcdf(f"data/datasetv1/{year}.nc")

        print(f"saved {year}.nc")

if __name__ == "__main__":
    print("---------- Starting Dataset Creation Workflow ----------\n")

    create_dataset()

    print("\n---------- Workflow Complete ----------")