import os
from glob import glob

import pandas as pd
import xarray as xr

def create_dataset(year):
    metadata_files = glob("data/metadata/unzips/*.txt")
    meta_dfs = []
    for f in metadata_files:
        df = pd.read_csv(f, sep=";", engine="python")
        meta_dfs.append(df)

    metadata = pd.concat(meta_dfs, ignore_index=True)
    metadata = metadata.rename(columns={
        "Geogr.Breite": "latitude",
        "Geogr.Laenge": "longitude",
        "Geogr.Stationshoehe": "height"
    })

    data_files = glob("data/*/concatenated/*.txt")
    data_list = []
    for f in data_files:
        df = pd.read_csv(f, sep=";", engine="python")
        df = df.rename(columns={"STATIONS_ID": "station_id"})

        df["time"] = pd.to_datetime(df["MESS_DATUM"])
        df = df[df["time"].dt.year == year]

        if not df.empty:
            df = df.set_index(["station_id", "time"])
            data_list.append(df)

    if not data_list:
        print(f"No data for year {year}!")

    combined = pd.concat(data_list, axis=0)
    combined = combined.reset_index()

    merged = pd.merge(combined, metadata, on="station_id", how="left")

    ds = merged.set_index(["station_id", "time"]).to_xarray()

    ds = ds.assign_coords(
        longitude = ("station_id", metadata.set_index("station_id")["longitude"]),
        latitude = ("station_id", metadata.set_index("station_id")["latitude"]),
        height = ("station_id", metadata.set_index("station_id")["height"])
    )

    if "time" not in ds.coords:
        ds = ds.swap_dims({"index": "time"})

    os.makedirs("data/datasetv1", exist_ok=True)
    ds.to_netcdf(f"data/datasetv1/{year}.nc")

    print(f"saved {year}.nc")