import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from glob import glob
from pathlib import Path


def load_station_metadata():
    metadata_files = glob("data/metadata/unzips/*.txt")
    meta_dfs = []
    columns_needed = ["Stations_id", "Geogr.Breite", "Geogr.Laenge", "Stationshoehe"]
    
    for f in metadata_files:
        df = pd.read_csv(
            f, 
            sep=";", 
            engine="python", 
            encoding="latin1", 
            usecols=columns_needed
        )
        meta_dfs.append(df.iloc[[-1]])
    
    metadata = pd.concat(meta_dfs, ignore_index=True)
    metadata = metadata.rename(columns={
        "Stations_id": "station_id",
        "Geogr.Breite": "latitude",
        "Geogr.Laenge": "longitude",
        "Stationshoehe": "height"
    })
    metadata = metadata.drop_duplicates(subset=["station_id"])
    metadata = metadata.dropna(subset=["latitude", "longitude"])
    
    print(f"Loaded metadata for {len(metadata)} unique stations")
    
    return metadata


def plot_stations_on_map(metadata, output_path="plots/stations_map.png"):
    fig = plt.figure(figsize=(12, 14))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Germany: Latitude 47-55°N, Longitude 6-15°E
    ax.set_extent([5, 16, 46, 56], crs=ccrs.PlateCarree())
    
    # map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor='black')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.5)
    
    # gridlines
    gl = ax.gridlines(
        draw_labels=True, 
        linewidth=0.5, 
        color='gray', 
        alpha=0.5, 
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    scatter = ax.scatter(
        metadata['longitude'],
        metadata['latitude'],
        c=metadata['height'],
        cmap='terrain',
        s=20,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.3,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    
    # Add colorbar for elevation
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Station Elevation (m)', fontsize=12)
    
    plt.title(
        f'DWD Weather Stations in Germany\n({len(metadata)} stations)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Saved map to {output_path}")
    
    plt.close()


def main():
    print("="*60)
    print("DWD WEATHER STATIONS MAP VISUALIZATION")
    print("="*60 + "\n")
    
    print("Loading station metadata...")
    metadata = load_station_metadata()
    
    print("Creating map visualization...")
    plot_stations_on_map(metadata)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()