import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_cube(ds, output_path="plots/v1/data_cube_showcase.png"):
    """
    Plots a 3D 'Data Cube' visualization.
    X-axis: Time (downsampled for visibility)
    Y-axis: Station ID
    Z-axis: Variable Type
    Color: Normalized Value
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    vars_to_plot = ["TT_10", "RWS_10", "RF_10"]
    # Downsample time to 1-hour intervals to avoid mesh overcrowding
    ds_plot = ds.isel(time=slice(0, 720, 6)) # First 5 days, hourly
    
    stations = ds_plot.station_id.values
    times = np.arange(len(ds_plot.time))
    var_indices = np.arange(len(vars_to_plot))
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    T, V, S = np.meshgrid(times, var_indices, np.arange(len(stations)), indexing='ij')
    
    values = np.zeros_like(T, dtype=float)
    for i, var in enumerate(vars_to_plot):
        data = ds_plot[var].values.T
        v_min, v_max = np.nanmin(data), np.nanmax(data)
        if v_max > v_min:
            norm_data = (data - v_min) / (v_max - v_min)
        else:
            norm_data = data * 0

        values[:, i, :] = norm_data

    img = ax.scatter(T, S, V, c=values.flatten(), cmap='viridis', s=20, alpha=0.6)
    
    ax.set_xlabel('Time (Hourly Steps)', fontsize=12, labelpad=15)
    ax.set_ylabel('Weather Station Index', fontsize=12, labelpad=15)
    ax.set_zlabel('Variable', fontsize=12, labelpad=15)
    
    ax.set_zticks(var_indices)
    ax.set_zticklabels(vars_to_plot)
    
    plt.title("SYNOP v1 Multivariate Data Cube Showcase", fontsize=16, pad=20)
    
    cbar = fig.colorbar(img, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Normalized Intensity', rotation=270, labelpad=15)
    
    ax.view_init(elev=20, azim=-45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D Cube visualization saved to: {output_path}")

def plot_heatmap_showcase(ds, output_path="plots/v1/heatmap_showcase.png"):
    """
    Creates a 2x2 grid of heatmaps (Time vs Station) for different variables.
    """
    vars_to_plot = ["TT_10", "RWS_10", "RF_10"]
    fig, axes = plt.subplots(len(vars_to_plot), 1, figsize=(15, 10), sharex=True)
    
    # Slice first 7 days
    ds_slice = ds.isel(time=slice(0, 144 * 7))
    
    for i, var in enumerate(vars_to_plot):
        data = ds_slice[var].values
        im = axes[i].imshow(data, aspect='auto', cmap='magma' if var != 'RWS_10' else 'Blues')
        axes[i].set_ylabel(f"{var}\n(Station)", fontsize=10)
        fig.colorbar(im, ax=axes[i], pad=0.01)
        
    axes[-1].set_xlabel("Time (10-min Intervals)", fontsize=12)
    plt.suptitle("SYNOP v1 Temporal Heatmap (7-Day Slice across Stations)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap grid visualization saved to: {output_path}")

if __name__ == "__main__":
    import sys
    
    path = "data/datasetv1/2019.nc"
    if os.path.exists(path):
        print(f"Found real dataset at {path}. Loading...")
        ds = xr.open_dataset(path)
    else:
        print("No .nc file found.")
        sys.exit(1)
        
    plot_3d_cube(ds)
    plot_heatmap_showcase(ds)
    print("\nVisualization complete. Check the 'plots/v1/' directory.")
