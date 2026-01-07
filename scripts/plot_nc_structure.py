import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_nc_structure(ds, output_path="plots/v1/nc_structure.png"):
    """Visualizes the NetCDF file structure mapping variables to dimensions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dims = list(ds.dims.keys())
    vars_ = list(ds.variables.keys())

    matrix = []
    for var in vars_:
        row = []
        for dim in dims:
            row.append(dim in ds[var].dims)
        matrix.append(row)

    df = pd.DataFrame(matrix, index=vars_, columns=dims)

    # Calculate size dynamically based on entry count
    fig, ax = plt.subplots(figsize=(2 + len(dims)*1.5, 1 + len(vars_)*0.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.replace({True: "✓", False: ""}).values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    # Styling labels
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        elif cell.get_text().get_text() == "✓":
            cell.set_facecolor('#e1e7f4')

    plt.title("SYNOP v1 NetCDF Structure: Variables vs Dimensions", fontsize=18, pad=30, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"NetCDF structure table saved to: {output_path}")

if __name__ == "__main__":
    path = "data/datasetv1/1990.nc"
    if os.path.exists(path):
        print(f"Loading real dataset from {path}...")
        ds = xr.open_dataset(path)
    else:
        print("No real .nc file found")
        sys.exit(1)
    
    plot_nc_structure(ds)
