import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_box(ax, origin, size, color, alpha=0.85, label=None):
    """Draws a 3D box representing a data component."""
    x, y, z = origin
    dx, dy, dz = size

    vertices = np.array([
        [x, y, z],
        [x+dx, y, z],
        [x+dx, y+dy, z],
        [x, y+dy, z],
        [x, y, z+dz],
        [x+dx, y, z+dz],
        [x+dx, y+dy, z+dz],
        [x, y+dy, z+dz],
    ])

    faces = [
        [vertices[i] for i in [0,1,2,3]], # Bottom
        [vertices[i] for i in [4,5,6,7]], # Top
        [vertices[i] for i in [0,1,5,4]], # Front
        [vertices[i] for i in [2,3,7,6]], # Back
        [vertices[i] for i in [1,2,6,5]], # Right
        [vertices[i] for i in [0,3,7,4]], # Left
    ]

    collection = Poly3DCollection(faces, facecolors=color, edgecolors="k", alpha=alpha)
    ax.add_collection3d(collection)
    
    if label:
        ax.text(x + dx/2, y + dy/2, z + dz + 0.3, label, ha='center', fontsize=10, weight='bold')

def plot_schematic_cube(output_path="plots/v1/data_cube_schematic.png"):
    """Creates a schematic 3D visualization of the NetCDF data structure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    # --- Draw blocks ---

    # 3D variable: temperature(time, lat, lon)
    draw_box(ax, origin=(0, 0, 0), size=(3, 3, 3), color="#8dd3c7", label="Temp(t, lat, lon)")

    # Another 3D variable: pressure(time, lat, lon)
    draw_box(ax, origin=(4, 0, 0), size=(3, 3, 3), color="#fccde5", label="Press(t, lat, lon)")

    # 2D slices (lat, lon) - Meta-data like Height
    draw_box(ax, origin=(8, 0, 0), size=(0.3, 3, 3), color="#fdb462", label="Height(lat, lon)")
    draw_box(ax, origin=(9, 0, 0), size=(0.3, 3, 3), color="#fb8072", label="StationID")

    # 1D time series
    ax.plot([10.5, 12.5], [0.5, 0.5], [1.5, 1.5], color="#e41a1c", linewidth=4)
    ax.text(11.5, 0.5, 2.0, "Time Axis", ha='center', fontsize=10, weight='bold')

    # --- Limits ---
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 4)
    
    # Adjust view for depth
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Schematic 3D Cube saved to: {output_path}")

if __name__ == "__main__":
    plot_schematic_cube()