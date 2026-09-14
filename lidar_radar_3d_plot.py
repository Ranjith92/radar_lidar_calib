import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from quick_view import get_matrices_init


def load_points(csv_path):
    """Load x,y,z columns from a CSV file."""
    df = pd.read_csv(csv_path)
    pts = df[['x','y','z']].to_numpy()

    # Filter: keep only points with -20 <= x <= 20
    b_box = 8
    mask = (pts[:,0] >= -b_box) & (pts[:,0] <= b_box) & (pts[:,1] >= -b_box) & (pts[:,1] <= b_box)
    return pts[mask]


def load_points_lidar(csv_path):
    """Load x,y,z columns from a CSV file."""
    df = pd.read_csv(csv_path)
    pts = df[['x','y','z']].to_numpy()

    # Filter: keep only points with -20 <= x <= 20
    b_box = 8
    mask = (pts[:,0] >= 6) & (pts[:,0] <= 8) & (pts[:,1] >= -1) & (pts[:,1] <= 5)
    return pts[mask]


# ------------------------------------------------------------
# Load radar + lidar CSV files
# ------------------------------------------------------------
lidar_pts = load_points_lidar("test_data/test_data/8.csv")
radar_pts = load_points("test_data/test_data/8_rad.csv")

# ------------------------------------------------------------
# Optional: apply radar→lidar transform
# ------------------------------------------------------------
R_rl = np.load("outputs/R_radar_lidar.npy")
t_rl = np.load("outputs/t_radar_lidar.npy")

# Initial rotation & translation value from quick_view.py
R_init, t_init = get_matrices_init("FL")

radar_aligned = (R_rl @ radar_pts.T).T + t_rl
radar_init = (R_init @ radar_pts.T).T + t_init
radar_init_aligned = (R_rl @ radar_init.T).T + t_rl

plt.figure(figsize=(8, 8))
plt.title("Radar–Lidar Alignment Visualization")
# Lidar points (blue)
plt.scatter(lidar_pts[:, 0], lidar_pts[:, 1], c='blue', label='Lidar points')
# Radar points before alignment (red)
plt.scatter(radar_pts[:, 0], radar_pts[:, 1], c='red', label='Radar (raw)')
# Radar points after alignment (green)
plt.scatter(radar_aligned[:, 0], radar_aligned[:, 1], c='green', label='Radar (aligned)')
# Radar points after alignment (orange)
plt.scatter(radar_init_aligned[:, 0], radar_init_aligned[:, 1], c='orange', label='Radar (init_aligned)')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 3D Plot
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Lidar points (blue)
ax.scatter(lidar_pts[:,0], lidar_pts[:,1], lidar_pts[:,2],
           c='blue', s=5, label='Lidar')

# Radar raw (red)
ax.scatter(radar_pts[:,0], radar_pts[:,1], radar_pts[:,2],
           c='red', s=10, label='Radar (raw)')
#print("radar Z-axis:\n", radar_pts[:,2])

# Radar aligned (green)
ax.scatter(radar_aligned[:,0], radar_aligned[:,1], radar_aligned[:,2],
           c='green', s=15, label='Radar (aligned)')

# Radar init aligned (orange)
ax.scatter(radar_init_aligned[:,0], radar_init_aligned[:,1], radar_init_aligned[:,2],
           c='orange', s=15, label='Radar (init_aligned)')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("3D Radar–Lidar Visualization")

plt.show()


