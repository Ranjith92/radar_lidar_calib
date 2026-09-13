import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Load your selected reflector means
# ------------------------------------------------------------
lidar = np.load("outputs/lidar_means.npy")[:, :2]   # XY only
radar = np.load("outputs/radar_means.npy")[:, :2]   # XY only

# ------------------------------------------------------------
# Load the solved radar→lidar extrinsics
# ------------------------------------------------------------
R = np.load("outputs/R_radar_lidar.npy")[:2, :2]    # 2×2 rotation
t = np.load("outputs/t_radar_lidar.npy")[:2]        # XY translation

# ------------------------------------------------------------
# Apply alignment: radar_aligned = R * radar + t
# ------------------------------------------------------------
radar_aligned = (R @ radar.T).T + t

# ------------------------------------------------------------
# Plot everything
# ------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.title("Radar–Lidar Alignment Visualization")

# Lidar points (blue)
plt.scatter(lidar[:, 0], lidar[:, 1], c='blue', label='Lidar points')

# Radar points before alignment (red)
plt.scatter(radar[:, 0], radar[:, 1], c='red', label='Radar (raw)')

# Radar points after alignment (green)
plt.scatter(radar_aligned[:, 0], radar_aligned[:, 1], c='green', label='Radar (aligned)')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('outputs/radar_lidar_align.png')
plt.show()
