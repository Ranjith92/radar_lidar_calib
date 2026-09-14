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
# SVD
R_svd = np.load("outputs/svd_R.npy")[:2, :2]    # 2×2 rotation
t_svd = np.load("outputs/svd_t.npy")[:2]        # XY translation
# Nonlinear Optimization
R_opt = np.load("outputs/nonlinear_R.npy")[:2, :2]    # 2×2 rotation
t_opt = np.load("outputs/nonlinear_t.npy")[:2]        # XY translation

# ------------------------------------------------------------
# Apply alignment: radar_aligned = R * radar + t
# ------------------------------------------------------------
radar_aligned = (R_svd @ radar.T).T + t_svd
radar_aligned_opt = (R_opt @ radar.T).T + t_opt

# ------------------------------------------------------------
# Plot everything
# ------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.title("Radar–Lidar Alignment Visualization")

# Lidar points (blue)
plt.scatter(lidar[:, 0], lidar[:, 1], c='blue', label='Lidar points')

# Radar points before alignment (red)
plt.scatter(radar[:, 0], radar[:, 1], c='red', label='Radar (raw)')

# Radar points after alignment (orange)
plt.scatter(radar_aligned[:, 0], radar_aligned[:, 1], c='orange', label='Radar (aligned_svd)')

# Radar points after alignment (green)
plt.scatter(radar_aligned_opt[:, 0], radar_aligned_opt[:, 1], c='green', label='Radar (aligned_opt)')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('outputs/radar_lidar_align.png')
plt.show()
