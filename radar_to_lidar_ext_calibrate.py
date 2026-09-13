import numpy as np
import cv2

def solve_radar_lidar_extrinsics(lidar_pts, radar_pts):
    """
    Estimate radar → lidar extrinsic calibration.
    Radar has no elevation, so we solve only for:
        - yaw rotation (2D rotation in XY plane)
        - translation (tx, ty, tz)
    """

    # Use only XY coordinate. Ignored unreliable Z.
    L = lidar_pts[:, :2]
    R = radar_pts[:, :2]

    # ----------------------------------------------
    # Compute centroids of both point sets
    # ----------------------------------------------

    # Centering removes translation and isolates rotation.
    cL = L.mean(axis=0)
    cR = R.mean(axis=0)
    # Subtract centroids → centered point clouds
    Lc = L - cL
    Rc = R - cR

    # ----------------------------------------------
    # Compute cross‑covariance matrix
    # ----------------------------------------------
    H = Rc.T @ Lc

    # ----------------------------------------------
    # Solve optimal 2D rotation using SVD
    # ----------------------------------------------
    U, _, Vt = np.linalg.svd(H)
    R2 = U @ Vt
    # Ensuring rotation matrix has det = +1
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt

    # ----------------------------------------------
    # Compute translation in XY plane
    # ----------------------------------------------
    yaw = np.arctan2(R2[1,0], R2[0,0])
    t2 = cL - R2 @ cR

    # ----------------------------------------------
    # Rotation and translation into 3D
    # ----------------------------------------------
    R3 = np.eye(3)
    R3[:2,:2] = R2
    t3 = np.array([t2[0], t2[1], 0.0])

    return yaw, R3, t3


if __name__ == "__main__":

    # Load Corner reflector points from Lidar and Radar
    lidar_means = np.load("outputs/lidar_means.npy")
    radar_means = np.load("outputs/radar_means.npy")

    # Extrinsic calibration: Radar -> Lidar
    yaw, R_rl, t_rl = solve_radar_lidar_extrinsics(lidar_means, radar_means)

    # Stored calibrated parameters to numpy files
    np.save("outputs/R_radar_lidar.npy", R_rl)
    np.save("outputs/t_radar_lidar.npy", t_rl)

    # Print the calibrated parameters
    print("Yaw (deg):", np.degrees(yaw))
    print("R:\n", R_rl)
    print("t:", t_rl)
