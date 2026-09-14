import cv2
import numpy as np
from scipy.optimize import least_squares
from quick_view import get_matrices_init

def solve_radar_lidar_extrinsics_svd(lidar_pts, radar_pts):
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


def solve_radar_lidar_extrinsics_ransac(lidar_pts, radar_pts, max_iters=500, threshold=0.15, min_inliers=3):
    """
    Robust radar→lidar calibration using RANSAC.
    - lidar_pts: Nx3 lidar reflector centers
    - radar_pts: Nx3 radar reflector centers
    - max_iters: number of RANSAC iterations
    - threshold: inlier distance threshold (meters)
    - min_inliers: minimum number of inliers needed to accept a model
    """

    # Use only XY (radar has no elevation)
    L = lidar_pts[:, :2]
    R = radar_pts[:, :2]
    N = len(L)

    best_inliers = []
    best_R2 = None
    best_t2 = None

    # ------------------------------------------------------------
    # RANSAC LOOP
    # ------------------------------------------------------------
    for _ in range(max_iters):

        # 1. Randomly pick 2 correspondences (minimum for 2D rotation)
        idx = np.random.choice(N, 2, replace=False)
        L_sample = L[idx]
        R_sample = R[idx]

        # 2. Compute rotation using SVD (same as your solver)
        cL = L_sample.mean(axis=0)
        cR = R_sample.mean(axis=0)
        Lc = L_sample - cL
        Rc = R_sample - cR

        H = Rc.T @ Lc
        U, _, Vt = np.linalg.svd(H)
        R2 = U @ Vt

        if np.linalg.det(R2) < 0:
            U[:, -1] *= -1
            R2 = U @ Vt

        # 3. Compute translation
        t2 = cL - R2 @ cR

        # 4. Compute inliers
        R_all = (R2 @ R.T).T + t2
        errors = np.linalg.norm(L - R_all, axis=1)
        inliers = np.where(errors < threshold)[0]

        # 5. Keep best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R2 = R2
            best_t2 = t2

    # ------------------------------------------------------------
    # Recompute final rotation/translation using all inliers
    # ------------------------------------------------------------
    if len(best_inliers) < min_inliers:
        raise RuntimeError("RANSAC failed: not enough inliers")

    L_in = L[best_inliers]
    R_in = R[best_inliers]

    cL = L_in.mean(axis=0)
    cR = R_in.mean(axis=0)
    Lc = L_in - cL
    Rc = R_in - cR

    H = Rc.T @ Lc
    U, _, Vt = np.linalg.svd(H)
    R2 = U @ Vt

    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt

    t2 = cL - R2 @ cR

    # ------------------------------------------------------------
    # Convert to 3D rotation + translation
    # ------------------------------------------------------------
    R3 = np.eye(3)
    R3[:2, :2] = R2
    t3 = np.array([t2[0], t2[1], 0.0])

    yaw = np.arctan2(R2[1,0], R2[0,0])

    return yaw, R3, t3, best_inliers


def refine_radar_lidar_extrinsics_nonlinear(lidar_pts, radar_pts, yaw_init, R_init, t_init):
    """
    Nonlinear least-squares refinement of radar→lidar extrinsics.
    Parameters to optimize: yaw, tx, ty, tz.
    lidar_pts: Nx3
    radar_pts: Nx3
    yaw_init: initial yaw (float, radians)
    R_init: 3x3 initial rotation (from SVD/RANSAC)
    t_init: 3-vector initial translation
    """

    L = lidar_pts  # (N, 3)
    R = radar_pts  # (N, 3)

    def residuals(params):
        yaw, tx, ty, tz = params

        # Build rotation from yaw only (around Z)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R_yaw = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0]
        ])

        t = np.array([tx, ty, tz])

        # Transform radar points
        R_transformed = (R_yaw @ R.T).T + t

        # Residuals: lidar - transformed radar
        res = (L - R_transformed).reshape(-1)
        return res

    # Initial parameter vector
    x0 = np.array([yaw_init, t_init[0], t_init[1], t_init[2]])

    # Nonlinear least-squares
    result = least_squares(residuals, x0, method="lm")

    yaw_opt, tx_opt, ty_opt, tz_opt = result.x

    cy = np.cos(yaw_opt)
    sy = np.sin(yaw_opt)
    R_opt = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])
    t_opt = np.array([tx_opt, ty_opt, tz_opt])

    return yaw_opt, R_opt, t_opt, result


def combine_R_t(R1, t1, R2, t2):
    """
    Combine two transforms:
        T1 = (R1, t1)
        T2 = (R2, t2)
    Result:
        T_combined = T2 * T1
    """
    # Combined rotation
    R_combined = R2 @ R1

    # Combined translation
    t_combined = R2 @ t1 + t2

    return R_combined, t_combined


if __name__ == "__main__":

    # Load Corner reflector points from Lidar and Radar
    lidar_means = np.load("outputs/lidar_means.npy")
    radar_means = np.load("outputs/radar_means.npy")

    # 1. Get solution (SVD or RANSAC)
    yaw, R, t = solve_radar_lidar_extrinsics_svd(lidar_means, radar_means)
    # yaw, R, t, inliers = solve_radar_lidar_extrinsics_ransac(lidar_means, radar_means)

    # 2. Refine with nonlinear optimization
    yaw_opt, R_opt, t_opt, result = refine_radar_lidar_extrinsics_nonlinear(
        lidar_means, radar_means, yaw, R, t)

    # The calibrated R and T values are based on the transformed values of radar data
    # Therefore we need to combine the initial transform to the calibrated Transform
    R_init, t_init = get_matrices_init("FL") # getting intial transform values from quick_view.py
    R_output, t_output = combine_R_t(R_init, t_init, R_opt, t_opt) # combinte two transforms

    # Stored calibrated parameters to numpy files for analysis
    # SVD
    np.save("outputs/svd_R.npy", R)
    np.save("outputs/svd_t.npy", t)
    # Nonlinear optimization
    np.save("outputs/nonlinear_R.npy", R_opt)
    np.save("outputs/nonlinear_t.npy", t_opt)

    # Stored final output parameters to numpy files
    np.save("outputs/R_radar_lidar.npy", R_output)
    np.save("outputs/t_radar_lidar.npy", t_output)

    # Print the calibrated parameters
    print("R:\n", R_output)
    print("t:", t_output)


