import numpy as np
import pandas as pd
import cv2
import math

# ----------------------------------------------------
# Load CSV points
# ----------------------------------------------------
def load_points(path):
    df = pd.read_csv(path)
    return df[['x','y','z']].to_numpy()


# ----------------------------------------------------
# Load camera intrinsics + lidar->camera extrinsics
# ----------------------------------------------------
def load_camera_calibration():
    cam = np.load("cfl_calibration.npz")
    K = cam["camera_matrix"]
    dist = cam["dist_coeffs"]

    lc = np.load("lidar2cfl_new_all.npz")
    R_lc = lc["R"]
    t_lc = lc["t"]

    return K, dist, R_lc, t_lc


# ----------------------------------------------------
# Compose transforms
# ----------------------------------------------------
def compose_transform(R_ab, t_ab, R_bc, t_bc):
    R_ac = R_bc @ R_ab
    t_ac = R_bc @ t_ab + t_bc
    return R_ac, t_ac


# ----------------------------------------------------
# Project 3D points to camera image
# ----------------------------------------------------
def project_to_image(points_cam, K, dist):
    proj, _ = cv2.projectPoints(
        points_cam,
        np.zeros((3,1)),
        np.zeros((3,1)),
        K,
        dist
    )
    return proj.reshape(-1, 2)


# ----------------------------------------------------
# Draw lidar and Radar projection points in image
# ----------------------------------------------------
def draw_lidar_radar_pts(img, lidar_proj, radar_proj):

    # Draw lidar (blue)
    for u, v in lidar_proj:
        if math.isnan(u) or math.isnan(v):
            continue
        u, v = int(u), int(v)
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 3, (255, 0, 0), -1)

    # Draw radar (red)
    for u, v in radar_proj:
        if math.isnan(u) or math.isnan(v):
            continue
        u, v = int(u), int(v)
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (u, v), 4, (0, 0, 255), -1)

    return img



if __name__ == "__main__":

    # Load camera image
    img = cv2.imread("test_data/16.jpg")

    # Load lidar and radar points
    lidar_pts = load_points("test_data/16.csv")
    radar_pts = load_points("test_data/16_rad.csv")

    # Load radar->lidar calibration (your solved extrinsics)
    R_rl = np.load("outputs/R_radar_lidar.npy")
    t_rl = np.load("outputs/t_radar_lidar.npy")

    # Load camera intrinsics + lidar->camera extrinsics
    K, dist, R_lc, t_lc = load_camera_calibration()

    # Compose radar->camera
    R_rc, t_rc = compose_transform(R_rl, t_rl, R_lc, t_lc)

    # Transform lidar->camera
    lidar_cam = (R_lc @ lidar_pts.T).T + t_lc

    # Transform radar->camera
    radar_cam = (R_rc @ radar_pts.T).T + t_rc

    # Project both onto camera image
    lidar_proj = project_to_image(lidar_cam, K, dist)
    radar_proj = project_to_image(radar_cam, K, dist)

    # projection of lidar and radar points in image
    proj_img = draw_lidar_radar_pts(img, lidar_proj, radar_proj)

    # Save image
    cv2.imwrite("outputs/lidar_radar_overlay.png", proj_img)
    # Display image
    cv2.namedWindow("Lidar + Radar Overlay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lidar + Radar Overlay", 960, 540)
    cv2.imshow("Lidar + Radar Overlay", proj_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


