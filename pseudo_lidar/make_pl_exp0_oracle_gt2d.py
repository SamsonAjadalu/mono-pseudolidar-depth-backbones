from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

"""
Fuses base-depth pseudo-LiDAR points with KITTI ground-truth 2D boxes.

Inputs:
  - work/pseudo_lidar_base/*.bin           (points in cam2 coords, float32 XYZ)
  - data/KITTI/training/label_2/*.txt      (GT 2D boxes)

Output:
  - results_GT_2D_base_depth/*.txt         (KITTI detection-format lines)

Run:
  python3 make_pl_exp0_oracle_gt2d.py
"""

# -----------------------------
# Paths / settings
# -----------------------------
VAL_SPLIT_FILE = Path("../data/KITTI/ImageSets/val.txt")
CALIB_DIR = Path("../data/KITTI/training/calib")
GT_LABEL_DIR = Path("../data/KITTI/training/label_2")

PSEUDO_LIDAR_DIR = Path("../work/pseudo_lidar_base")
RESULTS_DIR = Path("../work/results_GT_2D_base_depth")  # or ../results if you prefer


DBSCAN_EPS = 0.20
DBSCAN_MIN_POINTS = 10
MIN_POINTS_FOR_BOX = 20

KITTI_CLASSES = ("Car", "Pedestrian", "Cyclist")
CLASS_PRIORS = {
    "Car": {"h": 1.52, "w": 1.63, "l": 3.88},
    "Pedestrian": {"h": 1.73, "w": 0.60, "l": 0.80},
    "Cyclist": {"h": 1.73, "w": 0.60, "l": 1.76},
}


def load_calib_K_cam2(calib_path: Path) -> np.ndarray:
    """Loads K (3x3) for camera 2 from a KITTI calib file (P2)."""
    data: dict[str, np.ndarray] = {}

    with calib_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            data[key] = np.fromstring(val, sep=" ")

    if "P2" not in data:
        raise KeyError(f"Missing P2 in {calib_path}")

    P2 = data["P2"].reshape(3, 4)
    return P2[:3, :3]


def load_gt_2d_boxes(label_path: Path) -> list[dict]:
    """
    Reads KITTI label_2 format and returns a list of:
      {"cls": str, "bbox": [x1,y1,x2,y2]}
    """
    if not label_path.exists():
        return []

    boxes: list[dict] = []
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            cls_name = parts[0]
            if cls_name not in KITTI_CLASSES:
                continue

            x1, y1, x2, y2 = map(float, parts[4:8])
            boxes.append({"cls": cls_name, "bbox": [x1, y1, x2, y2]})

    return boxes


def project_cam2_to_img(K: np.ndarray, pts_cam2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projects cam2 XYZ points to image plane using K."""
    uvw = (K @ pts_cam2.T).T
    z = uvw[:, 2]
    u = uvw[:, 0] / (z + 1e-6)
    v = uvw[:, 1] / (z + 1e-6)
    return u, v


def robust_dims_yaw(pts_cam2: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Estimates (h,w,l,rotation_y) from points using robust percentiles + PCA in XZ.
    Returns None if there aren't enough points.
    """
    if pts_cam2.shape[0] < 10:
        return None

    y = pts_cam2[:, 1]
    h = float(np.percentile(y, 95) - np.percentile(y, 5))
    h = max(h, 0.1)

    xz = pts_cam2[:, [0, 2]]
    xz = xz - np.median(xz, axis=0, keepdims=True)

    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    axis_long = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)

    proj_long = xz @ axis_long
    l = float(np.percentile(proj_long, 95) - np.percentile(proj_long, 5))
    l = max(l, 0.1)

    axis_short = np.array([-axis_long[1], axis_long[0]])
    proj_short = xz @ axis_short
    w = float(np.percentile(proj_short, 95) - np.percentile(proj_short, 5))
    w = max(w, 0.1)

    rotation_y = float(np.arctan2(axis_long[0], axis_long[1]))
    return h, w, l, rotation_y


def pick_nearest_cluster(pts: np.ndarray) -> np.ndarray | None:
    """Runs DBSCAN and returns the cluster whose median Z is closest (smallest)."""
    if pts.shape[0] == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))

    labels = np.array(
        pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False)
    )
    if labels.size == 0:
        return None

    best_lab = -1
    best_med_z = float("inf")

    for lab in np.unique(labels):
        if lab == -1:
            continue
        idxs = np.where(labels == lab)[0]
        cluster_pts = np.asarray(pcd.select_by_index(idxs).points)
        med_z = float(np.median(cluster_pts[:, 2]))
        if med_z < best_med_z:
            best_med_z = med_z
            best_lab = int(lab)

    if best_lab == -1:
        return None

    idxs = np.where(labels == best_lab)[0]
    return np.asarray(pcd.select_by_index(idxs).points)


def main() -> None:
    if not VAL_SPLIT_FILE.exists():
        raise FileNotFoundError(f"Missing split file: {VAL_SPLIT_FILE}")

    frame_ids = [ln.strip() for ln in VAL_SPLIT_FILE.read_text().splitlines() if ln.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating results for {len(frame_ids)} frames (GT 2D boxes) -> {RESULTS_DIR}")

    for frame_id in tqdm(frame_ids):
        bin_path = PSEUDO_LIDAR_DIR / f"{frame_id}.bin"
        calib_path = CALIB_DIR / f"{frame_id}.txt"
        label_path = GT_LABEL_DIR / f"{frame_id}.txt"
        out_path = RESULTS_DIR / f"{frame_id}.txt"

        if not bin_path.exists() or not calib_path.exists():
            out_path.write_text("")  # keep output set consistent
            continue

        pts = np.fromfile(str(bin_path), dtype=np.float32)
        if pts.size == 0:
            out_path.write_text("")
            continue

        pts_cam2 = pts.reshape(-1, 3)

        try:
            K = load_calib_K_cam2(calib_path)
        except Exception:
            out_path.write_text("")
            continue

        u, v = project_cam2_to_img(K, pts_cam2)

        gt_objects = load_gt_2d_boxes(label_path)
        if not gt_objects:
            out_path.write_text("")
            continue

        result_lines: list[str] = []

        for obj in gt_objects:
            class_name = obj["cls"]
            x1, y1, x2, y2 = obj["bbox"]

            in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            pts_in_box = pts_cam2[in_box]

            if pts_in_box.shape[0] < MIN_POINTS_FOR_BOX:
                continue

            cluster_pts = pick_nearest_cluster(pts_in_box)
            if cluster_pts is None or cluster_pts.shape[0] < MIN_POINTS_FOR_BOX:
                continue

            xc, yc, zc = np.median(cluster_pts, axis=0).astype(float)

            dims = robust_dims_yaw(cluster_pts)
            if dims is None:
                prior = CLASS_PRIORS[class_name]
                h, w_box, l = prior["h"], prior["w"], prior["l"]
                rotation_y = 0.0
            else:
                h, w_box, l, rotation_y = dims

            alpha = float(rotation_y - np.arctan2(xc, zc))

            # Score is fixed to 1.0 since the 2D boxes are GT.
            line = (
                f"{class_name} -1 -1 {alpha:.2f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                f"{h:.2f} {w_box:.2f} {l:.2f} "
                f"{xc:.2f} {yc:.2f} {zc:.2f} "
                f"{rotation_y:.2f} 1.00"
            )
            result_lines.append(line)

        out_path.write_text("\n".join(result_lines))

    print("Done.")


if __name__ == "__main__":
    main()

