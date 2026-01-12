from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# -----------------------------
# Settings
# -----------------------------
MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
MIN_DEPTH_VALID = 1.0
MAX_DEPTH_VALID = 60.0

TARGET_POINTS = 16_384
OUTLIER_NB_NEIGHBORS = 50
OUTLIER_STD_RATIO = 2.0

KITTI_ROOT = Path("../data/KITTI")
VAL_SPLIT_FILE = KITTI_ROOT / "ImageSets" / "trainval.txt"
IMG_DIR = KITTI_ROOT / "training" / "image_2"
CALIB_DIR = KITTI_ROOT / "training" / "calib"
OUT_DIR = KITTI_ROOT / "training" / "velodyne_exp2_depth_anything"


def parse_calib_file(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses a KITTI calib file and returns:
      - P2 (3x4) projection matrix
      - R0 (4x4) rectification (homogeneous)
      - Tr (4x4) velodyne->cam0 transform (homogeneous)
    """
    calib: dict[str, np.ndarray] = {}

    with calib_path.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            calib[key] = np.fromstring(val, sep=" ")

    for k in ("P2", "R0_rect", "Tr_velo_to_cam"):
        if k not in calib:
            raise KeyError(f"Missing '{k}' in {calib_path}")

    P2 = calib["P2"].reshape(3, 4)

    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = calib["R0_rect"].reshape(3, 3)

    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :] = calib["Tr_velo_to_cam"].reshape(3, 4)

    return P2, R0, Tr


def cam_to_velo(points_cam: np.ndarray, R0: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """Transforms points from rectified camera coordinates into Velodyne coordinates."""
    if points_cam.size == 0:
        return points_cam

    points_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1), dtype=points_cam.dtype)])
    T_cam_to_velo = np.linalg.inv(Tr) @ np.linalg.inv(R0)
    points_velo_h = (T_cam_to_velo @ points_h.T).T
    return points_velo_h[:, :3]


def clean_and_thin_cloud(points_4d: np.ndarray) -> np.ndarray:
    """Removes statistical outliers and downsamples to a fixed point count (optional)."""
    if points_4d.shape[0] == 0:
        return points_4d

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_4d[:, :3].astype(np.float64, copy=False))

        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NB_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO,
        )
        points_clean = points_4d[np.asarray(ind, dtype=np.int64)]
    except Exception as e:
        print(f"Warning: Open3D outlier removal failed ({e}). Using raw points.")
        points_clean = points_4d

    if points_clean.shape[0] > TARGET_POINTS:
        idx = np.random.choice(points_clean.shape[0], TARGET_POINTS, replace=False)
        points_clean = points_clean[idx]

    return points_clean


def load_frame_ids() -> list[str]:
    if not VAL_SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {VAL_SPLIT_FILE}")
    return [ln.strip() for ln in VAL_SPLIT_FILE.read_text().splitlines() if ln.strip()]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Depth Anything V2: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(device).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_ids = load_frame_ids()
    print(f"Processing {len(frame_ids)} frames -> {OUT_DIR}")

    for frame_id in tqdm(frame_ids):
        img_path = IMG_DIR / f"{frame_id}.png"
        calib_path = CALIB_DIR / f"{frame_id}.txt"
        out_path = OUT_DIR / f"{frame_id}.bin"

        if not img_path.exists() or not calib_path.exists():
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        # Load once: RGB for depth, grayscale for intensity
        image_rgb = Image.open(img_path).convert("RGB")
        w, h = image_rgb.size
        image_gray = np.array(image_rgb.convert("L"), dtype=np.uint8)

        # Depth prediction (Depth Anything v2 metric model outputs metrically-scaled depth)
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.predicted_depth  # typically [B, H', W']

        depth = F.interpolate(
            pred.unsqueeze(1),  # [B, 1, H', W']
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        depth_map = depth.squeeze(0).squeeze(0).float().cpu().numpy()

        # Camera intrinsics/extrinsics from KITTI calib
        try:
            P2, R0, Tr = parse_calib_file(calib_path)
        except Exception:
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

        # Backproject depth into camera coordinates
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map
        mask = (z > MIN_DEPTH_VALID) & (z < MAX_DEPTH_VALID)

        if not np.any(mask):
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        points_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32, copy=False)

        # Convert to Velodyne coords and attach intensity
        points_velo = cam_to_velo(points_cam, R0, Tr).astype(np.float32, copy=False)
        intensity = (image_gray[mask].astype(np.float32) / 255.0).reshape(-1, 1)

        points_4d = np.concatenate([points_velo, intensity], axis=1).astype(np.float32, copy=False)

        points_final = clean_and_thin_cloud(points_4d)
        points_final.tofile(str(out_path))

    print(f"Done. Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

