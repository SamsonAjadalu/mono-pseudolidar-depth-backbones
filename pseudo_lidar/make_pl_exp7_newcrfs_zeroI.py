from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------
# Settings
# -----------------------------
SCALE_FACTOR = 1.0096
MIN_DEPTH_VALID = 1.0
MAX_DEPTH_VALID = 60.0

TARGET_POINTS = 16_384
OUTLIER_NB_NEIGHBORS = 50
OUTLIER_STD_RATIO = 2.0

KITTI_ROOT = Path("../data/KITTI")
VAL_SPLIT_FILE = KITTI_ROOT / "ImageSets" / "trainval.txt"
IMG_DIR = KITTI_ROOT / "training" / "image_2"
CALIB_DIR = KITTI_ROOT / "training" / "calib"
OUT_DIR = KITTI_ROOT / "training" / "velodyne_geometry"

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR / "newcrfs"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from networks.NewCRFDepth import NewCRFDepth  # noqa: E402

MODEL_PATH = THIS_DIR / "model_zoo" / "model_kittieigen.ckpt"
ENCODER = "large07"


def parse_calib_file(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses a KITTI calib file and returns:
      - P2 (3x4) projection matrix
      - R0 (4x4) rectification transform (homogeneous)
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
    """Removes statistical outliers and downsamples to a fixed point count."""
    if points_4d.shape[0] == 0:
        return points_4d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_4d[:, :3].astype(np.float64, copy=False))

    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO,
    )
    points_clean = points_4d[np.asarray(ind, dtype=np.int64)]

    if points_clean.shape[0] > TARGET_POINTS:
        idx = np.random.choice(points_clean.shape[0], TARGET_POINTS, replace=False)
        points_clean = points_clean[idx]

    return points_clean


def load_frame_ids() -> list[str]:
    """Loads frame ids from the split file if present, otherwise scans IMG_DIR."""
    if VAL_SPLIT_FILE.exists():
        return [ln.strip() for ln in VAL_SPLIT_FILE.read_text().splitlines() if ln.strip()]
    return sorted(p.stem for p in IMG_DIR.glob("*.png"))


def pad_to_multiple(img: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, int, int]:
    """Pads (1,C,H,W) to the next multiple of `multiple` using replicate padding."""
    _, _, h, w = img.shape
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    padding = (0, pw - w, 0, ph - h)
    return F.pad(img, padding, mode="replicate"), h, w


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NeWCRFs (encoder={ENCODER}) -> {OUT_DIR}")

    model = NewCRFDepth(version=ENCODER, inv_depth=False, max_depth=80.0)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_ids = load_frame_ids()
    print(f"Generating {len(frame_ids)} geometry-only point clouds (intensity = 0)")

    for frame_id in tqdm(frame_ids):
        img_path = IMG_DIR / f"{frame_id}.png"
        calib_path = CALIB_DIR / f"{frame_id}.txt"
        out_path = OUT_DIR / f"{frame_id}.bin"

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None or not calib_path.exists():
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        img_t, orig_h, orig_w = pad_to_multiple(img_t, multiple=32)
        img_t = img_t.to(device)

        with torch.no_grad():
            pred = model(img_t)[:, :, :orig_h, :orig_w].cpu().numpy().squeeze()

        depth_map = pred * SCALE_FACTOR

        try:
            P2, R0, Tr = parse_calib_file(calib_path)
        except Exception:
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map
        mask = (z > MIN_DEPTH_VALID) & (z < MAX_DEPTH_VALID)

        if not np.any(mask):
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        points_cam = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32, copy=False)

        points_velo = cam_to_velo(points_cam, R0, Tr).astype(np.float32, copy=False)

        # Geometry-only: keep intensity as 0
        intensity = np.zeros((points_velo.shape[0], 1), dtype=np.float32)
        points_4d = np.concatenate([points_velo, intensity], axis=1).astype(np.float32, copy=False)

        points_final = clean_and_thin_cloud(points_4d)
        points_final.tofile(str(out_path))

    print("Done.")


if __name__ == "__main__":
    main()

