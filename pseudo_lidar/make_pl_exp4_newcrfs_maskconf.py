from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------
# Settings
# -----------------------------
SCALE_FACTOR = 1.0096
MIN_DEPTH_VALID = 1.0
MAX_DEPTH_VALID = 60.0

CAR_SCORE_THRESHOLD = 0.3
MAX_POINTS_PER_FRAME = 16_384  # keep file sizes reasonable

KITTI_ROOT = Path("../data/KITTI")
VAL_SPLIT_FILE = KITTI_ROOT / "ImageSets" / "trainval.txt"
IMG_DIR = KITTI_ROOT / "training" / "image_2"
CALIB_DIR = KITTI_ROOT / "training" / "calib"

OUT_DIR = KITTI_ROOT / "training" / "velodyne_soft_mask"
JSON_PATH = Path("../work/mask_rcnn_safe.json")

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR / "newcrfs"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from networks.NewCRFDepth import NewCRFDepth  # noqa: E402

MODEL_PATH = THIS_DIR / "model_zoo" / "model_kittieigen.ckpt"
ENCODER = "large07"


def parse_calib_file(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - P2 (3x4) projection matrix
      - R0_rect (3x3) rectification rotation
      - Tr_velo_to_cam (3x4) velodyne->cam0 transform
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
    R0_rect = calib["R0_rect"].reshape(3, 3)
    Tr_velo_to_cam = calib["Tr_velo_to_cam"].reshape(3, 4)
    return P2, R0_rect, Tr_velo_to_cam


def cam_to_velo(points_cam: np.ndarray, R0_rect: np.ndarray, Tr_velo_to_cam: np.ndarray) -> np.ndarray:
    """Transforms points from rectified camera coordinates into Velodyne coordinates."""
    if points_cam.size == 0:
        return points_cam

    points_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1), dtype=points_cam.dtype)])

    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = R0_rect

    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :] = Tr_velo_to_cam

    T_cam_to_velo = np.linalg.inv(Tr) @ np.linalg.inv(R0)
    points_velo_h = (T_cam_to_velo @ points_h.T).T
    return points_velo_h[:, :3]


def pad_to_multiple(img: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, int, int]:
    """Pads (1,C,H,W) to the next multiple of `multiple` using replicate padding."""
    _, _, h, w = img.shape
    ph = ((h - 1) // multiple + 1) * multiple
    pw = ((w - 1) // multiple + 1) * multiple
    padding = (0, pw - w, 0, ph - h)
    return F.pad(img, padding, mode="replicate"), h, w


def load_frame_ids() -> list[str]:
    """Loads split ids if available, otherwise scans the image folder."""
    if VAL_SPLIT_FILE.exists():
        return [ln.strip() for ln in VAL_SPLIT_FILE.read_text().splitlines() if ln.strip()]
    return sorted(p.stem for p in IMG_DIR.glob("*.png"))


def build_confidence_mask(frame_id: str, mask_data: dict, h: int, w: int) -> np.ndarray:
    """
    Creates a per-pixel confidence map using Mask R-CNN detections.
    If masks overlap, the maximum confidence is kept.
    """
    conf = np.zeros((h, w), dtype=np.float32)

    frame_obj = mask_data.get(frame_id)
    if not frame_obj:
        return conf

    masks_rle = frame_obj.get("masks", []) or []
    scores = frame_obj.get("scores", []) or []
    n = min(len(masks_rle), len(scores))

    for i in range(n):
        score = float(scores[i])
        if score <= CAR_SCORE_THRESHOLD:
            continue

        rle = masks_rle[i]
        if isinstance(rle.get("counts"), str):
            rle = dict(rle)
            rle["counts"] = rle["counts"].encode("utf-8")

        m = mask_util.decode(rle)
        if m is None:
            continue

        if m.ndim == 3:
            m = m[:, :, 0]

        if m.shape[:2] != (h, w):
            continue

        conf = np.maximum(conf, m.astype(np.float32) * score)

    return conf


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Mask JSON not found: {JSON_PATH}")

    print(f"Loading masks: {JSON_PATH}")
    mask_data = json.loads(JSON_PATH.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NeWCRFs (encoder={ENCODER}) on {device}")

    model = NewCRFDepth(version=ENCODER, inv_depth=False, max_depth=80.0)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(str(MODEL_PATH), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_ids = load_frame_ids()
    print(f"Generating soft-masked pseudo-LiDAR -> {OUT_DIR}")

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

        input_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        input_t, orig_h, orig_w = pad_to_multiple(input_t, multiple=32)
        input_t = input_t.to(device)

        with torch.no_grad():
            depth = model(input_t)[:, :, :orig_h, :orig_w].cpu().numpy().squeeze()

        depth_map = depth * SCALE_FACTOR

        # Confidence mask is used as intensity (background stays at 0)
        confidence_mask = build_confidence_mask(frame_id, mask_data, h, w)

        # Backproject all valid depth points (mask only affects intensity, not selection)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map
        valid = (z > MIN_DEPTH_VALID) & (z < MAX_DEPTH_VALID)

        if not np.any(valid):
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        try:
            P2, R0_rect, Tr_velo_to_cam = parse_calib_file(calib_path)
        except Exception:
            np.zeros((0, 4), dtype=np.float32).tofile(str(out_path))
            continue

        fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        points_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32, copy=False)
        points_velo = cam_to_velo(points_cam, R0_rect, Tr_velo_to_cam).astype(np.float32, copy=False)

        intensities = confidence_mask[valid].reshape(-1, 1).astype(np.float32, copy=False)
        points_4d = np.hstack([points_velo, intensities]).astype(np.float32, copy=False)

        if points_4d.shape[0] > MAX_POINTS_PER_FRAME:
            idx = np.random.choice(points_4d.shape[0], MAX_POINTS_PER_FRAME, replace=False)
            points_4d = points_4d[idx]

        points_4d.tofile(str(out_path))

    print("Done.")


if __name__ == "__main__":
    main()

