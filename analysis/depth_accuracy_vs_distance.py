from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
VAL_SPLIT_FILE = Path("../data/KITTI/ImageSets/val.txt")
GT_DIR = Path("../data/KITTI/training/label_2")
PRED_DIR = Path("../results_GT_2D_depthanything_depth")  


ERROR_THRESHOLD_M = 1.5  # meters
DIST_LIMITS_M = (20.0, 40.0, 60.0, 80.0)
VALID_CLASSES = ("Car", "Pedestrian", "Cyclist")

MATCH_MAX_PIX_DIST = 50.0  # max 2D center distance (pixels) for a match


def load_objects(file_path: Path) -> list[dict]:
    """
    Loads KITTI-format label lines and extracts:
      - class name
      - 2D bbox center (cx, cy)
      - depth z (camera coords)

    Returns empty list if file doesn't exist.
    """
    if not file_path.exists():
        return []

    objects: list[dict] = []
    with file_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls_name = parts[0]
            if cls_name not in VALID_CLASSES:
                continue

            x1, y1, x2, y2 = map(float, parts[4:8])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            z = float(parts[13])

            objects.append({"class": cls_name, "z": z, "cx": cx, "cy": cy})

    return objects


def main() -> None:
    print(f"Depth model: {PRED_DIR}")

    if not VAL_SPLIT_FILE.exists():
        print(f"Error: split file not found: {VAL_SPLIT_FILE}")
        return

    frame_ids = [ln.strip() for ln in VAL_SPLIT_FILE.read_text().splitlines() if ln.strip()]

    stats = {
        dist: {cls: {"total_gt": 0, "correct": 0, "errors": []} for cls in VALID_CLASSES}
        for dist in DIST_LIMITS_M
    }

    print(f"Calculating depth accuracy for ranges: {list(DIST_LIMITS_M)} m (tol={ERROR_THRESHOLD_M} m)")

    for frame_id in tqdm(frame_ids):
        gt_file = GT_DIR / f"{frame_id}.txt"
        pred_file = PRED_DIR / f"{frame_id}.txt"

        gt_all = load_objects(gt_file)
        pred_all = load_objects(pred_file)

        # Pre-group predictions by class once per frame
        preds_by_class = {cls: [o for o in pred_all if o["class"] == cls] for cls in VALID_CLASSES}

        for dist_limit in DIST_LIMITS_M:
            for cls_name in VALID_CLASSES:
                gts = [o for o in gt_all if o["class"] == cls_name and o["z"] <= dist_limit]
                preds = preds_by_class[cls_name]

                stats[dist_limit][cls_name]["total_gt"] += len(gts)

                used_preds: set[int] = set()

                for gt in gts:
                    best_i = -1
                    best_d = float("inf")

                    for i, pred in enumerate(preds):
                        if i in used_preds:
                            continue

                        d = float(np.hypot(gt["cx"] - pred["cx"], gt["cy"] - pred["cy"]))
                        if d <= MATCH_MAX_PIX_DIST and d < best_d:
                            best_d = d
                            best_i = i

                    if best_i == -1:
                        continue

                    used_preds.add(best_i)
                    pred = preds[best_i]

                    err = abs(gt["z"] - pred["z"])
                    stats[dist_limit][cls_name]["errors"].append(err)

                    if err <= ERROR_THRESHOLD_M:
                        stats[dist_limit][cls_name]["correct"] += 1

    for dist_limit in DIST_LIMITS_M:
        print("\n" + "=" * 80)
        print(f"DEPTH ACCURACY @ {dist_limit:.0f} m   (tolerance: {ERROR_THRESHOLD_M:.2f} m)")
        print("=" * 80)
        print(f"{'Class':<12} | {'Total GT':<10} | {'Correct':<10} | {'Acc (%)':<10} | {'Med Err':<10}")
        print("-" * 80)

        for cls_name in VALID_CLASSES:
            s = stats[dist_limit][cls_name]
            total = s["total_gt"]
            correct = s["correct"]
            errors = s["errors"]

            if total == 0:
                print(f"{cls_name:<12} | {0:<10} | {0:<10} | {'N/A':<10} | {'N/A':<10}")
                continue

            acc = 100.0 * (correct / total)
            med_err = float(np.median(errors)) if errors else 0.0
            print(f"{cls_name:<12} | {total:<10} | {correct:<10} | {acc:<10.2f} | {med_err:<10.2f} m")

        print("=" * 80)


if __name__ == "__main__":
    main()

