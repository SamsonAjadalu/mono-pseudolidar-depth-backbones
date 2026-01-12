# mono-pseudolidar-depth-backbones

Utilities to generate **monocular pseudo-LiDAR** point clouds on **KITTI** using different depth backbones (NeWCRFs, Depth Anything v2 Metric), plus scripts for evaluation/analysis.

## Demo

[![Demo](assets/demo.gif)](https://www.youtube.com/watch?v=xuTg-L_vzZQ)


Pipeline (high level):

**KITTI image → depth model → backproject to 3D (cam) → cam→velodyne transform → save KITTI `.bin`**  
Then train/evaluate **PointRCNN (OpenPCDet)** on the generated `.bin` folders.

> Many scripts use relative paths like `../data/...` and `../work/...`.  
> Run scripts from inside their folder (e.g. `cd pseudo_lidar`) to avoid path issues.

---

## Repository layout

```
analysis/          # analysis scripts
configs/           # OpenPCDet PointRCNN configs
data/KITTI/        # KITTI dataset 
env/               # conda environment files
extras/            # helper / evaluation scripts
pseudo_lidar/      # pseudo-LiDAR generators (experiments)
results/           # outputs 
work/              # intermediate artifacts
```

---

## Setup

### Conda environment

Environment YAMLs are in `env/`. Create/activate one, for example:

```bash
conda env create -f env/environment_ss3d.yml
conda activate ss3d
```

### KITTI layout

Place KITTI under `data/KITTI`:

```
data/KITTI/
  ImageSets/
    trainval.txt
    val.txt
  training/
    image_2/
    calib/
    label_2/
```

### NeWCRFs requirements

NeWCRFs-based generators expect:

* `pseudo_lidar/newcrfs/` (NeWCRFs code)
* `pseudo_lidar/model_zoo/model_kittieigen.ckpt` (checkpoint)

### Mask JSON (Exp4 / Exp5)

Exp4 + Exp5 read:

```
work/mask_rcnn_safe.json
```

Expected structure:

```json
{
  "000123": { "masks": [RLE_1, RLE_2], "scores": [0.91, 0.77] },
  "000124": { "masks": [...], "scores": [...] }
}
```

* `masks` are COCO RLE masks (pycocotools format)
* `scores` are detection confidences
* scripts use a threshold of `0.3`

---

## Generate pseudo-LiDAR point clouds

All generators output KITTI-style point clouds:

* `.bin` float32 rows: `(x, y, z, intensity)` in **Velodyne coordinates**

Run generators from `pseudo_lidar/`:

```bash
cd pseudo_lidar
```

### Exp2 — NeWCRFs + grayscale intensity

```bash
python make_pl_exp2_newcrfs_gray.py
```

Output:

* `data/KITTI/training/velodyne_exp2/`

Notes:

* depth from NeWCRFs
* intensity = grayscale (`gray / 255`)
* statistical outlier removal + downsample to ~16,384 points

### Exp7 — NeWCRFs + intensity=0 (geometry only)

```bash
python make_pl_exp7_newcrfs_zeroI.py
```

Output:

* `data/KITTI/training/velodyne_geometry/`

Notes:

* same as Exp2, but intensity is always `0`
* downsample to ~16,384 points

### Exp4 — NeWCRFs + soft mask confidence intensity (full scene)

```bash
python make_pl_exp4_newcrfs_maskconf.py
```

Output:

* `data/KITTI/training/velodyne_soft_mask/`

Notes:

* keeps background (full scene)
* intensity = per-pixel confidence built from `work/mask_rcnn_safe.json`
* samples down to keep file sizes manageable (~16,384 points)

### Exp5 — NeWCRFs + mask-guided sampling (up to 40k points)

```bash
python make_pl_exp5_newcrfs_maskguided.py
```

Output:

* `data/KITTI/training/velodyne_smart_gray/`

Notes:

* intensity = grayscale
* mask JSON is used only to bias sampling (car points kept first)
* targets up to 40,000 points per frame

### Depth Anything v2 Metric — grayscale intensity

```bash
python make_pl_exp2_da_metric.py
```

Output:

* `data/KITTI/training/velodyne_exp2_depth_anything/`

Notes:

* HF model: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf`
* depth is resized to the original image resolution
* outlier removal + downsample to ~16,384 points

### Exp0 — Oracle GT-2D frustum baseline

```bash
python make_pl_exp0_oracle_gt2d.py
```

Notes:

* uses KITTI GT 2D boxes (`data/KITTI/training/label_2`)
* reads points from `work/pseudo_lidar_base/*.bin` (cam2 XYZ)
* clusters points (DBSCAN) and fits a box
* writes KITTI-format detection results to:

  * `work/results_GT_2D_base_depth/`

---

## Training / evaluation with OpenPCDet (PointRCNN)

Install OpenPCDet and use the configs in `configs/`.

Configs:

* `configs/pointrcnn.yaml` (base experiments)
* `configs/pointrcnn_kitti_exp5_40k.yaml` (40k points)

Example (from your OpenPCDet checkout):

```bash
python -m pcdet.tools.train --cfg_file /path/to/mono-pseudolidar-depth-backbones/configs/pointrcnn.yaml
```

### Selecting which pseudo-LiDAR folder OpenPCDet reads

OpenPCDet usually reads from `data/KITTI/training/velodyne/`. For each experiment, either:

1. edit the dataset path in the OpenPCDet config, **or**
2. symlink the experiment output folder to `velodyne`.

Symlink example (Exp2):

```bash
ln -sfn /path/to/mono-pseudolidar-depth-backbones/data/KITTI/training/velodyne_exp2 \
       /path/to/mono-pseudolidar-depth-backbones/data/KITTI/training/velodyne
```

---

## Analysis: depth accuracy vs distance

Run from `analysis/`:

```bash
cd analysis
python depth_accuracy_vs_distance.py
```

This script:

* loads GT objects from `data/KITTI/training/label_2/`
* loads predicted results from a folder (set `PRED_DIR` at the top)
* matches GT↔pred by 2D center distance
* reports depth accuracy using `|z_pred - z_gt| <= 1.5m` across distance buckets

---

## What not to commit

Do not commit datasets or generated artifacts:

* `data/KITTI/`
* `work/`
* `results/`
* `*.bin`, `*.ckpt`, `*.pth`, `*.pkl`

---
