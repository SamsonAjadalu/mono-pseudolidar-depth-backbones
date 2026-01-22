# CARLA Simulation Integration (Work In Progress)

This module implements the synthetic data generation pipeline to validate the Pseudo-LiDAR models in a controlled 3D environment.

## Goal
To stream real-time sensor data from the CARLA Simulator into the `pseudo_lidar/` generation pipeline, enabling closed-loop testing of 3D object detection without requiring real-world KITTI data.

## Current Status
- [x] **Ego Vehicle Control:** Basic autopilot navigation spawn.
- [x] **Sensor Setup:** RGB Camera and Depth Camera configuration matching KITTI intrinsics.
- [x] **Data Acquisition:** Frame capture and local logging.
- [ ] **Coordinate Transformation:** Converting CARLA (Left-Handed) to KITTI (Right-Handed) coordinates.
- [ ] **Real-time Inference:** Feeding frames directly to `Depth` backbone.

## Usage (Preview)
Ensure the CARLA server is running (localhost:2000), then run:
```bash
python carla_recorder.py --output_dir captured_data
```

Note: This integration is under active development. The goal is to produce "Fake KITTI" datasets for robust model evaluation.

