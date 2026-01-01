# Autonomous Car Camera Stack

## Depth Map Benchmarking
Robots and autonomous vehicles need to know how far things are. Depth maps help them understand the 3D world and move safely.

This module compares different depth‑estimation methods on stereo camera data. It loads image pairs, runs several models, visualizes their outputs, and creates a simple comparison video.

We use the well‑known KITTI dataset for stereo RGB images and ground‑truth depth. Three algorithms are evaluated:
- **Stereo SGBM** — a classic computer‑vision stereo matcher
- **Monodepth2** — a deep‑learning model trained on stereo images but run on a single image at inference
- **RAFT‑Stereo** — a state‑of‑the‑art deep‑learning stereo model known for high accuracy

To compare model performance, we compute standard depth metrics:
- **RMSE** — average error
- **AbsRel** — error normalized by depth
- **δ‑accuracy thresholds** — measure how many predictions fall within acceptable error bounds

The pipeline generates a video showing depth maps, error maps, and metrics for all approaches, along with a simple metrics plot.

To run the pipeline:
```
python3 run.py
```

Below is an example result from the KITTI 2011_09_26 sequence (about 100 frames).

RAFT‑Stereo performs best, Monodepth2 is a strong monocular baseline, and SGBM provides a classical reference point.