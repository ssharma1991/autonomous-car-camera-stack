import cv2
import numpy as np

from estimators.stereo_sgbm import StereoSGBM
from estimators.monodepth2 import Monodepth2
from estimators.raft_stereo import RAFTStereoEstimator
from pipeline.loader import load_kitti_sequence, load_frame
from pipeline.metrics import compute_metrics
from pipeline.plot import save_metric_plot
from pipeline.video import VideoWriter

dataset = load_kitti_sequence("data/2011_09_26/2011_09_26_drive_0001_sync")

# Register models
mono_scale = 5.1033 # Precomputed global scale for Monodepth2
models = {
    "SGBM": StereoSGBM(),
    "Monodepth2": Monodepth2(scale_factor=mono_scale),
    "RAFT-Stereo": RAFTStereoEstimator()
}

# Prepare video writer
sample_left = cv2.imread(dataset[0][0])
h, w = sample_left.shape[:2]
writer = VideoWriter("depth_comparison.mp4", frame_size=(w*2, h*5))

# Metric curves
curves = {f"{name}_{m}": [] for name in models for m in ["rmse","absrel","d1","d2","d3"]}

for left_path, right_path, gt_path in dataset:
    print(f"Processing frame: {left_path}")

    # Load frame (RGB left, RGB right, float32 GT)
    left, right, gt = load_frame(left_path, right_path, gt_path)

    # Predictions
    preds = {
        name: model.predict(left, right) if "SGBM" in name or "RAFT" in name
              else model.predict(left)
        for name, model in models.items()
    }

    # Resize preds to GT resolution
    H, W = gt.shape[:2]
    preds = {k: cv2.resize(v, (W, H)) for k, v in preds.items()}

    # Metrics + errors
    metrics = {}
    errors  = {}
    for name, pred in preds.items():
        rmse, absrel, d1, d2, d3, err = compute_metrics(pred, gt)
        metrics[name] = (rmse, absrel, d1, d2, d3)
        errors[name]  = err

        curves[f"{name}_rmse"].append(rmse)
        curves[f"{name}_absrel"].append(absrel)
        curves[f"{name}_d1"].append(d1)
        curves[f"{name}_d2"].append(d2)
        curves[f"{name}_d3"].append(d3)

    # Add frame to video
    writer.add_frame(left, right, preds, errors, metrics, gt)

writer.close()
writer.finalize_video("depth_comparison.mp4")
save_metric_plot(curves, "metrics.png")