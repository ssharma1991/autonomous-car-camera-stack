import cv2
import numpy as np
from estimators.monodepth2 import Monodepth2
from pipeline.loader import load_kitti_sequence, load_frame
from pipeline.metrics import compute_metrics


# ---------------------------------------------------------
# Load aligned KITTI sequence
# ---------------------------------------------------------
dataset = load_kitti_sequence("data/2011_09_26/2011_09_26_drive_0001_sync")
mono = Monodepth2()
ratios = []

# ---------------------------------------------------------
# Compute ratios incrementally (no large memory usage)
# ---------------------------------------------------------
for left_path, right_path, gt_path in dataset:
    left, right, gt = load_frame(left_path, right_path, gt_path)

    pred = mono.predict(left)
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    ratio = mono.compute_frame_scale(pred, gt)
    if ratio is not None:
        ratios.append(ratio)

# ---------------------------------------------------------
# Compute global scale
# ---------------------------------------------------------
scale = np.median(ratios) if ratios else 1.0
print(f"Global scale factor: {scale:.4f}")

# ---------------------------------------------------------
# Evaluate on a test frame
# ---------------------------------------------------------
test_left, test_right, test_gt = dataset[0]
left, right, gt = load_frame(test_left, test_right, test_gt)

pred = mono.predict(left)
pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
pred_scaled = pred * scale

rmse, absrel, d1, d2, d3, _ = compute_metrics(pred_scaled, gt)

print("\nMonodepth2 (Global Scale) Metrics:")
print(f"  RMSE:   {rmse:.3f}")
print(f"  AbsRel: {absrel:.3f}")
print(f"  d1:     {d1:.3f}")
print(f"  d2:     {d2:.3f}")
print(f"  d3:     {d3:.3f}")
