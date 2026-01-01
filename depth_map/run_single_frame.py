import cv2
import numpy as np

from estimators.stereo_sgbm import StereoSGBM
from estimators.monodepth2 import Monodepth2
from estimators.raft_stereo import RAFTStereoEstimator
from pipeline.metrics import compute_metrics
from pipeline.loader import load_kitti_sequence, load_frame


# ---------------------------------------------------------
# 1. Choose a single KITTI frame
# ---------------------------------------------------------
dataset = load_kitti_sequence("data/2011_09_26/2011_09_26_drive_0001_sync")
left_path, right_path, gt_path = dataset[0] # first valid aligned frame
left, right, gt = load_frame(left_path, right_path, gt_path)

# ---------------------------------------------------------
# 2. Initialize models
# ---------------------------------------------------------
mono_scale = 5.1331 # Precomputed global scale for Monodepth2
sgbm = StereoSGBM()
mono = Monodepth2(scale_factor=mono_scale)
raft = RAFTStereoEstimator()

# ---------------------------------------------------------
# 3. Run predictions
# ---------------------------------------------------------
pred_sgbm = sgbm.predict(left, right)
pred_mono = mono.predict(left)
pred_raft = raft.predict(left, right)

# Resize predictions to GT resolution for metrics
H, W = gt.shape[:2]
pred_sgbm = cv2.resize(pred_sgbm, (W, H))
pred_mono = cv2.resize(pred_mono, (W, H))
pred_raft = cv2.resize(pred_raft, (W, H))

# ---------------------------------------------------------
# 4. Compute metrics
# ---------------------------------------------------------
metrics_sgbm = compute_metrics(pred_sgbm, gt)
metrics_mono = compute_metrics(pred_mono, gt)
metrics_raft = compute_metrics(pred_raft, gt)

# ---------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------
def print_metrics(name, m):
    rmse, absrel, d1, d2, d3, _ = m
    print(f"\n{name} Metrics:")
    print(f"  RMSE:   {rmse:.3f}")
    print(f"  AbsRel: {absrel:.3f}")
    print(f"  d1:     {d1:.3f}")
    print(f"  d2:     {d2:.3f}")
    print(f"  d3:     {d3:.3f}")

print_metrics("SGBM", metrics_sgbm)
print_metrics("Monodepth2", metrics_mono)
print_metrics("RAFT-Stereo", metrics_raft)

# ---------------------------------------------------------
# 6. Visualize results (optional)
# ---------------------------------------------------------
def show_depth(title, depth):
    d = depth.copy()
    d[np.isnan(d)] = 0
    d = np.clip(d, 0, np.percentile(d, 95))
    d = (d / d.max() * 255).astype(np.uint8)
    d = cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)
    cv2.imshow(title, d)

show_depth("SGBM Depth", pred_sgbm)
show_depth("Monodepth2 Depth", pred_mono)
show_depth("RAFT-Stereo Depth", pred_raft)

cv2.imshow("Left Image", cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
