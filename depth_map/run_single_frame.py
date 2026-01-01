import cv2
import numpy as np

from estimators.stereo_sgbm import StereoSGBM
from estimators.monodepth2 import Monodepth2
from estimators.raft_stereo import RAFTStereoEstimator
from pipeline.metrics import compute_metrics
from pipeline.loader import load_frame

def median_scale(pred, gt):
    """Scale monocular depth prediction to match GT median."""
    mask = ~np.isnan(gt) & ~np.isnan(pred)
    if np.sum(mask) == 0:
        return pred  # nothing to scale against
    scale = np.median(gt[mask]) / np.median(pred[mask])
    return pred * scale


# ---------------------------------------------------------
# 1. Choose a single KITTI frame
# ---------------------------------------------------------
left_path  = "data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png"
right_path = "data/2011_09_26/2011_09_26_drive_0001_sync/image_03/data/0000000005.png"
gt_path    = "data/2011_09_26/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png"

# Load frame (left RGB, right RGB, GT float32)
left, right, gt = load_frame(left_path, right_path, gt_path)

# ---------------------------------------------------------
# 2. Initialize models
# ---------------------------------------------------------
sgbm = StereoSGBM()
mono = Monodepth2()
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
pred_mono = median_scale(pred_mono, gt)
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
