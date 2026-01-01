import cv2
import glob
import os
import numpy as np

def load_kitti_sequence(base):
    """
    Load KITTI stereo frames aligned by GT filenames.
    GT exists only for frames >= 5, so we match by filename.
    Returns: list of (left, right, gt) tuples.
    """
    lefts  = glob.glob(os.path.join(base, "image_02/data/*.png"))
    rights = glob.glob(os.path.join(base, "image_03/data/*.png"))
    gts    = glob.glob(os.path.join(base, "proj_depth/groundtruth/image_02/*.png"))

    lefts  = {os.path.basename(p): p for p in lefts}
    rights = {os.path.basename(p): p for p in rights}

    aligned = []
    for gt in sorted(gts):
        name = os.path.basename(gt)
        if name in lefts and name in rights:
            aligned.append((lefts[name], rights[name], gt))

    print(f"Loaded {len(aligned)} aligned frames")
    return aligned


def load_frame(left_path, right_path, gt_path):
    """
    Loads a single KITTI frame:
      - left RGB image (H,W,3)
      - right RGB image (H,W,3)
      - groundtruth depth (H,W float32)
    """
    # Left image (RGB)
    left = cv2.imread(left_path, cv2.IMREAD_COLOR)
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    # Right image (RGB)
    right = cv2.imread(right_path, cv2.IMREAD_COLOR)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    # Groundtruth depth (float32)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype("float32")
    gt_meters = gt / 256.0
    gt_meters[gt_meters == 0] = np.nan  # Set invalid depths to NaN

    return left, right, gt_meters