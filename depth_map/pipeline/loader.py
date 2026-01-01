import cv2
import glob
import os
import numpy as np

def load_kitti_sequence(base):
    """
    Loads KITTI stereo sequence:
      - left images  (image_02)
      - right images (image_03)
      - groundtruth depth (proj_depth/groundtruth/image_02)
    Returns three sorted lists of file paths.
    """
    left_paths  = sorted(glob.glob(os.path.join(base, "image_02/data/*.png")))
    right_paths = sorted(glob.glob(os.path.join(base, "image_03/data/*.png")))
    gt_paths    = sorted(glob.glob(os.path.join(base, "proj_depth/groundtruth/image_02/*.png")))

    assert len(left_paths) == len(right_paths) == len(gt_paths), \
        "KITTI sequence mismatch: left/right/gt counts differ"

    return list(zip(left_paths, right_paths, gt_paths))


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