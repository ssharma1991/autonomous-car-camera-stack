import cv2
import numpy as np

class StereoSGBM:
    def __init__(self, focal_length=721.5, baseline=0.537):
        self.f = focal_length
        self.B = baseline

        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,      # must be divisible by 16
            blockSize=5,
            P1=8 * 5 * 5,
            P2=32 * 5 * 5,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2
        )

    def predict(self, left, right):
        disp = self.matcher.compute(left, right).astype(np.float32) / 16.0

        # Mask invalid disparities
        disp[disp <= 0] = np.nan

        # Convert to depth
        depth = (self.f * self.B) / disp

        # Mask invalid depth values
        depth[np.isinf(depth)] = np.nan
        depth[depth <= 0] = np.nan

        return depth
