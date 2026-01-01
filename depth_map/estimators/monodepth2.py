import os
import sys
import subprocess
import torch
import cv2
import numpy as np

class Monodepth2:
    def __init__(self, model_name="mono+stereo_640x192", scale_factor=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale_factor = scale_factor

        # ---------------------------------------------------------
        # 1. Clone repo if missing
        # ---------------------------------------------------------
        repo = "third_party/monodepth2"
        if not os.path.exists(repo):
            os.makedirs("third_party", exist_ok=True)
            print("[Monodepth2] Cloning repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/nianticlabs/monodepth2.git", repo],
                check=True
            )
        sys.path.append(repo)

        # Import networks + download helper
        from networks import ResnetEncoder, DepthDecoder
        from utils import download_model_if_doesnt_exist

        # ---------------------------------------------------------
        # 2. Download weights (goes to ~/.monodepth2_models/)
        # ---------------------------------------------------------
        download_model_if_doesnt_exist(model_name)

        # Correct path to downloaded weights
        model_dir = os.path.join(os.path.expanduser("~/.monodepth2_models"), model_name)
        enc_path = os.path.join(model_dir, "encoder.pth")
        dec_path = os.path.join(model_dir, "depth.pth")

        # ---------------------------------------------------------
        # 3. Load encoder + decoder
        # ---------------------------------------------------------
        self.encoder = ResnetEncoder(18, False).to(self.device)
        enc_weights = torch.load(enc_path, map_location=self.device)
        self.encoder.load_state_dict(
            {k: v for k, v in enc_weights.items() if k in self.encoder.state_dict()}
        )
        self.encoder.eval()

        self.decoder = DepthDecoder(self.encoder.num_ch_enc, range(4)).to(self.device)
        self.decoder.load_state_dict(torch.load(dec_path, map_location=self.device))
        self.decoder.eval()

        # Input resolution
        self.h = enc_weights["height"]
        self.w = enc_weights["width"]


    def predict(self, left_img):
        # Preprocess
        img = cv2.resize(left_img, (self.w, self.h))
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            features = self.encoder(img)
            disp = self.decoder(features)[("disp", 0)].squeeze().cpu().numpy()

        # Convert sigmoid disparity → depth
        min_disp, max_disp = 1e-3, 10.0
        disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / disp

        # Apply global scale factor (metric correction)
        return depth * self.scale_factor


    def compute_frame_scale(self, pred, gt):
        """Return median(gt)/median(pred) for one frame, or None if invalid."""
        mask = ~np.isnan(gt) & ~np.isnan(pred)
        if np.sum(mask) == 0:
            return None

        med_gt = np.median(gt[mask])
        med_pred = np.median(pred[mask])

        if med_pred <= 0:
            return None

        return med_gt / med_pred

