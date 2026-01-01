import os
import sys
import subprocess
import torch
import numpy as np
from collections import OrderedDict

# ---------------------------------------------------------
# Args object matching RAFT-Stereo demo.py
# ---------------------------------------------------------
class Args:
    restore_ckpt = None
    mixed_precision = False
    valid_iters = 32

    hidden_dims = [128, 128, 128]
    corr_implementation = "reg"
    shared_backbone = False
    corr_levels = 4
    corr_radius = 4
    n_downsample = 2
    context_norm = "batch"
    slow_fast_gru = True
    n_gru_layers = 3

    save_numpy = False
    left_imgs = None
    right_imgs = None
    output_directory = None


class RAFTStereoEstimator:
    def __init__(self, focal_length=721.5, baseline=0.537):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.f = focal_length
        self.B = baseline

        # ---------------------------------------------------------
        # 1. Clone RAFT-Stereo repo if missing
        # ---------------------------------------------------------
        repo = "third_party/RAFT-Stereo"
        if not os.path.exists(repo):
            os.makedirs("third_party", exist_ok=True)
            print("[RAFT-Stereo] Cloning repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/princeton-vl/RAFT-Stereo.git", repo],
                check=True
            )

        sys.path.append(repo)

        # ---------------------------------------------------------
        # 2. Download official pretrained weights if missing
        # ---------------------------------------------------------
        models_dir = os.path.join(repo, "models")
        ckpt_path = os.path.join(models_dir, "raftstereo-middlebury.pth")

        if not os.path.exists(ckpt_path):
            print("[RAFT-Stereo] Downloading pretrained weights...")
            subprocess.run(
                ["bash", "download_models.sh"],
                cwd=repo,
                check=True
            )

        # ---------------------------------------------------------
        # 3. Import RAFTStereo
        # ---------------------------------------------------------
        from core.raft_stereo import RAFTStereo

        # ---------------------------------------------------------
        # 4. Build args and load checkpoint
        # ---------------------------------------------------------
        args = Args()
        args.restore_ckpt = ckpt_path

        self.model = RAFTStereo(args).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)

        # Unwrap DataParallel checkpoints (remove "module." prefix)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if len(state) > 0 and list(state.keys())[0].startswith("module."):
            new_state = OrderedDict()
            for k, v in state.items():
                new_state[k.replace("module.", "", 1)] = v
            state = new_state

        self.model.load_state_dict(state)
        self.model.eval()

    # ---------------------------------------------------------
    # 5. Predict depth
    # ---------------------------------------------------------
    def predict(self, left, right):
        left_t  = torch.from_numpy(left).permute(2,0,1).float().unsqueeze(0).to(self.device)
        right_t = torch.from_numpy(right).permute(2,0,1).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, disp = self.model(left_t, right_t,
                                 iters=Args.valid_iters,
                                 test_mode=True)
            disp = disp.squeeze().cpu().numpy()

        # RAFT-Stereo uses opposite sign; demo.py visualizes -disp
        disp = -disp

        disp[disp <= 0] = np.nan
        return (self.f * self.B) / disp
