import cv2
import numpy as np
import subprocess

FONT = cv2.FONT_HERSHEY_SIMPLEX

def _normalize(x):
    x = x.copy()
    x[np.isnan(x)] = 0
    x = np.clip(x, 0, np.percentile(x, 95))
    m = x.max()
    return np.zeros_like(x, dtype=np.uint8) if m <= 0 else (x / m * 255).astype(np.uint8)

def colorize_depth(depth):
    return cv2.applyColorMap(_normalize(depth), cv2.COLORMAP_MAGMA)

def colorize_error(err):
    return cv2.applyColorMap(_normalize(err), cv2.COLORMAP_TURBO)

def add_text(img, text, y=30):
    cv2.putText(img, text, (10, y), FONT, 0.9, (255,255,255), 2, cv2.LINE_AA)

def add_metrics(img, metrics, y=60):
    rmse, absrel, d1, d2, d3 = metrics
    txt = f"RMSE:{rmse:.2f} AbsRel:{absrel:.3f} d1:{d1:.2f} d2:{d2:.2f} d3:{d3:.2f}"
    cv2.putText(img, txt, (10, y), FONT, 0.8, (255,255,255), 2, cv2.LINE_AA)

def make_tile(image, label=None, metrics=None):
    tile = image.copy()
    if label:
        add_text(tile, label)
    if metrics:
        add_metrics(tile, metrics)
    return tile

class VideoWriter:
    def __init__(self, output_path, frame_size, fps=5):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def add_frame(self, left, right, preds, errors, metrics, gt):
        """
        preds:  dict {model_name: depth_map}
        errors: dict {model_name: error_map}
        metrics: dict {model_name: (rmse, absrel, d1, d2, d3)}
        """

        # Resize left/right to match depth map size
        h, w = next(iter(preds.values())).shape
        left  = cv2.resize(left,  (w, h))
        right = cv2.resize(right, (w, h))

        # Difficulty map
        difficulty = np.std(np.stack(list(preds.values()), axis=0), axis=0)

        # Build rows
        rows = []

        # Row 1: inputs
        left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        rows.append([
            make_tile(left,  "Left image"),
            make_tile(right, "Right image")
        ])

        # Rows for each model
        for name in preds:
            depth_tile = make_tile(colorize_depth(preds[name]), f"{name} depth")
            err_tile   = make_tile(colorize_error(errors[name]), f"{name} error", metrics[name])
            rows.append([depth_tile, err_tile])

        # Last row: GT + difficulty
        rows.append([
            make_tile(colorize_error(difficulty), "Model disagreement"),
            make_tile(colorize_depth(gt), "Groundtruth depth"),
        ])

        # Stack into final frame
        frame = np.vstack([np.hstack(r) for r in rows])
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def finalize_video(self, input_path):
        output_path = input_path.replace(".mp4", "_h264.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", output_path])
        return output_path