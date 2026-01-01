import numpy as np

def compute_metrics(pred, gt):
    """
    pred, gt: HxW depth arrays (float). KITTI GT is sparse.
    Returns:
      rmse, absrel, d1, d2, d3, error_map
    """
    mask = ~np.isnan(gt) & ~np.isnan(pred)
    p = pred[mask]
    g = gt[mask]

    # Avoid zeros in denominators
    g_safe = np.where(g == 0, 1e-6, g)
    p_safe = np.where(p == 0, 1e-6, p)

    abs_err = np.abs(p - g)
    rel_err = abs_err / g_safe

    rmse = np.sqrt(np.mean((p - g) ** 2))
    absrel = np.mean(rel_err)

    ratio = np.maximum(p_safe / g_safe, g_safe / p_safe)
    d1 = np.mean((ratio < 1.25).astype(np.float32))
    d2 = np.mean((ratio < 1.25 ** 2).astype(np.float32))
    d3 = np.mean((ratio < 1.25 ** 3).astype(np.float32))

    error_map = np.abs(pred - gt).astype(np.float32)
    error_map[~mask] = np.nan

    return rmse, absrel, d1, d2, d3, error_map
