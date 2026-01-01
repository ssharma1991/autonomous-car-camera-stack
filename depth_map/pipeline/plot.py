import matplotlib.pyplot as plt

def save_metric_plot(curves, path):
    plt.figure(figsize=(14, 10))

    # RMSE
    plt.subplot(3, 1, 1)
    plt.plot(curves["SGBM_rmse"], label="SGBM")
    plt.plot(curves["Monodepth2_rmse"], label="Monodepth2")
    plt.plot(curves["RAFT-Stereo_rmse"], label="RAFT-Stereo")
    plt.title("RMSE")
    plt.grid(True)
    plt.legend()

    # AbsRel
    plt.subplot(3, 1, 2)
    plt.plot(curves["SGBM_absrel"], label="SGBM")
    plt.plot(curves["Monodepth2_absrel"], label="Monodepth2")
    plt.plot(curves["RAFT-Stereo_absrel"], label="RAFT-Stereo")
    plt.title("AbsRel")
    plt.grid(True)
    plt.legend()

    # δ1
    plt.subplot(3, 1, 3)
    plt.plot(curves["SGBM_d1"], label="SGBM δ1")
    plt.plot(curves["Monodepth2_d1"], label="Monodepth2 δ1")
    plt.plot(curves["RAFT-Stereo_d1"], label="RAFT-Stereo δ1")
    plt.title("δ1 Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
