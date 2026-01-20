import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    data = np.load("sweep_results.npz")

    bad_counts = data["bad_counts"]
    s_values = data["s_values"]
    n_values = data["n_values"]

    # Mask zeros so they appear as "empty"
    masked = np.ma.masked_where(bad_counts == 0, bad_counts)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Log scale:
    # vmin must be >= 1 since log(0) is undefined
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        norm=LogNorm(vmin=1, vmax=masked.max()),
        cmap="viridis",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("NOT TSP-separated (log scale)")

    # Axes / labels
    ax.set_title("Bad tour counts (log scale)")
    ax.set_xlabel("separation factor s")
    ax.set_ylabel("num points")

    ax.set_xticks(np.arange(len(s_values)))
    ax.set_xticklabels([f"{s:.2f}" for s in s_values], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(n_values)))
    ax.set_yticklabels([str(n) for n in n_values])

    # Optional: annotate nonzero cells with raw counts
    for i in range(len(n_values)):
        for j in range(len(s_values)):
            if bad_counts[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(int(bad_counts[i, j])),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )

    fig.tight_layout()
    fig.savefig("heatmap_bad_counts_log.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
