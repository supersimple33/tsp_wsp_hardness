import math
import time
import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT: change this to your actual filename (without .py)
import cluster_sep as exp


def frange_inclusive(a: float, b: float, step: float) -> list[float]:
    # robust float stepping to avoid 0.30000000000004 stuff in labels
    n = int(round((b - a) / step))
    vals = [a + i * step for i in range(n + 1)]
    return [round(x, 10) for x in vals]


def main():
    # Sweep spec
    s_values = frange_inclusive(0.0, 0.5, 0.05)  # 0.0..0.5 inclusive
    n_values = list(range(5, 41, 5))  # 5..40 step 5

    TAKE = 10_000
    DISTRIB_CODE = (
        exp.DISTRIB_CODE
    )  # uses whatever you set in your main file (e.g. "cr")
    SCALE_SIZE = exp.SCALE_SIZE
    START_INDEX = 0
    CONCORDE_SEED = exp.CONCORDE_SEED

    # results matrix: rows = n_values, cols = s_values
    bad_counts = np.zeros((len(n_values), len(s_values)), dtype=np.int32)
    bad_rates = np.zeros_like(bad_counts, dtype=np.float64)

    t0 = time.time()
    total_runs = len(n_values) * len(s_values)
    run_i = 0

    for i, n in enumerate(n_values):
        for j, s in enumerate(s_values):
            run_i += 1
            bad_count, _bad_instances = exp.run_experiment(
                scale_size=SCALE_SIZE,
                num_points=n,
                take=TAKE,
                start_index=START_INDEX,
                distrib_code=DISTRIB_CODE,
                s_factor=float(s),
                concorde_seed=CONCORDE_SEED,
                print_every=0,  # keep sweep quiet
            )
            bad_counts[i, j] = bad_count
            bad_rates[i, j] = bad_count / TAKE

            print(
                f"[{run_i:02d}/{total_runs}] n={n:2d} s={s:0.2f} "
                f"bad={bad_count:4d}/{TAKE} rate={bad_rates[i,j]:.4f}"
            )

    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.2f}s")

    # Save raw arrays for later analysis
    np.savez(
        "sweep_results.npz",
        s_values=np.array(s_values, dtype=np.float64),
        n_values=np.array(n_values, dtype=np.int32),
        bad_counts=bad_counts,
        bad_rates=bad_rates,
        take=np.int32(TAKE),
        distrib_code=np.array([DISTRIB_CODE]),
        scale_size=np.int32(SCALE_SIZE),
        concorde_seed=np.int32(CONCORDE_SEED),
    )

    # --- Heatmap (counts) ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(bad_counts, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label="NOT TSP-separated (count)")

    ax.set_title(f"Bad tour count (TAKE={TAKE}, distrib={DISTRIB_CODE})")
    ax.set_xlabel("separation factor s")
    ax.set_ylabel("num points")

    ax.set_xticks(np.arange(len(s_values)))
    ax.set_xticklabels([f"{s:0.2f}" for s in s_values], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(n_values)))
    ax.set_yticklabels([str(n) for n in n_values])

    # annotate cells (counts)
    for i in range(len(n_values)):
        for j in range(len(s_values)):
            ax.text(j, i, str(int(bad_counts[i, j])), ha="center", va="center")

    fig.tight_layout()
    fig.savefig("heatmap_bad_counts.png", dpi=200)

    # --- Heatmap (rates) ---
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    im2 = ax2.imshow(bad_rates, aspect="auto", origin="lower")
    fig2.colorbar(im2, ax=ax2, label="NOT TSP-separated (rate)")

    ax2.set_title(f"Bad tour rate (TAKE={TAKE}, distrib={DISTRIB_CODE})")
    ax2.set_xlabel("separation factor s")
    ax2.set_ylabel("num points")

    ax2.set_xticks(np.arange(len(s_values)))
    ax2.set_xticklabels([f"{s:0.2f}" for s in s_values], rotation=45, ha="right")

    ax2.set_yticks(np.arange(len(n_values)))
    ax2.set_yticklabels([str(n) for n in n_values])

    # annotate cells (rate, 3 decimals)
    for i in range(len(n_values)):
        for j in range(len(s_values)):
            ax2.text(j, i, f"{bad_rates[i, j]:.3f}", ha="center", va="center")

    fig2.tight_layout()
    fig2.savefig("heatmap_bad_rates.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
