#!/usr/bin/env python
"""
Bay Area: sample OSM nodes, build Euclidean distance matrix from projected coordinates,
then call metric_split.best_bipartition(D, s).

Install:
  pip install osmnx numpy scipy

Examples:
  python bay_points_euclidean.py --place "San Francisco, California, USA" --n 600 --s 1.5
  python bay_points_euclidean.py --places "San Francisco, California, USA" "Oakland, California, USA" "Berkeley, California, USA" --n 800 --s 1.5
  python bay_points_euclidean.py --bbox 37.90 37.45 -122.20 -122.60 --n 1000 --s 1.5
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from scipy.spatial.distance import pdist, squareform

# 🔧 Adjust this import to match your repo layout
from metric_split import best_bipartition  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--place", type=str, help='e.g. "San Francisco, California, USA"')
    mode.add_argument("--places", type=str, nargs="+", help="multiple place strings to merge")
    mode.add_argument("--bbox", type=float, nargs=4, metavar=("NORTH", "SOUTH", "EAST", "WEST"),
                      help="Bounding box lat/lon: north south east west")

    ap.add_argument("--n", type=int, default=600, help="number of sampled points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--network-type", type=str, default="drive", choices=["drive", "walk", "bike", "all"],
                    help="used only to fetch a graph to get node coordinates")
    ap.add_argument("--s", type=float, default=1.5, help="separation factor passed to best_bipartition")
    ap.add_argument("--tol", type=float, default=1e-12, help="tol passed to best_bipartition")
    
    ap.add_argument("--plot", action="store_true", help="Plot sampled points")
    ap.add_argument("--plot-cut", action="store_true", help="If cut exists, plot A/B coloring")
    ap.add_argument("--plot-size", type=int, default=5, help="Matplotlib marker size")

    ap.add_argument("--save", type=str, default="",
                    help="prefix to save outputs: <prefix>_xy.npy, <prefix>_D.npy, <prefix>_cut.npy")
    return ap.parse_args()


def build_graph(args: argparse.Namespace):
    ox.settings.log_console = False
    ox.settings.use_cache = True

    if args.place:
        print(f"Downloading OSM graph for place: {args.place} ({args.network_type})")
        G = ox.graph_from_place(args.place, network_type=args.network_type, simplify=True)
    elif args.places:
        graphs = []
        for p in args.places:
            print(f"Downloading OSM graph for place: {p} ({args.network_type})")
            graphs.append(ox.graph_from_place(p, network_type=args.network_type, simplify=True))
        print("Merging graphs...")
        G = graphs[0]
        import networkx as nx
        for Gi in graphs[1:]:
            G = nx.compose(G, Gi)
    else:
        north, south, east, west = args.bbox
        print(f"Downloading OSM graph for bbox N={north} S={south} E={east} W={west} ({args.network_type})")
        G = ox.graph_from_bbox(north, south, east, west, network_type=args.network_type, simplify=True)

    # Project to metric CRS so x,y are in meters
    G = ox.project_graph(G)
    return G


def sample_xy(G, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes(data=True))
    if n > len(nodes):
        raise ValueError(f"Requested n={n} but graph has only {len(nodes)} nodes.")

    idx = rng.choice(len(nodes), size=n, replace=False)
    xy = np.empty((n, 2), dtype=np.float64)

    for k, t in enumerate(idx):
        _node_id, data = nodes[int(t)]
        # OSMnx projected graphs store x/y (meters)
        xy[k, 0] = float(data["x"])
        xy[k, 1] = float(data["y"])

    return xy


def euclidean_distance_matrix(xy: np.ndarray) -> np.ndarray:
    # pdist is fast + memory-efficient; squareform returns dense n x n
    return squareform(pdist(xy, metric="euclidean")).astype(np.float64)

def plot_points(xy: np.ndarray, A=None, B=None, size=5):
    plt.figure(figsize=(8, 8))

    if A is None or B is None:
        plt.scatter(xy[:, 0], xy[:, 1], s=size)
    else:
        plt.scatter(xy[A, 0], xy[A, 1], s=size, label="A")
        plt.scatter(xy[B, 0], xy[B, 1], s=size, label="B")
        plt.legend()

    plt.title("Sampled Bay Area Points")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

def main() -> None:
    args = parse_args()
    G = build_graph(args)
    print(f"Graph nodes available: {G.number_of_nodes()}")

    xy = sample_xy(G, args.n, args.seed)
    print(f"Sampled points: {xy.shape[0]}")
    if args.plot and not args.plot_cut:
        plot_points(xy, size=args.plot_size)

    D = euclidean_distance_matrix(xy)
    print(f"Built Euclidean D: shape={D.shape}, dtype={D.dtype}")

    print(f"Calling best_bipartition(D, s={args.s}) ...")
    res = best_bipartition(D, s=float(args.s), tol=float(args.tol))

    if res is None:
        print("best_bipartition returned None.")
        if args.plot:
            plot_points(xy, size=args.plot_size)
    else:
        A, B = res
        A = np.asarray(A, dtype=int)
        B = np.asarray(B, dtype=int)

        print(f"✅ cut found: |A|={A.size}, |B|={B.size}, balance={max(A.size, B.size)}")

        if args.plot:
            plot_points(xy, A, B, size=args.plot_size)

    if args.save:
        np.save(args.save + "_xy.npy", xy)
        np.save(args.save + "_D.npy", D)
        if cut is not None:
            # Save as object array: [A, B]
            np.save(args.save + "_cut.npy", np.array([cut[0], cut[1]], dtype=object))
        print(f"Saved outputs with prefix: {args.save}")


if __name__ == "__main__":
    main()