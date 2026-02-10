#!/usr/bin/env python3
"""
Generate a road-network metric distance matrix from the Bay Area (OSM),
then evaluate it using your metric_split.py best_bipartition(D, s).

Assumptions:
- You have metric_split.py on your PYTHONPATH (or same dir), exporting:
    best_bipartition(D: np.ndarray, s: float, tol=...) -> tuple[np.ndarray, np.ndarray] | None
  (Adjust import below if your module path differs.)

Install:
  pip install osmnx networkx numpy tqdm

Examples:
  python bay_roads_instance.py --place "San Francisco, California, USA" --n 200 --s 1.5
  python bay_roads_instance.py --places "San Francisco, California, USA" "Oakland, California, USA" "Berkeley, California, USA" --n 250 --s 1.5
  python bay_roads_instance.py --bbox 37.92 37.25 -121.75 -122.75 --n 250 --s 1.5
"""

from __future__ import annotations

import argparse
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import networkx as nx
from tqdm import tqdm

import osmnx as ox

# 🔧 Adjust this import to match your repo layout.
from metric_split import best_bipartition  # type: ignore


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--place", type=str, help='e.g. "San Francisco, California, USA"')
    mode.add_argument("--places", type=str, nargs="+", help="multiple place strings to merge")
    mode.add_argument("--bbox", type=float, nargs=4, metavar=("NORTH", "SOUTH", "EAST", "WEST"),
                      help="Bounding box lat/lon: north south east west")

    ap.add_argument("--network-type", type=str, default="drive", choices=["drive", "walk", "bike", "all"])
    ap.add_argument("--n", type=int, default=200, help="number of sampled graph nodes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-dist", type=float, default=float("inf"),
                    help="Dijkstra cutoff radius in meters (speed knob). inf = no cutoff.")
    ap.add_argument("--sym", type=str, default="avg", choices=["max", "min", "avg"],
                    help="Symmetrize directed road distances into a symmetric matrix.")
    ap.add_argument("--ensure-finite", action="store_true",
                    help="Keep only largest subset where all pairwise distances are finite.")
    ap.add_argument("--s", type=float, default=1.5, help="separation factor passed to best_bipartition")
    ap.add_argument("--tol", type=float, default=1e-7, help="tol passed to best_bipartition")

    ap.add_argument("--save-npy", type=str, default="",
                    help="optional path to save the final distance matrix D as .npy")
    return ap.parse_args()


def build_graph(args: argparse.Namespace) -> nx.MultiDiGraph:
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
        for Gi in graphs[1:]:
            G = nx.compose(G, Gi)
    else:
        north, south, east, west = args.bbox
        print(f"Downloading OSM graph for bbox N={north} S={south} E={east} W={west} ({args.network_type})")
        G = ox.graph_from_bbox(north, south, east, west, network_type=args.network_type, simplify=True)

    # Project graph to a metric CRS (helps ensure length units are meters consistently)
    G = ox.project_graph(G)

    # Ensure edge attribute "length" exists (meters). In OSMnx v2, it usually already does.
    try:
        edge_data = next(iter(G.edges(data=True)))[2]
    except StopIteration:
        raise RuntimeError("Graph has no edges; try a different place/bbox/network_type.")

    if "length" not in edge_data:
        # OSMnx v2 location: osmnx.distance.add_edge_lengths
        import osmnx.distance as oxdist
        G = oxdist.add_edge_lengths(G)

    return G


def sample_nodes(G: nx.MultiDiGraph, n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    nodes = np.array(list(G.nodes()))
    if n > nodes.size:
        raise ValueError(f"Requested n={n} but graph has only {nodes.size} nodes.")
    return rng.choice(nodes, size=n, replace=False).tolist()


def all_pairs_road_distances(
    G: nx.MultiDiGraph,
    sampled: List[int],
    max_dist: float,
) -> np.ndarray:
    """
    Compute directed shortest path distances (meters) between sampled nodes.
    """
    idx_of: Dict[int, int] = {node: i for i, node in enumerate(sampled)}
    n = len(sampled)
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    cutoff = None if math.isinf(max_dist) else float(max_dist)

    for i, src in enumerate(tqdm(sampled, desc="Dijkstra (road metric)")):
        lengths = nx.single_source_dijkstra_path_length(G, src, cutoff=cutoff, weight="length")
        for tgt, d in lengths.items():
            j = idx_of.get(tgt)
            if j is not None:
                D[i, j] = float(d)

    return D


def symmetrize(D: np.ndarray, method: str) -> np.ndarray:
    A = D
    B = D.T
    if method == "max":
        S = np.maximum(A, B)
    elif method == "min":
        S = np.minimum(A, B)
    else:
        finite = np.isfinite(A) & np.isfinite(B)
        S = np.where(finite, 0.5 * (A + B), np.maximum(A, B))
    np.fill_diagonal(S, 0.0)
    return S


def largest_all_finite_subset(D: np.ndarray) -> np.ndarray:
    """
    Keep largest subset where distances are finite along a connectivity proxy.
    Then we re-check all-finite and error if still not all finite.
    """
    n = D.shape[0]
    finite = np.isfinite(D)
    adj = finite & (~np.eye(n, dtype=bool))
    Gf = nx.from_numpy_array(adj)
    comps = list(nx.connected_components(Gf))
    comps.sort(key=len, reverse=True)
    keep = np.array(sorted(comps[0]), dtype=int)
    return keep


def main() -> None:
    args = parse_args()

    G = build_graph(args)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    sampled = sample_nodes(G, args.n, args.seed)
    print(f"Sampled {len(sampled)} nodes")

    D_dir = all_pairs_road_distances(G, sampled, args.max_dist)
    D = symmetrize(D_dir, args.sym)

    if args.ensure_finite:
        keep = largest_all_finite_subset(D)
        if keep.size < D.shape[0]:
            print(f"Reducing to largest finite-connected subset: {keep.size}/{D.shape[0]}")
            D = D[np.ix_(keep, keep)]

    if not np.all(np.isfinite(D)):
        bad = np.count_nonzero(~np.isfinite(D))
        raise SystemExit(
            f"D contains {bad} inf entries. Try: smaller --n, larger --max-dist, or --ensure-finite."
        )

    if args.save_npy:
        np.save(args.save_npy, D)
        print(f"Saved D to {args.save_npy} (shape={D.shape})")

    # ✅ Your actual “is it good?” test
    print(f"Calling best_bipartition(D, s={args.s}) ...")
    res = best_bipartition(D, s=float(args.s), tol=float(args.tol))

    if res is None:
        print("best_bipartition returned None (instance NOT good under this s).")
    else:
        A, B = res
        # tolerate list / ndarray return types
        A = np.asarray(A)
        B = np.asarray(B)
        print(f"✅ best_bipartition returned a cut: |A|={A.size}, |B|={B.size}, n={D.shape[0]}")
        print(f"Balance max(|A|,|B|)={max(A.size, B.size)}")


if __name__ == "__main__":
    main()