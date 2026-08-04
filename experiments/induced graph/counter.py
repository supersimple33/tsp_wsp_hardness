import itertools
import networkx as nx
import matplotlib.pyplot as plt

# Redundant path example
# K = 4: 5 6 7 8

def find_and_plot_valid_digraphs(k: int):
    """
    Finds, prints, and plots all non-isomorphic digraph configurations on k vertices.
    """
    if k <= 1:
        print(f"For k = {k}, no valid Eulerian cycle spanning all vertices exists.")
        return []

    possible_edges = [(u, v) for u in range(k) for v in range(k) if u != v]
    num_possible_edges = len(possible_edges)

    wl_hashes = {}
    valid_graphs = []

    # Iterate over all possible subgraphs
    for edge_mask in itertools.product([0, 1], repeat=num_possible_edges):
        edges = [possible_edges[i] for i in range(num_possible_edges) if edge_mask[i] == 1]

        if len(edges) < k:
            continue

        G = nx.DiGraph()
        G.add_nodes_from(range(k))
        G.add_edges_from(edges)

        # Condition 1: Eulerian (strongly connected, in-degree == out-degree)
        if any(G.in_degree(n) != G.out_degree(n) for n in G.nodes()):
            continue
        if not nx.is_strongly_connected(G):
            continue

        # Condition 2: No mirrored cycles of length >= 3
        edge_set = set(G.edges())
        has_mirror_cycle = False

        for cycle in nx.simple_cycles(G):
            m = len(cycle)
            if m >= 3:
                reversed_edges = [(cycle[0], cycle[-1])] + [
                    (cycle[i], cycle[i - 1]) for i in range(m - 1, 0, -1)
                ]
                if all(e in edge_set for e in reversed_edges):
                    has_mirror_cycle = True
                    break

        if has_mirror_cycle:
            continue

        # Condition 3: Unique up to Isomorphism
        ghash = nx.weisfeiler_lehman_graph_hash(G)
        is_duplicate = False

        if ghash in wl_hashes:
            for existing_G in wl_hashes[ghash]:
                if nx.is_isomorphic(G, existing_G):
                    is_duplicate = True
                    break

        if not is_duplicate:
            if ghash not in wl_hashes:
                wl_hashes[ghash] = []
            wl_hashes[ghash].append(G)
            valid_graphs.append(G)

    # --- PRINT AND RENDER RESULTS ---
    print(f"\n==================================================")
    print(f" RESULTS FOR k = {k} VERTICES")
    print(f" Total Unique Valid Configurations Found: {len(valid_graphs)}")
    print(f"==================================================\n")

    for idx, G in enumerate(valid_graphs, 1):
        edges_str = sorted(list(G.edges()))
        adj_list = {node: sorted(list(G.successors(node))) for node in sorted(G.nodes())}
        
        print(f"Graph #{idx}:")
        print(f"  • Edge List      : {edges_str}")
        print(f"  • Adjacency List : {adj_list}")
        print("-" * 50)

        # Initialize the plot
        plt.figure(figsize=(6, 6))
        plt.title(f"Valid Configuration #{idx} for k={k}", fontsize=14, fontweight='bold')
        
        # Circular layout prevents vertex overlap and looks neat for cycles
        pos = nx.circular_layout(G) 
        
        # Draw the graph with customized visual settings
        nx.draw_networkx(
            G, 
            pos, 
            with_labels=True, 
            node_color="#89CFF0",  # Baby blue nodes
            node_size=1500, 
            edge_color="#333333",  # Dark gray edges
            linewidths=2, 
            font_size=14, 
            font_weight="bold",
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.15" # Curves edges to prevent overlapping 2-cycles
        )
        
        # Remove axis grid/border lines
        plt.axis("off") 
        
        # Display the window (Script pauses here until you close the plot window)
        plt.show()

    return valid_graphs


if __name__ == "__main__":
    TARGET_K = 4

    find_and_plot_valid_digraphs(TARGET_K)
