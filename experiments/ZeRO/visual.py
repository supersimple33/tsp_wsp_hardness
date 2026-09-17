import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np


def get_matrix(s):
    M = np.zeros((7, 7))
    lower = [
        [0],
        [1, 0],
        [s, s, 0],
        [s, s, 1, 0],
        [s, s + 1, s, s + 1, 0],
        [s + 1, s, s, s + 1, 2*s, 0],
        [s + 1, s, s + 1, s, 2*s + 1, 2*s, 0]
    ]
    for i in range(7):
        for j in range(i + 1):
            M[i, j] = lower[i][j]
            M[j, i] = lower[i][j]
    return M



# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

# 1. Heatmap with algebraic annotations
s_val = 5
M_numeric = get_matrix(s_val)

s_val = 5
M = get_matrix(s_val)

essential_edges = []
n = 7
for i in range(n):
    for j in range(i + 1, n):
        # check if d(i, j) == d(i, k) + d(k, j) for any k
        decomposable = False
        for k in range(n):
            if k != i and k != j:
                if np.isclose(M[i, j], M[i, k] + M[k, j]):
                    decomposable = True
                    break
        if not decomposable:
            essential_edges.append((i, j, M[i, j]))

print("Essential edges without internal nodes:")
for u, v, w in essential_edges:
    print(f"({u}, {v}): {w}")

G = nx.Graph()
for u, v, w in essential_edges:
    G.add_edge(u, v, weight=w)

sp = dict(nx.all_pairs_dijkstra_path_length(G))
matches = True
for i in range(n):
    for j in range(n):
        if not np.isclose(sp[i][j], M[i, j]):
            print(f"Mismatch at ({i}, {j}): graph {sp[i][j]} vs matrix {M[i, j]}")
            matches = False
print(f"All pairs match: {matches}")

sym_matrix = [
    ["0", "1", "s", "s", "s", "s+1", "s+1"],
    ["1", "0", "s", "s", "s+1", "s", "s"],
    ["s", "s", "0", "1", "s", "s", "s+1"],
    ["s", "s", "1", "0", "s+1", "s+1", "s"],
    ["s", "s+1", "s", "s+1", "0", "2s", "2s+1"],
    ["s+1", "s", "s", "s+1", "2s", "0", "2s"],
    ["s+1", "s", "s+1", "s", "2s+1", "2s", "0"]
]

cax = ax1.imshow(M_numeric, cmap='YlGnBu', interpolation='nearest')
cbar = fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label(f'Metric Distance (evaluated at s = {s_val})', rotation=270, labelpad=15, fontsize=10)

labels = [f"$v_{i}$" for i in range(7)]
ax1.set_xticks(range(7))
ax1.set_yticks(range(7))
ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax1.set_yticklabels(labels, fontsize=11, fontweight='bold')
ax1.set_title("Distance Matrix Heatmap (with Symbolic Entries)", fontsize=13, fontweight='bold', pad=12)

# Annotate each cell
for i in range(7):
    for j in range(7):
        val_str = sym_matrix[i][j]
        color = 'white' if M_numeric[i, j] > 7 else 'black'
        ax1.text(j, i, val_str, ha='center', va='center', color=color, fontsize=10, fontweight='semibold')

# 2. Structural Graph of Essential Distances (Shortest Path Metric Generator)
# Layout design:
# Pair A: v0, v1 (left)
# Pair B: v2, v3 (right)
# Outer: v4 (top), v5 (bottom-left/middle), v6 (bottom-right)
pos = {
    0: np.array([-1.0, 1.0]),
    1: np.array([-1.0, -1.0]),
    2: np.array([1.0, 1.0]),
    3: np.array([1.0, -1.0]),
    4: np.array([0.0, 2.8]),
    5: np.array([-0.2, -2.8]),
    6: np.array([2.5, -2.2])
}

# Essential edges:
# Weight 1 edges:
unit_edges = [(0, 1), (2, 3)]
# Weight s edges:
s_edges_core = [(0, 2), (0, 3), (1, 2), (1, 3)]
s_edges_outer = [(4, 0), (4, 2), (5, 1), (5, 2), (6, 1), (6, 3)]

# Draw edges
nx.draw_networkx_edges(G, pos, edgelist=unit_edges, ax=ax2, edge_color='#D9534F', width=3.5, label='Weight = 1')
nx.draw_networkx_edges(G, pos, edgelist=s_edges_core, ax=ax2, edge_color='#4A90E2', width=1.8, style='dashed', alpha=0.7, label='Weight = s (cross-core)')
nx.draw_networkx_edges(G, pos, edgelist=s_edges_outer, ax=ax2, edge_color='#2E7D32', width=2.2, label='Weight = s (outer spokes)')

# Draw nodes
node_colors = ['#FFD54F', '#FFD54F', '#81D4FA', '#81D4FA', '#E1BEE7', '#E1BEE7', '#E1BEE7']
nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, node_size=850, edgecolors='#333333', linewidths=1.8)
nx.draw_networkx_labels(G, pos, labels={i: f"$v_{i}$" for i in range(7)}, ax=ax2, font_size=12, font_weight='bold')

# Edge labels
edge_labels = {
    (0, 1): '1',
    (2, 3): '1',
    (4, 0): 's',
    (4, 2): 's',
    (5, 1): 's',
    (5, 2): 's',
    (6, 1): 's',
    (6, 3): 's'
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2, font_size=10, font_color='#222222', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

ax2.set_title("Shortest-Path Generator Graph (Tight Skeleton)", fontsize=13, fontweight='bold', pad=12)
ax2.legend(loc='upper right', frameon=True, fontsize=9)
ax2.axis('off')

plt.tight_layout()
plt.savefig('distance_matrix_visualization.png', dpi=300)
plt.show()