import matplotlib.pyplot as plt

# The exact sequence of points from the violation
path_data = [
    {"step": 1, "idx": 0, "type": "Inner", "coord": (-0.4115, -0.1562)},
    {"step": 2, "idx": 3, "type": "Outer", "coord": (-4.4756, -0.0900)},
    {"step": 3, "idx": 2, "type": "Inner", "coord": (-0.0413, -0.2084)},
    {"step": 4, "idx": 4, "type": "Outer", "coord": (1.7915, -1.3349)},
    {"step": 5, "idx": 1, "type": "Inner", "coord": (0.2360, 0.3503)},
    {"step": 6, "idx": 5, "type": "Outer", "coord": (1.2587, 3.1593)},
]

# Extract coordinates for scatter plotting
inner_x = [d["coord"][0] for d in path_data if d["type"] == "Inner"]
inner_y = [d["coord"][1] for d in path_data if d["type"] == "Inner"]
outer_x = [d["coord"][0] for d in path_data if d["type"] == "Outer"]
outer_y = [d["coord"][1] for d in path_data if d["type"] == "Outer"]

# Create the plot
plt.figure(figsize=(10, 8))

# Draw the points
plt.scatter(inner_x, inner_y, c='dodgerblue', s=150, label='Inner Cloud', edgecolors='black', zorder=5)
plt.scatter(outer_x, outer_y, c='crimson', s=150, label='Outer Cloud', edgecolors='black', zorder=5)

# Draw an approximate boundary for the inner cloud (radius 0.5)
inner_circle = plt.Circle((0, 0), 0.5, color='dodgerblue', fill=False, linestyle='--', alpha=0.5, label='Inner Boundary (r=0.5)')
plt.gca().add_patch(inner_circle)

# Draw the path with arrows
for i in range(len(path_data) - 1):
    start = path_data[i]["coord"]
    end = path_data[i+1]["coord"]
    
    # Use annotate to draw arrows between steps
    plt.annotate(
        "", 
        xy=end, 
        xytext=start,
        arrowprops=dict(arrowstyle="->", color="gray", lw=2, shrinkA=8, shrinkB=8),
        zorder=3
    )

# Label each point with its Step and Index
for d in path_data:
    x, y = d["coord"]
    plt.text(
        x + 0.15, y + 0.15, 
        f"Step {d['step']}\n(Idx {d['idx']})", 
        fontsize=10, 
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
        zorder=10
    )

# Formatting
plt.title("Hamiltonian Path Violation: Bouncing Between Clouds", fontsize=14, pad=15)
plt.xlabel("X Coordinate", fontsize=12)
plt.ylabel("Y Coordinate", fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, alpha=0.3)
plt.axvline(0, color='black', linewidth=0.5, alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=11)

# Set axis limits to give the plot some breathing room
plt.xlim(-5, 2.5)
plt.ylim(-2, 3.5)

plt.tight_layout()
plt.show()