import tkinter as tk
import math
import itertools

# You can now safely change this to 6, 7, 8, etc. 
# (Note: Anything above 10 will start to lag because brute-force TSP scales factorially!)
NUM_POINTS = 9

class TSPSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Fixed Start-End TSP Path Solver ({NUM_POINTS} Points)")
        self.root.geometry("650x450")

        self.norm_var = tk.StringVar(value="L2")

        # --- UI Control Panel ---
        control_frame = tk.Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(control_frame, text="Distance Metric:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(control_frame, text="L1 (Manhattan)", variable=self.norm_var, value="L1", command=self.solve_and_draw).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="L2 (Euclidean)", variable=self.norm_var, value="L2", command=self.solve_and_draw).pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="L_inf (Chebyshev)", variable=self.norm_var, value="L_inf", command=self.solve_and_draw).pack(side=tk.LEFT)

        # Added export button
        tk.Button(control_frame, text="Copy Weight Matrix", command=self.export_matrix_to_clipboard, bg="#e0e0e0").pack(side=tk.RIGHT, padx=10)

        # --- Interactive Canvas ---
        self.canvas = tk.Canvas(root, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initialize the points dynamically so it matches NUM_POINTS perfectly
        self.points = []
        for i in range(NUM_POINTS):
            # Spread points horizontally across the screen, alternating vertical positions
            x = 100 + i * (400 / max(1, NUM_POINTS - 1))
            y = 250 if i % 2 == 0 else 100
            self.points.append((int(x), int(y)))

        self.point_radius = 8
        self.dragged_idx = None
        self.distance_view_idx = None  # Tracks which node is toggled for the "all-distances" view

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Bind Right-Click (Button-3 for Windows/Linux, Button-2 for some Mac setups)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.on_right_click)

        # Initial solve and render
        self.solve_and_draw()

    def get_distance(self, p1, p2):
        """Calculates distance based on the currently selected mathematical norm."""
        norm = self.norm_var.get()
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])

        if norm == "L1":
            return dx + dy
        elif norm == "L2":
            return math.sqrt(dx**2 + dy**2)
        elif norm == "L_inf":
            return max(dx, dy)
        return 0

    def solve_tsp(self):
        """Finds the shortest path visiting all points from Start to End."""
        start_idx = 0
        end_idx = NUM_POINTS - 1
        
        # We only need to permute the intermediate points
        middle_indices = list(range(1, NUM_POINTS - 1))

        min_dist = float('inf')
        best_path = None

        # Brute force search (Permutations)
        for perm in itertools.permutations(middle_indices):
            path = [start_idx] + list(perm) + [end_idx]
            
            # Calculate total distance for this specific path permutation
            dist = sum(self.get_distance(self.points[path[i]], self.points[path[i+1]]) 
                       for i in range(NUM_POINTS - 1))

            if dist < min_dist:
                min_dist = dist
                best_path = path

        return best_path, min_dist

    def solve_and_draw(self):
        """Solves the TSP path problem and renders the nodes and lines."""
        self.canvas.delete("all")

        best_path, min_dist = self.solve_tsp()

        # 1. Draw the main TSP connection lines + edge lengths (Drawn first so they are on the bottom)
        if best_path:
            for i in range(NUM_POINTS - 1):
                p1 = self.points[best_path[i]]
                p2 = self.points[best_path[i+1]]

                # Draw the edge
                self.canvas.create_line(p1, p2, fill="black", width=2, dash=(4, 2))

                # Compute distance
                dist = self.get_distance(p1, p2)

                # Midpoint for label
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2

                # Slight offset so text doesn't sit exactly on the line
                offset = 10
                self.canvas.create_rectangle(mx-12, my-18, mx+12, my-2, fill="white", outline="")
                self.canvas.create_text(
                    mx, my - offset,
                    text=f"{dist:.1f}",
                    font=("Arial", 9, "bold"),
                    fill="purple"
                )

        # 2. Draw the points on top of the main path lines
        for i, (x, y) in enumerate(self.points):
            if i == 0:
                color, label = "green", "Start"
            elif i == NUM_POINTS - 1:
                color, label = "red", "End"
            else:
                color, label = "dodgerblue", f"P{i}"
                
            # Highlight the toggled node
            outline_color = "orange" if i == self.distance_view_idx else "black"
            outline_width = 4 if i == self.distance_view_idx else 2

            # Point Graphic
            self.canvas.create_oval(
                x - self.point_radius, y - self.point_radius,
                x + self.point_radius, y + self.point_radius,
                fill=color, outline=outline_color, width=outline_width
            )
            # Text Label
            self.canvas.create_text(x, y - 18, text=label, font=("Arial", 9, "bold"))

        # 3. Print the active metric and distance text
        metric_name = {"L1": "Manhattan", "L2": "Euclidean", "L_inf": "Chebyshev"}[self.norm_var.get()]
        self.canvas.create_text(
            15, 15, anchor=tk.NW,
            text=f"Shortest Path ({metric_name}): {min_dist:.2f}",
            font=("Arial", 12, "bold"), fill="darkred"
        )
        
        self.canvas.create_text(
            15, 35, anchor=tk.NW,
            text="Right-click a node to toggle all distances",
            font=("Arial", 9, "italic"), fill="gray"
        )

        # 4. Draw "All Distances" view (Drawn LAST so it sits entirely on top of everything else)
        if self.distance_view_idx is not None:
            p1 = self.points[self.distance_view_idx]
            for i, p2 in enumerate(self.points):
                if i == self.distance_view_idx:
                    continue
                
                # Draw the foreground edge
                self.canvas.create_line(p1, p2, fill="orange", width=2, dash=(2, 4))
                
                dist = self.get_distance(p1, p2)
                
                # Midpoint for label
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                
                # Background for the text to make it completely opaque over other lines/nodes
                self.canvas.create_rectangle(mx-14, my-8, mx+14, my+8, fill="white", outline="orange", width=1)
                self.canvas.create_text(
                    mx, my,
                    text=f"{dist:.1f}",
                    font=("Arial", 8, "bold"),
                    fill="darkorange"
                )

    # --- Clipboard Export ---
    def export_matrix_to_clipboard(self):
        """Calculates the NxN weight matrix and copies it to the clipboard."""
        rows = []
        for i in range(NUM_POINTS):
            row_dists = []
            for j in range(NUM_POINTS):
                dist = self.get_distance(self.points[i], self.points[j])
                row_dists.append(f"{dist:.2f}")
            # Join columns with a comma for easy pasting into Excel/CSV
            rows.append(",".join(row_dists))
        
        matrix_string = "\n".join(rows)

        # Push to system clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(matrix_string)
        self.root.update()  # Required on some OS to keep clipboard contents after function exits

        # Give the user visual feedback
        toast = self.canvas.create_text(
            self.canvas.winfo_width() / 2, 20, 
            text="Weight Matrix Copied to Clipboard!", 
            font=("Arial", 11, "bold"), fill="green"
        )
        self.root.after(2000, lambda: self.canvas.delete(toast))


    # --- Mouse Event Handlers ---
    def on_press(self, event):
        """Selects the point clicked by the user for dragging."""
        self.dragged_idx = None
        for i, (x, y) in enumerate(self.points):
            if math.hypot(event.x - x, event.y - y) <= self.point_radius + 5:
                self.dragged_idx = i
                break

    def on_drag(self, event):
        """Updates the point coordinate visually without re-solving the path."""
        if self.dragged_idx is not None:
            # Constrain dragging to the canvas boundaries
            new_x = max(10, min(event.x, self.canvas.winfo_width() - 10))
            new_y = max(10, min(event.y, self.canvas.winfo_height() - 10))
            self.points[self.dragged_idx] = (new_x, new_y)
            
            self.draw_dragging()

    def draw_dragging(self):
        """Renders points only during dragging."""
        self.canvas.delete("all")
        for i, (x, y) in enumerate(self.points):
            color = "green" if i == 0 else ("red" if i == NUM_POINTS - 1 else "dodgerblue")
            label = "Start" if i == 0 else ("End" if i == NUM_POINTS - 1 else f"P{i}")
            
            self.canvas.create_oval(
                x - self.point_radius, y - self.point_radius,
                x + self.point_radius, y + self.point_radius,
                fill=color, outline="black"
            )
            self.canvas.create_text(x, y - 18, text=label, font=("Arial", 9, "bold"))
            
        self.canvas.create_text(
            15, 15, anchor=tk.NW, text="Dragging... Release to calculate.", 
            font=("Arial", 12, "italic"), fill="gray"
        )

    def on_release(self, event):
        """Resolves the TSP math and redraws upon releasing the mouse."""
        if self.dragged_idx is not None:
            self.dragged_idx = None
            self.solve_and_draw()

    def on_right_click(self, event):
        """Toggles the 'all distances' view for the right-clicked node."""
        clicked_idx = None
        for i, (x, y) in enumerate(self.points):
            if math.hypot(event.x - x, event.y - y) <= self.point_radius + 5:
                clicked_idx = i
                break
                
        if clicked_idx is not None:
            # Toggle logic: if clicking the already selected node, turn it off. Otherwise, turn it on.
            if self.distance_view_idx == clicked_idx:
                self.distance_view_idx = None
            else:
                self.distance_view_idx = clicked_idx
            
            self.solve_and_draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPSolverApp(root)
    root.mainloop()