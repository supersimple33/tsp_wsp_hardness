import tkinter as tk
import math
import itertools
import random

# 1. Define your exact mathematical points here
STARTING_POINTS = None

NUM_POINTS = 8
# --- Configuration Logic ---
if STARTING_POINTS is not None:
    NUM_POINTS = len(STARTING_POINTS)
else:
    # Generate NUM_POINTS random coordinates in a reasonable math range
    STARTING_POINTS = [(random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)) for _ in range(NUM_POINTS)]

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

        # 3. Scale mathematical coordinates to fit the GUI canvas
        self.points = []
        min_x = min(p[0] for p in STARTING_POINTS)
        max_x = max(p[0] for p in STARTING_POINTS)
        min_y = min(p[1] for p in STARTING_POINTS)
        max_y = max(p[1] for p in STARTING_POINTS)
        
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1

        scale_x = 550 / range_x
        scale_y = 350 / range_y
        uniform_scale = min(scale_x, scale_y) # Using the smaller scale guarantees it fits
        offset_x = 50 + (550 - (range_x * uniform_scale)) / 2
        offset_y = 400 - (350 - (range_y * uniform_scale)) / 2

        for x, y in STARTING_POINTS:
            # Map X and Y using the exact same multiplier
            screen_x = offset_x + (x - min_x) * uniform_scale
            screen_y = offset_y - (y - min_y) * uniform_scale 
            self.points.append((int(screen_x), int(screen_y)))

        self.point_radius = 8
        self.dragged_idx = None
        self.distance_view_idx = None

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
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

        if best_path:
            for i in range(NUM_POINTS - 1):
                p1 = self.points[best_path[i]]
                p2 = self.points[best_path[i+1]]

                self.canvas.create_line(p1, p2, fill="black", width=2, dash=(4, 2))

                dist = self.get_distance(p1, p2)
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2

                offset = 10
                self.canvas.create_rectangle(mx-12, my-18, mx+12, my-2, fill="white", outline="")
                self.canvas.create_text(mx, my - offset, text=f"{dist:.1f}", font=("Arial", 9, "bold"), fill="purple")

        for i, (x, y) in enumerate(self.points):
            if i == 0:
                color, label = "green", "Start"
            elif i == NUM_POINTS - 1:
                color, label = "red", "End"
            else:
                color, label = "dodgerblue", f"P{i}"
                
            outline_color = "orange" if i == self.distance_view_idx else "black"
            outline_width = 4 if i == self.distance_view_idx else 2

            self.canvas.create_oval(
                x - self.point_radius, y - self.point_radius,
                x + self.point_radius, y + self.point_radius,
                fill=color, outline=outline_color, width=outline_width
            )
            self.canvas.create_text(x, y - 18, text=label, font=("Arial", 9, "bold"))

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

        if self.distance_view_idx is not None:
            p1 = self.points[self.distance_view_idx]
            for i, p2 in enumerate(self.points):
                if i == self.distance_view_idx:
                    continue
                
                self.canvas.create_line(p1, p2, fill="orange", width=2, dash=(2, 4))
                dist = self.get_distance(p1, p2)
                mx = (p1[0] + p2[0]) / 2
                my = (p1[1] + p2[1]) / 2
                
                self.canvas.create_rectangle(mx-14, my-8, mx+14, my+8, fill="white", outline="orange", width=1)
                self.canvas.create_text(mx, my, text=f"{dist:.1f}", font=("Arial", 8, "bold"), fill="darkorange")

    def export_matrix_to_clipboard(self):
        """Calculates the NxN weight matrix and copies it to the clipboard."""
        rows = []
        for i in range(NUM_POINTS):
            row_dists = []
            for j in range(NUM_POINTS):
                dist = self.get_distance(self.points[i], self.points[j])
                row_dists.append(f"{dist:.2f}")
            rows.append(",".join(row_dists))
        
        matrix_string = "\n".join(rows)

        self.root.clipboard_clear()
        self.root.clipboard_append(matrix_string)
        self.root.update() 

        toast = self.canvas.create_text(
            self.canvas.winfo_width() / 2, 20, 
            text="Weight Matrix Copied to Clipboard!", 
            font=("Arial", 11, "bold"), fill="green"
        )
        self.root.after(2000, lambda: self.canvas.delete(toast))

    def on_press(self, event):
        self.dragged_idx = None
        for i, (x, y) in enumerate(self.points):
            if math.hypot(event.x - x, event.y - y) <= self.point_radius + 5:
                self.dragged_idx = i
                break

    def on_drag(self, event):
        if self.dragged_idx is not None:
            new_x = max(10, min(event.x, self.canvas.winfo_width() - 10))
            new_y = max(10, min(event.y, self.canvas.winfo_height() - 10))
            self.points[self.dragged_idx] = (new_x, new_y)
            self.draw_dragging()

    def draw_dragging(self):
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
        if self.dragged_idx is not None:
            self.dragged_idx = None
            self.solve_and_draw()

    def on_right_click(self, event):
        clicked_idx = None
        for i, (x, y) in enumerate(self.points):
            if math.hypot(event.x - x, event.y - y) <= self.point_radius + 5:
                clicked_idx = i
                break
                
        if clicked_idx is not None:
            if self.distance_view_idx == clicked_idx:
                self.distance_view_idx = None
            else:
                self.distance_view_idx = clicked_idx
            
            self.solve_and_draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPSolverApp(root)
    root.mainloop()