import tkinter as tk
import math
import itertools

NUM_POINTS = 8

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

        tk.Button(control_frame, text="Copy Weight Matrix", command=self.export_matrix_to_clipboard, bg="#e0e0e0").pack(side=tk.RIGHT, padx=10)

        # --- Interactive Canvas ---
        self.canvas = tk.Canvas(root, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pre-load the exact mathematical coordinates you provided
        self.points = [
            (-0.1951, 0.5742),  # Index 0 (Start)
            (0.4465, 0.9571),   # Index 1
            (-0.4828, -0.0679), # Index 2
            (0.5953, -0.4092),  # Index 3
            (1.7428, 0.3848),   # Index 4
            (-1.0539, 1.6879),  # Index 5
            (0.4866, -1.8278),  # Index 6
            (-1.8723, -0.5382)  # Index 7 (End)
        ]

        self.point_radius = 8
        self.dragged_idx = None
        self.distance_view_idx = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Button-2>", self.on_right_click)

        self.solve_and_draw()

    # --- Coordinate Mapping Helpers ---
    def logical_to_screen(self, lx, ly):
        """Converts true mathematical coordinates to screen pixels. Centers at 325, 225 with scale of 100."""
        sx = 325 + lx * 100
        sy = 225 - ly * 100 # Invert Y because screen Y goes down
        return (sx, sy)

    def screen_to_logical(self, sx, sy):
        """Converts screen pixels back to true mathematical coordinates."""
        lx = (sx - 325) / 100
        ly = (225 - sy) / 100
        return (lx, ly)

    def get_distance(self, p1, p2):
        """Calculates TRUE mathematical distance based on the currently selected mathematical norm."""
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
        start_idx = 0
        end_idx = NUM_POINTS - 1
        middle_indices = list(range(1, NUM_POINTS - 1))

        min_dist = float('inf')
        best_path = None

        for perm in itertools.permutations(middle_indices):
            path = [start_idx] + list(perm) + [end_idx]
            dist = sum(self.get_distance(self.points[path[i]], self.points[path[i+1]]) 
                       for i in range(NUM_POINTS - 1))

            if dist < min_dist:
                min_dist = dist
                best_path = path

        return best_path, min_dist

    def solve_and_draw(self):
        self.canvas.delete("all")
        best_path, min_dist = self.solve_tsp()

        # 1. Main Path
        if best_path:
            for i in range(NUM_POINTS - 1):
                p1_log = self.points[best_path[i]]
                p2_log = self.points[best_path[i+1]]
                
                # Draw using screen coordinates
                p1_scr = self.logical_to_screen(*p1_log)
                p2_scr = self.logical_to_screen(*p2_log)

                self.canvas.create_line(p1_scr, p2_scr, fill="black", width=2, dash=(4, 2))

                # Display true mathematical distance
                dist = self.get_distance(p1_log, p2_log)
                mx = (p1_scr[0] + p2_scr[0]) / 2
                my = (p1_scr[1] + p2_scr[1]) / 2

                self.canvas.create_rectangle(mx-16, my-18, mx+16, my-2, fill="white", outline="")
                self.canvas.create_text(mx, my - 10, text=f"{dist:.2f}", font=("Arial", 9, "bold"), fill="purple")

        # 2. Points
        for i, (lx, ly) in enumerate(self.points):
            sx, sy = self.logical_to_screen(lx, ly)
            
            if i == 0:
                color, label = "green", "Start"
            elif i == NUM_POINTS - 1:
                color, label = "red", "End"
            else:
                color, label = "dodgerblue", f"P{i}"
                
            outline_color = "orange" if i == self.distance_view_idx else "black"
            outline_width = 4 if i == self.distance_view_idx else 2

            self.canvas.create_oval(
                sx - self.point_radius, sy - self.point_radius,
                sx + self.point_radius, sy + self.point_radius,
                fill=color, outline=outline_color, width=outline_width
            )
            self.canvas.create_text(sx, sy - 18, text=label, font=("Arial", 9, "bold"))

        # 3. Header Texts
        metric_name = {"L1": "Manhattan", "L2": "Euclidean", "L_inf": "Chebyshev"}[self.norm_var.get()]
        self.canvas.create_text(
            15, 15, anchor=tk.NW,
            text=f"Shortest Path ({metric_name}): {min_dist:.4f}",
            font=("Arial", 12, "bold"), fill="darkred"
        )
        self.canvas.create_text(
            15, 35, anchor=tk.NW,
            text="Right-click a node to toggle all true distances",
            font=("Arial", 9, "italic"), fill="gray"
        )

        # 4. All Distances View
        if self.distance_view_idx is not None:
            p1_log = self.points[self.distance_view_idx]
            p1_scr = self.logical_to_screen(*p1_log)
            
            for i, p2_log in enumerate(self.points):
                if i == self.distance_view_idx:
                    continue
                
                p2_scr = self.logical_to_screen(*p2_log)
                self.canvas.create_line(p1_scr, p2_scr, fill="orange", width=2, dash=(2, 4))
                
                dist = self.get_distance(p1_log, p2_log)
                mx = (p1_scr[0] + p2_scr[0]) / 2
                my = (p1_scr[1] + p2_scr[1]) / 2
                
                self.canvas.create_rectangle(mx-16, my-8, mx+16, my+8, fill="white", outline="orange", width=1)
                self.canvas.create_text(mx, my, text=f"{dist:.2f}", font=("Arial", 8, "bold"), fill="darkorange")

    def export_matrix_to_clipboard(self):
        rows = []
        for i in range(NUM_POINTS):
            row_dists = []
            for j in range(NUM_POINTS):
                dist = self.get_distance(self.points[i], self.points[j])
                row_dists.append(f"{dist:.4f}")
            rows.append(",".join(row_dists))
        
        matrix_string = "\n".join(rows)
        self.root.clipboard_clear()
        self.root.clipboard_append(matrix_string)
        self.root.update()

        toast = self.canvas.create_text(
            self.canvas.winfo_width() / 2, 20, 
            text="True Weight Matrix Copied to Clipboard!", 
            font=("Arial", 11, "bold"), fill="green"
        )
        self.root.after(2000, lambda: self.canvas.delete(toast))

    def on_press(self, event):
        self.dragged_idx = None
        for i, (lx, ly) in enumerate(self.points):
            sx, sy = self.logical_to_screen(lx, ly)
            if math.hypot(event.x - sx, event.y - sy) <= self.point_radius + 5:
                self.dragged_idx = i
                break

    def on_drag(self, event):
        if self.dragged_idx is not None:
            new_x = max(10, min(event.x, self.canvas.winfo_width() - 10))
            new_y = max(10, min(event.y, self.canvas.winfo_height() - 10))
            
            # Save the new position as true mathematical coordinates
            self.points[self.dragged_idx] = self.screen_to_logical(new_x, new_y)
            self.draw_dragging()

    def draw_dragging(self):
        self.canvas.delete("all")
        for i, (lx, ly) in enumerate(self.points):
            sx, sy = self.logical_to_screen(lx, ly)
            color = "green" if i == 0 else ("red" if i == NUM_POINTS - 1 else "dodgerblue")
            label = "Start" if i == 0 else ("End" if i == NUM_POINTS - 1 else f"P{i}")
            
            self.canvas.create_oval(
                sx - self.point_radius, sy - self.point_radius,
                sx + self.point_radius, sy + self.point_radius,
                fill=color, outline="black"
            )
            self.canvas.create_text(sx, sy - 18, text=label, font=("Arial", 9, "bold"))
            
        self.canvas.create_text(15, 15, anchor=tk.NW, text="Dragging... Release to calculate.", font=("Arial", 12, "italic"), fill="gray")

    def on_release(self, event):
        if self.dragged_idx is not None:
            self.dragged_idx = None
            self.solve_and_draw()

    def on_right_click(self, event):
        clicked_idx = None
        for i, (lx, ly) in enumerate(self.points):
            sx, sy = self.logical_to_screen(lx, ly)
            if math.hypot(event.x - sx, event.y - sy) <= self.point_radius + 5:
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