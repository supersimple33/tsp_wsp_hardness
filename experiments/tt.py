import matplotlib.pyplot as plt

points = [
    (0.255691, 0.152402),
    (-0.242313, 0.038200),
    (0.281690, 0.080247),
    (-0.124978, -0.764436),
    (0.719447, -0.235692),
    (-0.089001, 0.232604),
    (0.058391, 0.211552),
    (0.162232, 0.178466),
    (-0.265936, -0.103417),
    (-0.211332, 0.128331),
]

x_vals = [p[0] for p in points]
y_vals = [p[1] for p in points]

plt.figure()
plt.scatter(x_vals, y_vals)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot of Provided Coordinates")
plt.show()
