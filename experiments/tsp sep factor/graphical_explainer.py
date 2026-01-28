import numpy as np
import matplotlib.pyplot as plt

# Grid resolution (increase for smoother boundary)
N = 2000

x = np.linspace(-1.5, 1.5, N)
y = np.linspace(-1.2, 1.2, N)
X, Y = np.meshgrid(x, y)

# Constraints
mask = (
    (X**2 + Y**2 <= 1.0)  # disk centered at (0,0)
    & (X >= 0.5)  # half-plane
    & ((X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 1.0)  # disk centered at (0.5, 0.5)
    & ((X - 0.5) ** 2 + (Y + 0.5) ** 2 <= 1.0)  # disk centered at (0.5, -0.5)
)

mask2 = (
    (X**2 + Y**2 <= 1.0)  # disk centered at (0,0)
    & (X <= -0.5)  # half-plane
    & ((X + 0.5) ** 2 + (Y - 0.5) ** 2 <= 1.0)  # disk centered at (0.5, 0.5)
    & ((X + 0.5) ** 2 + (Y + 0.5) ** 2 <= 1.0)  # disk centered at (0.5, -0.5)
)

plt.figure(figsize=(6, 6))
plt.scatter(X[mask], Y[mask], s=1, alpha=0.6)
plt.scatter(X[mask2], Y[mask2], s=1, alpha=0.6)

# Optional: draw circle boundaries for clarity
theta = np.linspace(0, 2 * np.pi, 1000)
plt.plot(np.cos(theta), np.sin(theta), color="g", alpha=0.5)
plt.plot(0.5 + np.cos(theta), 0.5 + np.sin(theta), "k--", alpha=0.5)
plt.plot(0.5 + np.cos(theta), -0.5 + np.sin(theta), "k--", alpha=0.5)

plt.axvline(0.5, color="r", linestyle="--", alpha=0.6)

plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Feasible region")
plt.show()
