import sys
import numpy as np

# פונקציה
def z(x, y):
    return np.sin(x) + np.sin(y)

# גרדיאנט
def grad_z(x, y):
    return np.array([np.cos(x), np.cos(y)])

def gradient_descent(x0, y0, learning_rate=0.1, num_iterations=1000):
    point = np.array([x0, y0], dtype=float)
    for _ in range(num_iterations):
        gradient = grad_z(point[0], point[1])
        point -= learning_rate * gradient
    return point

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gradient_descent_z.py x0 y0")
        sys.exit(1)

    x0 = float(sys.argv[1])
    y0 = float(sys.argv[2])

    print(f"Starting point:  x={x0:.4f},  y={y0:.4f},  z={z(x0,y0):.4f}")
    result = gradient_descent(x0, y0)
    print(f"Minimum found:   x={result[0]:.4f},  y={result[1]:.4f},  z={z(result[0],result[1]):.4f}")

import matplotlib.pyplot as plt

def gradient_descent_with_trace(x0, y0, learning_rate=0.1, num_iterations=1000):
    point = np.array([x0, y0], dtype=float)
    trace = [point.copy()]
    for _ in range(num_iterations):
        gradient = grad_z(point[0], point[1])
        point -= learning_rate * gradient
        trace.append(point.copy())
    return np.array(trace)

trace = gradient_descent_with_trace(x0, y0)
plt.plot(trace[:,0], trace[:,1], marker='o', markersize=2)
plt.title("Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
