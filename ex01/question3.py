import numpy as np
import matplotlib.pyplot as plt
import math

# --- פונקציות עזר ---

def degrees_to_radians(deg):
    return deg * math.pi / 180

def rotation_matrix(deg):
    theta = degrees_to_radians(deg)
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

def scale_matrix(sx, sy):
    return np.array([
        [sx, 0],
        [0, sy]
    ])

# --- א. מטריצת סיבוב ב־30 מעלות ---
r_30 = rotation_matrix(30)
print("r_30 =\n", r_30)

# --- ב. מטריצת scale פי 2 בציר x ---
sx_2 = scale_matrix(2, 1)
print("sx_2 =\n", sx_2)

# --- ג. rs = r_30 @ sx_2 ---
rs = r_30 @ sx_2
print("rs =\n", rs)

# --- ד. sr = sx_2 @ r_30 ---
sr = sx_2 @ r_30
print("sr =\n", sr)

# --- ה. ציור מלבן מרכזי ברוחב 2 וגובה 1 ---

# נקודות המלבן (בכיוון השעון)
rectangle = np.array([
    [-1, -0.5],
    [ 1, -0.5],
    [ 1,  0.5],
    [-1,  0.5],
    [-1, -0.5]   # סוגרים את הצורה
]).T  # טרנספוזה כדי לקבל 2×N

# --- ו. סיבוב המלבן ב־30 מעלות ---
rect_rotated = r_30 @ rectangle

# --- ז. מתיחה פי 2 בציר x ---
rect_scaled = sx_2 @ rectangle

# --- ח. הפעלת sr ו־rs ---
rect_sr = sr @ rectangle
rect_rs = rs @ rectangle

# --- ט. ציור כל 5 המלבנים ---

plt.figure(figsize=(8, 8))
plt.axis('equal')

# מלבן מקורי
plt.plot(rectangle[0], rectangle[1], label="Original")

# סיבוב
plt.plot(rect_rotated[0], rect_rotated[1], label="Rotated 30°")

# מתיחה
plt.plot(rect_scaled[0], rect_scaled[1], label="Scaled x2")

# sr
plt.plot(rect_sr[0], rect_sr[1], label="sr = sx_2 @ r_30")

# rs
plt.plot(rect_rs[0], rect_rs[1], label="rs = r_30 @ sx_2")

plt.legend()
plt.title("Rectangle Transformations")
plt.grid(True)
plt.show()