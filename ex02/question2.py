import numpy as np
import matplotlib.pyplot as plt

# פונקציה ליצירת מטריצת סיבוב (במעלות)
def rotation_matrix(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

# פונקציה ליצירת מטריצת מתיחה
def scale_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

# ציור צורה
def draw_shape(points, label, color):
    plt.plot(points[0, :], points[1, :], color=color, label=label)

# מלבן ברוחב 2 וגובה 1 שמרכזו בראשית
# נקודות המלבן (סגור)
rectangle = np.array([
    [-1,  1,  1, -1, -1],   # x
    [-0.5, -0.5, 0.5, 0.5, -0.5],  # y
    [1, 1, 1, 1, 1]         # הומוגני
])

# א. מלבן מקורי
rect_original = rectangle

# ב. סיבוב 30 מעלות
R30 = rotation_matrix(30)
rect_rot30 = R30 @ rectangle

# ג. סיבוב 45 מעלות ואז מתיחה פי 2 בציר x
R45 = rotation_matrix(45)
Sx2 = scale_matrix(2, 1)
rect_rot45_scale = Sx2 @ (R45 @ rectangle)

# ד. מתיחה פי 2 בציר x ואז סיבוב 45 מעלות
rect_scale_rot45 = R45 @ (Sx2 @ rectangle)

# ציור
plt.figure(figsize=(8, 8))
draw_shape(rect_original, "מקורי", "black")
draw_shape(rect_rot30, "סיבוב 30°", "red")
draw_shape(rect_rot45_scale, "סיבוב 45° ואז מתיחה", "blue")
draw_shape(rect_scale_rot45, "מתיחה ואז סיבוב 45°", "green")

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.title("טרנספורמציות של מלבן בקואורדינטות הומוגניות")
plt.grid(True)
plt.show()