import numpy as np

# ב
def translation_matrix(a, b):
    return np.array([
        [1, 0, a],
        [0, 1, b],
        [0, 0, 1]
    ], dtype=np.float32)

# ג
def rotation_matrix(theta):
    # המרה למעלות → רדיאנים
    rad = np.deg2rad(theta)

    c = np.cos(rad)
    s = np.sin(rad)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

# ד
def scale_matrix(sx, sy=None):
    # אם sy לא ניתן → uniform scale
    if sy is None:
        sy = sx

    return np.array([
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1]
    ], dtype=np.float32)

# ה
def rotation_around_point_30():
    px, py = 100, 200
    theta = 30

    # מטריצת הזזה
    T1 = np.array([
        [1, 0, -px],
        [0, 1, -py],
        [0, 0,  1]
    ], dtype=np.float32)

    # מטריצת סיבוב
    rad = np.deg2rad(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    # הזזה חזרה
    T2 = np.array([
        [1, 0, px],
        [0, 1, py],
        [0, 0, 1]
    ], dtype=np.float32)

    return T2 @ R @ T1