import numpy as np

def warp_image_nn(image: np.ndarray,
                  angle_deg: float,
                  scale_x: float,
                  scale_y: float) -> np.ndarray:

    H, W, C = image.shape

    # מרכז התמונה
    cx = W / 2.0
    cy = H / 2.0

    # זווית ברדיאנים
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # מטריצת מתיחה
    S = np.array([
        [scale_x, 0,       0],
        [0,       scale_y, 0],
        [0,       0,       1]
    ], dtype=np.float32)

    # מטריצת סיבוב
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=np.float32)

    A = R @ S

    # טרנסלציה למרכז וחזרה
    T1 = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)

    T2 = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # מטריצת טרנספורמציה כוללת
    M = T2 @ A @ T1
    M_inv = np.linalg.inv(M)

    # יצירת גריד של כל פיקסלי היציאה
    j_out, i_out = np.meshgrid(np.arange(W), np.arange(H))

    # מרכז הפיקסל
    x_out = j_out + 0.5
    y_out = i_out + 0.5

    # הפיכת הגריד לוקטורים
    ones = np.ones_like(x_out)
    pts = np.stack([x_out, y_out, ones], axis=-1)  # shape (H, W, 3)

    # החלת מטריצת ה־backward mapping על כל הפיקסלים בבת אחת
    src = pts @ M_inv.T

    x_src = src[..., 0] - 0.5
    y_src = src[..., 1] - 0.5

    # אינטרפולציית nearest neighbor
    j_nn = np.round(x_src).astype(int)
    i_nn = np.round(y_src).astype(int)

    # מסכה של פיקסלים שנמצאים בתוך גבולות התמונה
    mask = (i_nn >= 0) & (i_nn < H) & (j_nn >= 0) & (j_nn < W)

    # תמונת פלט
    output = np.zeros_like(image)

    # העתקת ערכים רק עבור פיקסלים חוקיים
    output[mask] = image[i_nn[mask], j_nn[mask]]

    return output
