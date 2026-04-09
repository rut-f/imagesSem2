import numpy as np

def warp_image_bilinear(image: np.ndarray,
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
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float32)
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float32)

    M = T2 @ A @ T1
    M_inv = np.linalg.inv(M)

    # גריד של כל פיקסלי היציאה
    j_out, i_out = np.meshgrid(np.arange(W), np.arange(H))

    x_out = j_out + 0.5
    y_out = i_out + 0.5

    pts = np.stack([x_out, y_out, np.ones_like(x_out)], axis=-1)

    # מיפוי אחורה
    src = pts @ M_inv.T
    x_src = src[..., 0] - 0.5
    y_src = src[..., 1] - 0.5

    # אינדקסים שלמים
    j0 = np.floor(x_src).astype(int)
    i0 = np.floor(y_src).astype(int)
    j1 = j0 + 1
    i1 = i0 + 1

    # משקלים
    dx = x_src - j0
    dy = y_src - i0

    w00 = (1 - dx) * (1 - dy)
    w01 = (1 - dx) * dy
    w10 = dx * (1 - dy)
    w11 = dx * dy

    # מסכה של פיקסלים חוקיים
    mask = (i0 >= 0) & (i1 < H) & (j0 >= 0) & (j1 < W)

    # פלט
    output = np.zeros_like(image, dtype=np.float32)

    # אינטרפולציה וקטורית לכל הערוצים
    for c in range(C):
        channel = image[..., c]

        v00 = channel[i0, j0]
        v01 = channel[i1, j0]
        v10 = channel[i0, j1]
        v11 = channel[i1, j1]

        out_c = w00*v00 + w01*v01 + w10*v10 + w11*v11
        output[..., c] = out_c

    # אפס פיקסלים מחוץ לתחום
    output[~mask] = 0

    return np.clip(output, 0, 255).astype(image.dtype)
