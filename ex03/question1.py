import numpy as np
# מימוש הפונקציה warp_image 
def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:

    H, W, C = image.shape

    # 1. מרכז התמונה (במערכת קואורדינטות רציפה)
    cx = W / 2.0
    cy = H / 2.0

    # 2. זווית ברדיאנים
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 3. מטריצת מתיחה
    S = np.array([
        [scale_x, 0,       0],
        [0,       scale_y, 0],
        [0,       0,       1]
    ], dtype=np.float32)

    # 4. מטריצת סיבוב סביב הראשית
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=np.float32)

    # קומפוזיציה: קודם מתיחה ואז סיבוב
    A = R @ S

    # 5. טרנסלציה למרכז וחזרה
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

    # מטריצת הטרנספורמציה הכוללת
    M = T2 @ A @ T1

    # 6. מטריצה הפוכה ל-backward mapping
    M_inv = np.linalg.inv(M)

    # 7. תמונת פלט
    output = np.zeros_like(image)

    # 8. לולאה על כל פיקסל ביציאה
    for i_out in range(H):
        for j_out in range(W):

            # מרכז הפיקסל ביציאה
            x_out = j_out + 0.5
            y_out = i_out + 0.5

            # מיפוי אחורה למקור
            x_src, y_src, _ = M_inv @ np.array([x_out, y_out, 1.0])

            # חזרה למערכת אינדקסים (i,j)
            x_src -= 0.5
            y_src -= 0.5

            # אינדקסים שלמים
            j0 = int(np.floor(x_src))
            i0 = int(np.floor(y_src))
            j1 = j0 + 1
            i1 = i0 + 1

            # אם מחוץ לגבולות — דולגים
            if i0 < 0 or i1 >= H or j0 < 0 or j1 >= W:
                continue

            # משקלים לאינטרפולציה ביליניארית
            dx = x_src - j0
            dy = y_src - i0

            w00 = (1 - dx) * (1 - dy)
            w01 = (1 - dx) * dy
            w10 = dx * (1 - dy)
            w11 = dx * dy

            # אינטרפולציה לכל ערוץ צבע
            for c in range(C):
                v00 = image[i0, j0, c]
                v01 = image[i1, j0, c]
                v10 = image[i0, j1, c]
                v11 = image[i1, j1, c]

                value = w00*v00 + w01*v01 + w10*v10 + w11*v11
                output[i_out, j_out, c] = np.clip(value, 0, 255)

    return output.astype(image.dtype)