import numpy as np
import time

# נניח שיש את הפונקציות:
# warp_image (עם לולאות)
# warp_image_nn (וקטורי NN)
# warp_image_bilinear (וקטורי ביליניארי)

def measure_time(func, image, *args): #פונקציה למדידת ביצועים
    #הקוד משתמש ב־time.perf_counter() כדי למדוד זמן ריצה בצורה מדויקת.
    start = time.perf_counter()
    func(image, *args)
    end = time.perf_counter()
    return end - start


sizes = [
    (200, 200),
    (400, 400),
    (800, 800),
    (1200, 1200),
]

angle = 30
sx = 1.2
sy = 0.8

results = []

for H, W in sizes:
    img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

    t_loop = measure_time(warp_image, img, angle, sx, sy)
    t_nn   = measure_time(warp_image_nn, img, angle, sx, sy)
    t_bi   = measure_time(warp_image_bilinear, img, angle, sx, sy)

    results.append((H, W, t_loop, t_nn, t_bi))

# הדפסת טבלה
print("Height | Width | Loop Time | NN Time | Bilinear Time")
for row in results:
    print(f"{row[0]:6d} | {row[1]:5d} | {row[2]:9.4f} | {row[3]:8.4f} | {row[4]:14.4f}")

