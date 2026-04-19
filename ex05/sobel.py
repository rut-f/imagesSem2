import sys
import os
import numpy as np
from scipy import ndimage
from PIL import Image


def normalize_to_uint8(arr):
    """
    מקבלת מערך float ומחזירה תמונה מנורמלת ל‑0..255 מסוג uint8
    """
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype(np.uint8)
    return arr


def main():
    if len(sys.argv) != 2:
        print("Usage: python sobel.py image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # טעינת התמונה
    img = Image.open(image_path).convert("L")  # המרה לגווני אפור
    img_np = np.array(img, dtype=np.float32)

    # מסנני Sobel
    gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    gy_kernel = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # חישוב gx ו‑gy
    gx = ndimage.convolve(img_np, gx_kernel, mode='reflect')
    gy = ndimage.convolve(img_np, gy_kernel, mode='reflect')

    # ערך מוחלט
    gx_abs = np.abs(gx)
    gy_abs = np.abs(gy)

    # גודל הגרדיאנט
    magnitude = np.sqrt(gx**2 + gy**2)

    # נורמליזציה
    gx_norm = normalize_to_uint8(gx_abs)
    gy_norm = normalize_to_uint8(gy_abs)
    mag_norm = normalize_to_uint8(magnitude)

    # שמות קבצים
    base, ext = os.path.splitext(image_path)
    gray_path = f"{base}_grayscale{ext}"
    gx_path = f"{base}_gx{ext}"
    gy_path = f"{base}_gy{ext}"
    mag_path = f"{base}_magnitude{ext}"

    # שמירת תמונות
    img.save(gray_path)
    Image.fromarray(gx_norm).save(gx_path)
    Image.fromarray(gy_norm).save(gy_path)
    Image.fromarray(mag_norm).save(mag_path)

    print("Saved:")
    print(gray_path)
    print(gx_path)
    print(gy_path)
    print(mag_path)


if __name__ == "__main__":
    main()
