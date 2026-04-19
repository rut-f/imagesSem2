import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import correlate2d


# א. יצירת kernel בגודל 3x3 מסוג float32
def initialize_kernel():
    kernel = np.array([
        [-1,  2,  1],
        [-2,  1, -3],
        [ 3,  0, -1]
    ], dtype=np.float32)
    return kernel


# ב. יצירת תמונה בגודל 4x4 מסוג uint8
def get_image():
    image = np.array([
        [103, 102, 101, 100],
        [104, 103, 102, 101],
        [ 53,  52,  51,  50],
        [ 45,  53,  52,  51]
    ], dtype=np.uint8)
    return image


# ג. cross-correlation באמצעות לולאות Python
def cross_correlate_loop(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape

    out_h = H - kH + 1
    out_w = W - kW + 1

    result = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i:i+kH, j:j+kW]
            result[i, j] = np.sum(patch * kernel)

    return result


# ד. cross-correlation באמצעות numpy + sliding_window_view
def cross_correlate_np(image, kernel):
    windows = sliding_window_view(image, kernel.shape)
    # windows shape: (2,2,3,3)

    multiplied = windows * kernel
    result = np.sum(multiplied, axis=(2, 3), dtype=np.float32)

    return result.astype(np.float32)


# ה. cross-correlation באמצעות scipy
def cross_correlate_scipy(image, kernel):
    result = correlate2d(image, kernel, mode='valid')
    return result.astype(np.float32)


# ו. השוואת שלושת התוצאות
def compare_cross_correlations():
    image = get_image()
    kernel = initialize_kernel()

    r1 = cross_correlate_loop(image, kernel)
    r2 = cross_correlate_np(image, kernel)
    r3 = cross_correlate_scipy(image, kernel)

    return np.allclose(r1, r2) and np.allclose(r1, r3)
