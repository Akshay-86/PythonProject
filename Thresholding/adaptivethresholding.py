import cv2
import time
import numpy as np


def adaptive_threshold(src: np.ndarray, maxval: float = 255, adaptiveMethod: int = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType: int = cv2.THRESH_BINARY, blockSize: int = 11, C: float = 2) -> np.ndarray:
    """
    Applying adaptive thresholding to calculate the threshold value is calculated for smaller regions.

    Parameters
    ----------
    src : np.ndarray
        Input image (grayscale or color).
    maxval : float
        Maximum value to assign.
    adaptiveMethod : int
        ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C.
    thresholdType : int
        THRESH_BINARY or THRESH_BINARY_INV.
    blockSize : int
        Size of neighbourhood (must be odd and >1).
    C : float
        Constant subtracted from the mean or weighted mean.

    Returns
    -------
    np.ndarray
        Adaptive thresholded image.
    """
    start_time = time.time()

    if src is None:
        raise ValueError("`src` is None.")
    

    if blockSize <= 1 or blockSize % 2 == 0:
        raise ValueError("`blockSize` must be odd and > 1.")

    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    out_img = cv2.adaptiveThreshold(
        gray,
        maxval,
        adaptiveMethod,
        thresholdType,
        blockSize,
        C
    )
    end_time = time.time()
    print(f"Executed Time: {end_time-start_time:.6f}s")

    return out_img


if __name__ == "__main__":
    img = cv2.imread("inputs/3.jpg")
    if img is None:
        raise SystemExit("Image not found")

    res = adaptive_threshold(
        img,
        adaptiveMethod=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    cv2.imshow("Adaptive Threshold", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
