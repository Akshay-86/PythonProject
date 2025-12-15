import cv2
import time
import numpy as np
from typing import Tuple


def canny_edge(src: np.ndarray, threshold1: float = None, threshold2: float = None, apertureSize: int = 3, L2gradient: bool = False) -> np.ndarray:
    """
    Apply Canny edge detection.

    Parameters
    ----------
    src : np.ndarray
        Input image (grayscale or color).
    threshold1 : float
        Lower threshold for hysteresis.
    threshold2 : float
        Upper threshold for hysteresis.
    apertureSize : int
        Aperture size for Sobel operator (odd, 3-7).
    L2gradient : bool
        If True, uses L2 norm for gradient magnitude.

    Returns
    -------
    np.ndarray
        Binary edge map.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """

    start_time = time.time()
    if src is None:
        raise ValueError("`src` is None — image not loaded.")
    
    if threshold1 == None or threshold2 == None:
        raise ValueError("Threshold1 and threshold2 are mandatary parameters ")

    if threshold1 <= 0 or threshold2 <= 0:
        raise ValueError("Thresholds must be positive.")

    if threshold2 <= threshold1:
        raise ValueError("threshold2 must be greater than threshold1.")

    if apertureSize not in (3, 5, 7):
        raise ValueError("`apertureSize` must be 3, 5, or 7.")

    # Convert to grayscale if needed
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    out_img = cv2.Canny(
        gray,
        threshold1=threshold1,
        threshold2=threshold2,
        apertureSize=apertureSize,
        L2gradient=L2gradient
    )

    end_time = time.time()
    print(f"Executed Time: {end_time - start_time:.6f} s")

    return out_img


if __name__ == "__main__":
    src_path = "inputs/1.jpg"
    img = cv2.imread(src_path)

    if img is None:
        raise SystemExit(f"Error: could not load image from {src_path}")

    im = canny_edge(img,threshold1=100,threshold2=200)

    cv2.imshow("Canny Edge Detection",im )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.Canny()