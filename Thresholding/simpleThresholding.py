import cv2
import time
import numpy as np

def simple_threshold(src: np.ndarray, thresh: float, maxval: float = 255,thresh_type: int = cv2.THRESH_BINARY) -> np.ndarray:
    """
    Apply simple (global) thresholding.

    Parameters
    ----------
    src : np.ndarray
        Input image (grayscale or color).
    thresh : float
        Threshold value.
    maxval : float
        Maximum value to assign.
    thresh_type : int
        Thresholding type (THRESH_BINARY, THRESH_BINARY_INV, etc.)

    Returns
    -------
    np.ndarray
        Thresholded image.
    """
    start_time = time.time()

    if src is None:
        raise ValueError("`src` is None.")
    
    if thresh >= maxval: 
        raise ValueError("thresh must be less then maxval(i.e 255)")

    if thresh < 0:
        raise ValueError("`thresh` must be non-negative.")

    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    total_pixels = gray.size
    fg_pixels = np.count_nonzero(gray)

    
    _, img_out = cv2.threshold(gray, thresh, maxval, thresh_type)
    end_time = time.time()
    print(f"Executed Time: {end_time - start_time:.6f}s")

    return img_out,fg_pixels / total_pixels


if __name__ == "__main__":
    img = cv2.imread("inputs/4.jpg")
    if img is None:
        raise SystemExit("Image not found")

    res,re = simple_threshold(img, thresh=150,maxval=200,thresh_type=cv2.THRESH_BINARY)

    print(re)
    cv2.imshow("Simple Threshold", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
