import cv2
import time
import numpy as np
from typing import Tuple


def sobel_edge(src: np.ndarray, ddepth: int = cv2.CV_64F, dx: int = 1, dy: int = 1, ksize: int = 3, scale: float = 1.0, delta: float = 0.0, borderType: int = cv2.BORDER_DEFAULT) -> np.ndarray:
    """
    Apply Sobel edge detection.

    Parameters
    ----------
    src : np.ndarray
        Input image (grayscale or color).
    ddepth : int 
        The desired depth of the output image. 
    dx : int
        Order of derivative in x direction.
    dy : int
        Order of derivative in y direction.
    ksize : int
        Size of the extended Sobel kernel (must be odd).
    scale : float
        Optional scale factor for the computed derivative values.
    delta : float
        Optional delta added to the result.
    borderType : int
        Pixel extrapolation method.

    Returns
    -------
    np.ndarray
        Sobel edge magnitude image.

    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    start = time.time()

    if src is None:
        raise ValueError("`src` is None")

    if dx < 0 or dy < 0:
        raise ValueError("dx and dy must be >= 0")

    if dx == 0 and dy == 0:
        raise ValueError("At least one of dx or dy must be > 0")

    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be positive and odd")

    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src


    grad_x = None
    grad_y = None

    # Compute only requested derivatives
    if dx > 0:
        grad_x = cv2.Sobel(
            gray, ddepth, dx, 0,
            ksize=ksize, scale=scale, delta=delta, borderType=borderType
        )

    if dy > 0:
        grad_y = cv2.Sobel(
            gray, ddepth, 0, dy,
            ksize=ksize, scale=scale, delta=delta, borderType=borderType
        )

    # Combine results correctly
    if grad_x is not None and grad_y is not None:
        magnitude = cv2.magnitude(
            grad_x.astype(np.float32),
            grad_y.astype(np.float32)
        )
    elif grad_x is not None:
        magnitude = np.abs(grad_x)
    else:
        magnitude = np.abs(grad_y)

    magnitude = cv2.convertScaleAbs(magnitude)

    print(f"Executed Time: {time.time() - start:.6f}s")
    return magnitude


if __name__ == "__main__":
    src_path = "inputs/4.jpg"
    img = cv2.imread(src_path)


    edges = sobel_edge(img, dx=1,dy=2,ksize=3)

    cv2.imshow("Sobel Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.Sobel()