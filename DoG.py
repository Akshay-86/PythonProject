import cv2
import time
import numpy as np

def difference_of_gaussian(src: np.ndarray, ksize1: tuple[int, int] = (5, 5), sigma1: float = 1.0, ksize2: tuple[int, int] = (9, 9), sigma2: float = 2.0, normalize: bool = True) -> np.ndarray:
    """
    Apply Difference of Gaussian (DoG) edge detection.

    Parameters
    ----------
    src : np.ndarray
        Input image (grayscale or color).
    ksize1 : tuple[int, int]
        Kernel size for first Gaussian blur (odd, positive).
    sigma1 : float
        Sigma for first Gaussian blur.
    ksize2 : tuple[int, int]
        Kernel size for second Gaussian blur (odd, positive).
    sigma2 : float
        Sigma for second Gaussian blur (must be > sigma1).
    normalize : bool
        If True, normalize output to 0-255.

    Returns
    -------
    np.ndarray
        DoG response image.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    start_time = time.time()

    # Validation
    if src is None:
        raise ValueError("`src` is None")

    if sigma1 <= 0 or sigma2 <= 0:
        raise ValueError("sigma values must be positive")

    if sigma2 <= sigma1:
        raise ValueError("sigma2 must be greater than sigma1")

    if ksize1[0] <= 0 or ksize1[0] % 2 == 0:
        raise ValueError("ksize1 must be positive and odd")

    if ksize2[0] <= 0 or ksize2[0] % 2 == 0:
        raise ValueError("ksize2 must be positive and odd")

    # Convert to grayscale
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    gray = gray.astype(np.float32)

    # DoG computation 

    blur1 = cv2.GaussianBlur(gray, ksize1, sigma1)
    blur2 = cv2.GaussianBlur(gray, ksize2, sigma2)

    out_img = blur1 - blur2

    if normalize:
        out_img = cv2.normalize(out_img, None, 0, 255, cv2.NORM_MINMAX)
        out_img = out_img.astype(np.uint8)
        
    end_time = time.time()
    print(f"Executed Time: {end_time - start_time:.6f}s")
    return out_img


if __name__ == "__main__":
    img = cv2.imread("inputs/1.jpg")
    if img is None:
        raise SystemExit("Image not found")

    dog_edges = difference_of_gaussian(
        img,
        ksize1=(5, 5),
        sigma1=1.0,
        ksize2=(9, 9),
        sigma2=2.0
    )

    cv2.imshow("Difference of Gaussian", dog_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
