import cv2
import time
import numpy as np

def crop(src: np.ndarray, start_point: tuple[int, int], end_point: tuple[int, int]) -> np.ndarray:
    """
    Crop a region from an image.

    Parameters
    ----------
    src : np.ndarray
        Input image (must not be None).
    start_point : tuple[int, int]
        (x1, y1) - Top-left corner of crop region.
    end_point : tuple[int, int]
        (x2, y2) - Bottom-right corner of crop region.

    Returns
    -------
    np.ndarray
        Cropped image.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    start_time = time.time()

    if src is None:
        raise ValueError("src is None — image not loaded.")

    if len(start_point) != 2 or len(end_point) != 2:
        raise ValueError("start_point and end_point must be tuples of (x, y).")

    x1, y1 = start_point
    x2, y2 = end_point

    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates (x2,y2 must be greater than x1,y1).")

    h, w = src.shape[:2]

    if x2 > w or y2 > h:
        raise ValueError(f"Crop region ({x2},{y2}) exceeds image size ({w},{h}).")

    #  Crop Operation
    out_img = src[y1:y2, x1:x2]

    end_time = time.time()
    print(f"Executed Time: {end_time-start_time:.6f} s")
    return out_img


if __name__ == "__main__":
    src_path = "inputs/2.jpg"
    img = cv2.imread(src_path)

    if img is None:
        raise SystemExit(f"Error: couldn't load image from {src_path}")

    # Example: crop
    # Cropping a region (x1=50, y1=50) → (x2=300, y2=300)
    result = crop(img, (50, 50), (300, 300))

    cv2.imshow("Cropped Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
