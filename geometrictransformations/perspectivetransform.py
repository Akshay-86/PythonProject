import cv2
import time
import numpy as np
from typing import Tuple


def perspective_transform(
    src: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    dsize: Tuple[int, int],
    flags: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Apply a perspective (projective) transformation to an image.

    Parameters
    ----------
    src : np.ndarray
        Input image (must not be None).
    src_points : np.ndarray
        Source points (4x2) in the input image.
        Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    dst_points : np.ndarray
        Destination points (4x2) defining output mapping.
    dsize : tuple[int, int]
        Size of the output image (width, height).
    flags : int
        Interpolation method (cv2.INTER_LINEAR, INTER_CUBIC, etc.).
    borderMode : int
        Pixel extrapolation method.
    borderValue : tuple
        Value used for constant border (B, G, R).

    Returns
    -------
    np.ndarray
        Perspective-transformed image.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """

    start_time = time.time()
    # Validation
    if src is None:
        raise ValueError("`src` is None — image not loaded.")

    if not isinstance(dsize, tuple) or len(dsize) != 2:
        raise ValueError("`dsize` must be a tuple (width, height).")

    if dsize[0] <= 0 or dsize[1] <= 0:
        raise ValueError("`dsize` values must be positive.")

    src_points = np.asarray(src_points, dtype=np.float32)
    dst_points = np.asarray(dst_points, dtype=np.float32)

    if src_points.shape != (4, 2):
        raise ValueError("`src_points` must have shape (4, 2).")

    if dst_points.shape != (4, 2):
        raise ValueError("`dst_points` must have shape (4, 2).")

   
    # Compute perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    out_img = cv2.warpPerspective(
        src,
        M,
        dsize,
        flags=flags,
        borderMode=borderMode,
        borderValue=borderValue,
    )

    end_time = time.time()
    print(f"Executed Time: {end_time-start_time:.6f} s")

    return out_img


if __name__ == "__main__":
    src_path = "inputs/1.jpg"
    img = cv2.imread(src_path)

    if img is None:
        raise SystemExit(f"Error: couldn't load image from {src_path}")

    h, w = img.shape[:2]

    # Example source points (corners of a tilted document)
    src_pts = np.array([
        [200, 30],
        [850, 30],
        [1000, 300],
        [100, 300]
    ], dtype=np.float32)

    # Destination points (perfect rectangle)
    dst_pts = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    result = perspective_transform(
        img,
        src_points=src_pts,
        dst_points=dst_pts,
        dsize=(w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    cv2.imshow("Original", img)
    cv2.imshow("Perspective Transformed", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
