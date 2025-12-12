import cv2
import time
import numpy as np
from typing import Optional, Tuple


def rotate(
    src: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    dst: Optional[np.ndarray] = None,
    flags: int = cv2.INTER_LINEAR,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Rotate an image by a given angle around a center point.

    Parameters
    ----------
    src : np.ndarray
        Input image (must not be None).
    angle : float
        Angle in degrees. Positive values rotate counter-clockwise.
    center : tuple[float, float] or None
        (cx, cy) center of rotation in pixels. If None, image center is used.
    scale : float
        Isotropic scale factor applied after rotation. Must be > 0.
    dst : np.ndarray or None
        Optional destination image. Typically None.
    flags : int
        Interpolation flags (e.g., cv2.INTER_LINEAR, INTER_CUBIC).
    borderMode : int
        Pixel extrapolation method (e.g., cv2.BORDER_CONSTANT, BORDER_REFLECT).
    borderValue : tuple
        Value used in case of a constant border (B, G, R).

    Returns
    -------
    np.ndarray
        Rotated image.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """
    if src is None:
        raise ValueError("`src` is None — image not loaded or wrong argument passed.")

    if not isinstance(angle, (int, float)):
        raise ValueError("`angle` must be a number (int or float).")

    if not (isinstance(scale, (int, float)) and scale > 0):
        raise ValueError("`scale` must be a positive number.")

    h, w = src.shape[:2]

    # determine rotation center
    if center is None:
        cx, cy = (w / 2.0, h / 2.0)
    else:
        if not (isinstance(center, tuple) and len(center) == 2):
            raise ValueError("`center` must be a tuple (cx, cy) or None.")
        cx, cy = float(center[0]), float(center[1])

    # build rotation matrix
    start_time = time.time()
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # compute bounding box to avoid cropping (optional: keep same size here)
    # If you want to keep whole rotated image without cropping, compute new size.
    # For simplicity we'll keep original image size (w, h) — same as many cv2 apps.
    rotated = cv2.warpAffine(
        src,
        M,
        (w, h),
        dst=dst,
        flags=flags,
        borderMode=borderMode,
        borderValue=borderValue,
    )
    end_time = time.time()
    print(f"Executed Time: {end_time-start_time:.6f} s")

    return rotated


if __name__ == "__main__":
    src_path = "inputs/3.jpg"
    img = cv2.imread(src_path)
    if img is None:
        raise SystemExit(f"Error: couldn't load image from {src_path}")

    # Example usage:
    # rotate 30 degrees counter-clockwise about center, keep same size
    res = rotate(img, angle=90.0, scale=1.0, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    cv2.imshow("Rotated", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
