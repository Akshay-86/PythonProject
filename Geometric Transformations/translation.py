import cv2
import time
import numpy as np

def translate(src: np.ndarray, tx: float, ty: float, dst: np.ndarray | None = None, flags: int = cv2.INTER_LINEAR, borderMode: int = cv2.BORDER_CONSTANT, borderValue: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Translate (shift) an image by tx, ty pixels.

    Parameters
    ----------
    src : np.ndarray
        Input image (must not be None).
    tx : float
        Horizontal translation in pixels. Positive shifts right.
    ty : float
        Vertical translation in pixels. Positive shifts down.
    dst : np.ndarray or None
        Optional destination image. Typically None.
    flags : int
        Interpolation flags (cv2.INTER_NEAREST, INTER_LINEAR, etc.).
    borderMode : int
        Pixel extrapolation method for empty areas.
    borderValue : tuple
        Value for constant border (B, G, R).

    Returns
    -------
    np.ndarray
        Translated image.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """
    start_time = time.time()

    if src is None:
        raise ValueError("src is None — image not loaded or wrong argument passed.")

    if not isinstance(tx, (int, float)) or not isinstance(ty, (int, float)):
        raise ValueError("tx and ty must be numbers (int or float).")

    h, w = src.shape[:2]

    # Construct affine translation matrix: [1 0 tx; 0 1 ty]
    M = np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)]], dtype=np.float32)

    out_img = cv2.warpAffine(
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

    return out_img


if __name__ == "__main__":
    src_path = "inputs/3.jpg"
    img = cv2.imread(src_path)
    if img is None:
        raise SystemExit(f"Error: couldn't load image from {src_path}")

    # Example usage:
    # shift right by 100 px and down by 50 px
    res = translate(img, tx=100, ty=50, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    cv2.imshow("Translated", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
