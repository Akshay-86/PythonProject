import time
from typing import Optional
import cv2
import numpy as np

def channel_normalize(
    src: np.ndarray,
    alpha: float = 0.0,
    beta: float = 1.0,
    norm_type: int = cv2.NORM_MINMAX,
    dtype: Optional[int] = None,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Channel-wise normalization. Single function that auto-selects dtype
    when `dtype` is None, using `beta` as a hint:
        - beta == 1.0  -> CV_32F (float32)
        - beta == 255.0 -> CV_8U  (uint8)
        - otherwise: CV_32F if beta <= 1.0 else CV_8U

    Parameters
    ----------
    src : np.ndarray
        Input image (H, W) or (H, W, C).
    alpha : float
        Lower bound of normalized output range.
    beta : float
        Upper bound of normalized output range.
    norm_type : int
        OpenCV normalization type (cv2.NORM_MINMAX recommended).
    dtype : Optional[int]
        OpenCV dtype flag (cv2.CV_32F, cv2.CV_8U, ...). If None, auto-selected.
    mask : Optional[np.ndarray]
        Optional single-channel mask (H, W) where non-zero pixels are normalized.

    Returns
    -------
    np.ndarray
        Normalized image with per-channel normalization applied.
    """
    start_time = time.time()

    if src is None:
        raise ValueError("Source image is None.")

    src = np.asarray(src)
    if src.size == 0:
        raise ValueError("Source image is empty.")

    # Validate mask
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape != src.shape[:2]:
            raise ValueError("Mask must be single-channel and match image HxW.")
        # cv2.normalize expects mask of type uint8
        if mask.dtype != np.uint8:
            # accept boolean or other numeric -> convert
            mask = (mask != 0).astype(np.uint8)

    # Auto dtype selection when not provided
    if dtype is None:
        if beta == 1.0:
            dtype = cv2.CV_32F
        elif beta == 255.0:
            dtype = cv2.CV_8U
        else:
            dtype = cv2.CV_32F if beta <= 1.0 else cv2.CV_8U

    # Prepare channels: always operate on float32 input to get stable scaling
    if src.ndim == 2:
        channels = [src.astype(np.float32)]
        single_channel = True
    else:
        # keep channel count same, convert channels to float32 for normalization
        channels = [c.astype(np.float32) for c in cv2.split(src)]
        single_channel = False

    normalized_channels = []
    for c in channels:
        # dst None -> create new array; pass dtype flag and optional mask
        dst = cv2.normalize(c, None, alpha, beta, norm_type, dtype, mask)
        normalized_channels.append(dst)

    # Merge or return single
    if single_channel:
        out = normalized_channels[0]
    else:
        out = cv2.merge(normalized_channels)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.6f} s")

    return out


if __name__ == "__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src)
    out = channel_normalize(img)
    
    cv2.imshow("",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()