import cv2
import time
import numpy as np
from typing import Optional

def channel_normalize(src: np.ndarray, alpha: float = 0.0, beta: float = 1.0, norm_type: int = cv2.NORM_MINMAX, dtype: Optional[int] = None, mask: Optional[np.ndarray] = None) -> np.ndarray:
    start_time = time.time()
    """
    Channel-wise normalization.

    Each channel is normalized independently to [alpha, beta].

    Parameters:
        src: Input image (H,W) or (H,W,C).
        alpha: Lower bound of normalized range.
        beta: Upper bound of normalized range.
        norm_type: cv2.NORM_MINMAX recommended.
        dtype: Output dtype. Auto-select if None:
               - if beta <= 1.0  → CV_32F
               - if beta > 1.0  → CV_8U
        mask: Optional mask (H,W) applied to all channels.

    Returns:
        Channel-wise normalized image.
    """

    if src is None:
        raise ValueError("Source image cannot be None.")

    src = np.asarray(src)

    # auto dtype selection
    if dtype is None:
        dtype = cv2.CV_32F if beta <= 1.0 else cv2.CV_8U

    # split channels safely for gray or color
    if src.ndim == 2:
        channels = [src]
    else:
        channels = cv2.split(src.astype(np.float32))

    normalized_channels = []

    for c in channels:
        dst = cv2.normalize(c, None, alpha, beta, norm_type, dtype, mask)
        normalized_channels.append(dst)

    # merge channels back
    out_img = normalized_channels[0] if len(normalized_channels) == 1 else cv2.merge(normalized_channels)

    end_time = time.time()
    print(f"Exicution Time: {end_time - start_time}")
    return out_img


def channel_norm_0_1(src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Channel-wise normalization to float32 range [0,1]."""
    return channel_normalize(src, alpha=0.0, beta=1.0, dtype=cv2.CV_32F, mask=mask)


def channel_norm_0_255(src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Channel-wise normalization to uint8 range [0,255]."""
    return channel_normalize(src, alpha=0.0, beta=255.0, dtype=cv2.CV_8U, mask=mask)


if __name__ == "__main__":
    src = "inputs/6.jpg"
    img = cv2.imread(src)
    out = channel_norm_0_255(img)

    cv2.imshow("",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()