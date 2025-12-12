import time
import cv2
import numpy as np
from typing import Optional

def normalize(src: np.ndarray, alpha: float = 0.0, beta: float = 1.0, norm_type: int = cv2.NORM_MINMAX, dtype: Optional[int] = None, mask: Optional[np.ndarray] = None) -> np.ndarray:
    start_time = time.time()
    """
    Normalizing images based on 0-1 and 0-255 normalization.

    Parameters:
        src: input image (H,W) or (H,W,C). Any numeric dtype.
        alpha: lower bound of output range (e.g. 0.0).
        beta: upper bound of output range (e.g. 1.0 or 255.0).
        norm_type: cv2.NORM_MINMAX recommended.
        dtype: OpenCV dtype flag (cv2.CV_32F, cv2.CV_8U, ...) or None for auto:
               - if None and beta <= 1.0 -> CV_32F
               - if None and beta > 1.0 -> CV_8U
        mask: optional mask for region-limited normalization (same HxW).

    Returns:
        normalized image (dtype chosen by `dtype` or auto).
    """
    if src is None:
        raise ValueError("Source image is None.")
    src = np.asarray(src)
    if mask is not None:
        if mask.shape != src.shape[:2]:
            raise ValueError("Mask must match image height/width.")
    # select output dtype
    if dtype is None:
        dtype = cv2.CV_32F if beta <= 1.0 else cv2.CV_8U

    # cv2.normalize accepts multi-channel images directly
    out_img = cv2.normalize(src, None, alpha, beta, norm_type, dtype, mask)
    end_time = time.time()
    print(f"Exicution Time: {end_time - start_time}")
    return out_img


def normalize_0_1(src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Convenience: returns float32 in [0.0, 1.0]."""
    return normalize(src, alpha=0.0, beta=1.0, dtype=cv2.CV_32F, mask=mask)


def normalize_0_255(src: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Convenience: returns uint8 in [0, 255]."""
    return normalize(src, alpha=0.0, beta=255.0, dtype=cv2.CV_8U, mask=mask)

if __name__ == "__main__":
    src = "inputs/7.jpg"
    img = cv2.imread(src)
    

    res = normalize_0_1(img)
    # cv2.putText(res,f"alpha:{alpha},beta:{beta},normtype:{norm_type}",(0,15),5,1,(0,0,255),1)
    cv2.imshow("",res)
    cv2.imwrite("test/normalization/0-1normalization/5.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()