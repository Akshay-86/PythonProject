import cv2
import time
import numpy as np
from typing import Optional

def normalize(src: np.ndarray, alpha: float = 0.0, beta: float = 1.0, norm_type: int = cv2.NORM_MINMAX, dtype: int = None, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Performs 0-1, 0-255, or custom-range normalization depending on beta

    Parameters
    ----------
    src : np.ndarray
        Input image (H, W) or (H, W, C).
    alpha : float
        Lower bound of output range.
    beta : float
        Upper bound of output range.
    norm_type : int
        Normalization type (cv2.NORM_MINMAX recommended).
    dtype : Optional[int]
        Output dtype. If None → auto dtype selection.
    mask : Optional[np.ndarray]
        Mask (HxW) for region-limited normalization.

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    start_time = time.time()

    if src is None:
        raise ValueError("Source image is None.")

    src = np.asarray(src)
    if src.size == 0:
        raise ValueError("Input image is empty.")
    
    if alpha >= beta:
        raise ValueError("Eneter a valid alpha value")

    # Mask validation
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim != 2 or mask.shape != src.shape[:2]:
            raise ValueError("Mask must be single-channel and match image HxW.")
        if mask.dtype != np.uint8:
            mask = (mask != 0).astype(np.uint8)


    if dtype is None:
        if beta == 1.0:
            dtype = cv2.CV_32F
        elif beta == 255.0:
            dtype = cv2.CV_8U
        else:
            dtype = cv2.CV_32F if beta <= 1.0 else cv2.CV_8U

    # OpenCV allows direct multi-channel normalization
    out_img = cv2.normalize(src, None, alpha, beta, norm_type, dtype, mask)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.6f} s")

    return out_img

if __name__ == "__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src)
    

    res = normalize(img,beta=1)
    # cv2.putText(res,f"alpha:{alpha},beta:{beta},normtype:{norm_type}",(0,15),5,1,(0,0,255),1)
    cv2.imshow("",res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()