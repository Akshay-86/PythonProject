import time
from typing import Optional
import numpy as np
import cv2

def nonlocalmean_gray(src: np.ndarray, dst: Optional[np.ndarray] = None, h: float = 3.0,templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    start_time = time.time()
    """
    Denoise a single-channel (grayscale) image using OpenCV fastNlMeansDenoising.

    Parameters
    ----------
    src : np.ndarray
        2D grayscale image (H x W).
    dst : np.ndarray or None
        Optional output buffer.
    h : float
        Filter strength for luminance.
    templateWindowSize : int
        Odd positive integer (default 7).
    searchWindowSize : int
        Odd positive integer (default 21).

    Returns
    -------
    np.ndarray
        Denoised grayscale image.
    """
    if not isinstance(src, np.ndarray):
        raise TypeError("src must be a numpy.ndarray.")
    if src.ndim != 2:
        raise TypeError("nonlocalmean_gray expects a single-channel (grayscale) image.")

    if templateWindowSize <= 0 or templateWindowSize % 2 == 0:
        raise ValueError("templateWindowSize must be a positive odd integer.")
    if searchWindowSize <= 0 or searchWindowSize % 2 == 0:
        raise ValueError("searchWindowSize must be a positive odd integer.")
    
    out_img = cv2.fastNlMeansDenoising(src, dst=dst, h=h, templateWindowSize=templateWindowSize,searchWindowSize=searchWindowSize)
    end_time = time.time()
    print(f"Exicution Time: {end_time - start_time}")
    return out_img


def nonlocalmean_color(src: np.ndarray, dst: Optional[np.ndarray] = None, h: float = 3.0,hColor: float = 3.0, templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
    start_time = time.time()
    """
    Denoise a color image using OpenCV fastNlMeansDenoisingColored.

    Parameters
    ----------
    src : np.ndarray
        3-channel (BGR) image (H x W x 3) or 4-channel (B G R A).
    dst : np.ndarray or None
        Optional output buffer.
    h : float
        Filter strength for luminance.
    hColor : float
        Filter strength for color components.
    templateWindowSize : int
        Odd positive integer (default 7).
    searchWindowSize : int
        Odd positive integer (default 21).

    Returns
    -------
    np.ndarray
        Denoised color image.
    """
    if not isinstance(src, np.ndarray):
        raise TypeError("src must be a numpy.ndarray.")
    if src.ndim != 3 or src.shape[2] < 3:
        raise TypeError("nonlocalmean_color expects a color image with at least 3 channels (BGR).")

    if templateWindowSize <= 0 or templateWindowSize % 2 == 0:
        raise ValueError("templateWindowSize must be a positive odd integer.")
    if searchWindowSize <= 0 or searchWindowSize % 2 == 0:
        raise ValueError("searchWindowSize must be a positive odd integer.")
    
    out_img = cv2.fastNlMeansDenoisingColored(src, dst=dst, h=h, hColor=hColor,templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
    end_time = time.time()
    print(f"Exicution Time: {end_time - start_time}")
    return out_img

    

if __name__=="__main__":
    src  = "inputs/1.jpg"
    img = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    dst = np.array([])
    h = 10
    hColor = 10
    templateWindowSize = 7
    searchWindowSize = 21

    res = nonlocalmean_gray(img,h=h)
    cv2.putText(res,f"h: {h},hColor: {hColor},templateWindowSize:{templateWindowSize},searchWindowSize:{searchWindowSize}",(0,15),5,1,(0,0,255))
    cv2.imshow("",res)
    cv2.imwrite("test/noice_reduction/nonLocalMean_filter/1.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()