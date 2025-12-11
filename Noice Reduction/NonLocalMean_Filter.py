import os
from typing import List, Union
import numpy as np
import cv2

ImageLike = Union[np.ndarray, List[np.ndarray]]

def nonlocalmean(src: ImageLike, imgToDenoiseIndex: int = 0, temporalWindowSize: int = 0, dst: np.ndarray | None = None, h: float = 3.0, hColor: float = 3.0, templateWindowSize: int = 7,searchWindowSize: int = 21, sel: int = 0) -> np.ndarray:
    """
    Advanced blur technique that excels at reducing noise while preserving sharp edges.

    Parameters
    ----------
    src : ndarray or list of ndarrays
        If sel in {0,1} pass a single image ndarray.
        If sel in {2,3} pass a list of nearby frames (all same shape & dtype).
    imgToDenoiseIndex : int
        Index of target frame in list (used for sel 2 or 3).
    temporalWindowSize : int
        Number of frames used (must be odd and >=1) for sel 2 or 3.
    dst : ndarray or None
        Optional output buffer.
    h : float
        Filter strength for luminance.
    hColor : float
        Filter strength for color (for colored functions).
    templateWindowSize : int
        Odd positive int (default 7).
    searchWindowSize : int
        Odd positive int (default 21).
    sel : int
        0 -> fastNlMeansDenoising (grayscale), 
        1 -> fastNlMeansDenoisingColored (color), 
        2 -> fastNlMeansDenoisingMulti (grayscale list), 
        3 -> fastNlMeansDenoisingColoredMulti (color list).
    """
    # basic sel validation
    if not isinstance(sel, int) or sel < 0 or sel > 3:
        raise ValueError("sel must be an integer in [0,1,2,3].")

    # validate odd window sizes
    if templateWindowSize <= 0 or templateWindowSize % 2 == 0:
        raise ValueError("templateWindowSize must be a positive odd integer.")
    if searchWindowSize <= 0 or searchWindowSize % 2 == 0:
        raise ValueError("searchWindowSize must be a positive odd integer.")

    # sel 0: grayscale single image
    if sel == 0:
        if not (isinstance(src, np.ndarray) and src.ndim == 2):
            raise TypeError("sel=0 expects a single-channel (grayscale) ndarray.")
        return cv2.fastNlMeansDenoising(src, dst=dst, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

    # sel 1: colored single image
    if sel == 1:
        if not (isinstance(src, np.ndarray) and src.ndim == 3 and src.shape[2] in (3, 4)):
            raise TypeError("sel=1 expects a colored image ndarray with 3 or 4 channels.")
        return cv2.fastNlMeansDenoisingColored(src, dst=dst, h=h, hColor=hColor, templateWindowSize=templateWindowSize,searchWindowSize=searchWindowSize)

    # sel 2 or 3: multi-frame (list of images)
    if not isinstance(src, (list, tuple)) or len(src) == 0:
        raise TypeError("sel 2 and 3 require 'src' to be a non-empty list of images.")

    n_frames = len(src)
    # ensure all frames share shape and dtype
    first_shape = src[0].shape
    first_dtype = src[0].dtype
    for i, f in enumerate(src):
        if not isinstance(f, np.ndarray):
            raise TypeError(f"src[{i}] is not an ndarray.")
        if f.shape != first_shape:
            raise ValueError(f"All frames must have the same shape. src[0].shape={first_shape}, src[{i}].shape={f.shape}")
        if f.dtype != first_dtype:
            raise ValueError(f"All frames must have the same dtype. src[0].dtype={first_dtype}, src[{i}].dtype={f.dtype}")

    # temporalWindowSize validation
    if temporalWindowSize <= 0 or temporalWindowSize % 2 == 0:
        raise ValueError("temporalWindowSize must be an odd positive integer (e.g. 3,5,7).")
    if temporalWindowSize > n_frames:
        raise ValueError("temporalWindowSize cannot be greater than number of provided frames.")
    if not (0 <= imgToDenoiseIndex < n_frames):
        raise ValueError("imgToDenoiseIndex must be within the range of provided frames indices.")

    if sel == 2:
        # frames must be single-channel
        if src[0].ndim != 2:
            raise TypeError("sel=2 expects list of single-channel (grayscale) frames.")
        return cv2.fastNlMeansDenoisingMulti(src, imgToDenoiseIndex, temporalWindowSize, dst=dst, h=h, templateWindowSize=templateWindowSize,searchWindowSize=searchWindowSize)

    # sel == 3
    if src[0].ndim == 2:
        raise TypeError("sel=3 expects list of colored frames (3-channel).")
    return cv2.fastNlMeansDenoisingColoredMulti(src, imgToDenoiseIndex, temporalWindowSize, dst=dst, h=h, hColor=hColor,templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

    

if __name__=="__main__":
    img = cv2.imread("/home/robo/Documents/mouse.jpeg",cv2.IMREAD_GRAYSCALE)

    res = nonlocalmean(img,h=10,hColor=10,sel=0)

    cv2.imshow("",res)
    cv2.imwrite("test/Noice Reduction/NonLocalMean_Filter/h=10,hColor=10,sel=0.jpeg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()