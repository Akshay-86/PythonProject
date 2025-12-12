import cv2
import numpy as np 
import time
from typing import Optional
from brisque import BRISQUE

def non_local_filter(src: np.ndarray, dst: Optional[np.ndarray] = None, h: float = 3.0,hColor: float = 3.0, templateWindowSize: int = 7, searchWindowSize: int = 21) -> np.ndarray:
        
        """
        Denoise a image using OpenCV fastNlMeansDenoising.

        Parameters
        ----------
        src : np.ndarray
            2D or 3D image (H x W).
        dst : np.ndarray or None
            Optional output buffer.
        h : float
            Filter strength for luminance.
        hColor : float
            Filter strength for color components.needed when color images.
        templateWindowSize : int
            Odd positive integer (default 7).
        searchWindowSize : int
            Odd positive integer (default 21).

        Returns
        -------
        np.ndarray
            Denoised grayscale image.
        """
        start_time = time.time()


        if not isinstance(src, np.ndarray):
            raise TypeError("src must be a numpy.ndarray.")
        if templateWindowSize <= 0 or templateWindowSize % 2 == 0:
            raise ValueError("templateWindowSize must be a positive odd integer.")
        if searchWindowSize <= 0 or searchWindowSize % 2 == 0:
            raise ValueError("searchWindowSize must be a positive odd integer.")
        
        if src.ndim == 2 :
            out_img = cv2.fastNlMeansDenoising(src, dst=dst, h=h, templateWindowSize=templateWindowSize,searchWindowSize=searchWindowSize)
        if src.ndim == 3:
            out_img=cv2.fastNlMeansDenoisingColored(src, dst=dst, h=h, hColor=hColor,templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        
        end_time = time.time()

        print(f'Execution Time: {end_time-start_time}')

        return out_img

if __name__=="__main__":
    src  = "inputs/3.jpg"
    img = cv2.imread(src)
    dst = np.array([])
    h = 10
    hColor = 10
    templateWindowSize = 7
    searchWindowSize = 21
    res = non_local_filter(img,h=h,hColor=hColor)
    cv2.imshow("",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()