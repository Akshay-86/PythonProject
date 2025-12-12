import cv2
import time 
import numpy as np

def median_blur(src: np.ndarray,ksize: int)->np.ndarray:
    starttime = time.time()
    """
        Applying non-linear medianBlur filter

        Paramaters with * at end are mandatory parameters 
        
        Parameters:
            src(Imgae ndarray): Actual image you want to blur.*
            ksize(int): Single positive, odd integer.*
        
        Returns: 
            Image_ndarray: Result Blured Image

        Raises: 
            ValueError: if bad arguments recived
    """
    
    if src is None:
        raise ValueError("Source is NONE")
    if not ksize>0 or ksize%2==0:
        raise ValueError("ksize must be positive and odd number")
    
    outimg = cv2.medianBlur(src,ksize)
    endtime = time.time()
    print("Runtime: ",endtime-starttime)
    return outimg


if __name__ == "__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src)
    ksize = 3

    res = median_blur(img,ksize)
    cv2.putText(res,f"ksize:{ksize}",(0,15),5,1,(0,0,255),1)
    cv2.imshow("",res)
    cv2.imwrite(f"test/noice_reduction/median_filter/1.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()