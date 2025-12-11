import cv2
import numpy as np

def median_blur(src: np.ndarray,ksize: int)->np.ndarray:
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

    return cv2.medianBlur(src,ksize)


if __name__ == "__main__":

    img = cv2.imread("/home/robo/Documents/Jelly.jpg")

    res = median_blur(img,3)

    cv2.imshow("",res)
    cv2.imwrite(f"test/Noice Reduction/Median_Filter/ksize:3.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()