import cv2
import time
import numpy as np
from brisque import BRISQUE

def bilateral_blur(src: np.ndarray,d: int,sigmaColor: float,sigmaSpace: float,dst: np.ndarray | None = None,borderType: int = cv2.BORDER_DEFAULT)->np.ndarray:
    '''
        Advanced blur technique that excels at reducing noise while preserving sharp edges.

        Paramaters with * at end are mandatory parameters. 

        Parameters:
            src(Image ndarray):  Actual image you want to blur.*
            d(int): Diameater of the pixel neighborhood. MinVal = 1(or -1)* 
            sigmaColor(float):  Controls the influence of color difference. Larger values mean more colors within the neighborhood will be mixed. RecomendedVal = >75 and <100 *
            sigmaSpace(float): Controls the influence of spatial distance. Larger values mean farther pixels will influence each other if their colors are similar enough. RecomendedVal = >75 and <100*
            dst(Image ndarray): The destination image.
            borderType(int): Pixel extra polation method. DefaultVAl = cv2.BORDER_DEFAULT

        Returns:
            Image_ndarray: Result Blured Image.
        
        Raises:
            ValueError: If bad arguments recived.
    '''
    start_time = time.time()

    if src is None:
        raise ValueError("Source is NONE")
    if sigmaColor<0 or sigmaSpace<0:
        raise ValueError("sigmaColor and sigmaValue must be >=0")
    
    out_img = cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace,dst=dst,borderType=borderType)
    end_time = time.time()
    print(f"Exicution Time: {end_time - start_time}")
    return out_img

if __name__=="__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src)
    d = 10
    sigmaColor = 50
    sigmaSpace = 25
    dst = np.array([])
    bordertype = cv2.BORDER_DEFAULT

    res = bilateral_blur(img,d,sigmaColor,sigmaSpace)
    cv2.putText(res,f"d:{d},SigmaColor:{sigmaColor},sigmaSpace:{sigmaSpace}",(0,15),5,1,(0,0,255))
    cv2.imshow("",res)
    cv2.imwrite("test/noice_reduction/bilateral_filter/1.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
