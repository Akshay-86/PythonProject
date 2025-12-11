import cv2
import numpy as np

def bilateral_blur(src: np.ndarray,d: int,sigmaColor: float,sigmaSpace: float,dst: np.ndarray | None = None,borderType: int = cv2.BORDER_DEFAULT)->np.ndarray:
    '''
        Advanced blur technique that excels at reducing noise while preserving sharp edges.

        Paramaters with * at end are mandatory parameters. 

        Parameters:
            src(Image ndarray):  Actual image you want to blur.*
            d(int): Diameater of the pixel neighborhood. MinVal = 1(or -1)* 
            sigmaColor(float):  Controls the influence of color difference. Larger values mean more colors within the neighborhood will be mixed. BestVal = >75 and <100 *
            sigmaSpace(float): Controls the influence of spatial distance. Larger values mean farther pixels will influence each other if their colors are similar enough. BestVal = >75 and <100*
            dst(Image ndarray): The destination image.
            borderType(int): Pixel extra polation method. DefaultVAl = cv2.BORDER_DEFAULT

        Returns:
            Image_ndarray: Result Blured Image.
        
        Raises:
            ValueError: If bad arguments recived.
    '''

    if src is None:
        raise ValueError("Source is NONE")
    if sigmaColor<0 or sigmaSpace<0:
        raise ValueError("sigmaColor and sigmaValue must be >=0")
    
    return cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace,dst=dst,borderType=borderType)

if __name__=="__main__":
    img = cv2.imread("/home/robo/Documents/test.png")
    res = bilateral_blur(img,20,150,150)
    cv2.imshow("",res)
    cv2.imwrite("test/Noice Reduction/Bilateral_Filter/d:20,sigmaColor:150,sigmaSpace:150.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
