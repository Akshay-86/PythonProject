import cv2
import time
import numpy as np

def gaussian_blur(src: np.ndarray,ksize: tuple[int,int],sigmaX: float,dst: np.ndarray | None= None,sigmaY: float = 0 ,borderType: int=cv2.BORDER_DEFAULT)->np.ndarray:
    '''
        Blur the image using Gaussian_Blur
        
        Paramaters with * at end are mandatory parameters 

        Parameters:
            src(Image ndarray):  Actual image you want to blur.*
            ksize(tuple):  Positive,odd no. kernal matrix(e.g:(3,3) or (5,5)). Min Value = (1,1)*
            sigmaX(float):  Standard deviation in the X direction. MinVal = 0*
            dst(Image ndarray): Destination ndarray for result image.
            sigmaY(float): Standard deviation in the Y direction. MinVal = 0,DefaultVal = same as sigmaX
            borderType(int): Pixel extra polation method. DefaultVAl = cv2.BORDER_DEFAULT
        
        Returns:
            Image_ndarray: Result Blured Image.
        
        Raises:
            ValueError: If bad arguments recived.

    '''
    start_time = time.time()

    if src is None:
        raise ValueError("Source is NONE")
    if not ksize[0]>0 and ksize[1]>0:
        raise ValueError("ksize must be positive")
    if ksize[0]%2 == 0 or ksize[1]%2 == 0:
        raise ValueError("ksize values must be odd Number")
    
    out_img = cv2.GaussianBlur(src,ksize,sigmaX,dst=dst,sigmaY=sigmaY,borderType=borderType)
    end_time = time.time()
    print(f"Exicution Time: {end_time - start_time}")
    return  out_img

    


if __name__=="__main__":
    src = "inputs/5.jpg"
    img = cv2.imread(src)
    ksize = (3,3)
    sigmaX = 5
    dst = np.array([])
    sigmaY = 20 
    res = gaussian_blur(img,ksize,sigmaX,sigmaY)
    cv2.putText(res,f"ksize{ksize},sigmaX:{sigmaX},sigmaY:{sigmaY}",(0,15),5,1,(0,0,255))
    cv2.imshow("",res)
    cv2.imwrite("test/noice_reduction/gaussian_filter/1.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()