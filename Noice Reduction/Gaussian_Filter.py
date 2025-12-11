import cv2
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

    if src is None:
        raise ValueError("Source is NONE")
    if not ksize[0]>0 and ksize[1]>0:
        raise ValueError("ksize must be positive")
    if ksize[0]%2 == 0 or ksize[1]%2 == 0:
        raise ValueError("ksize values must be odd Number")


    return  cv2.GaussianBlur(src,ksize,sigmaX,dst=dst,sigmaY=sigmaY,borderType=borderType)

    


if __name__=="__main__":

    img = cv2.imread("/home/robo/Documents/image.jpg")

    res = gaussian_blur(img,(3,3),5,sigmaY=20)

    cv2.imshow("",res)
    cv2.imwrite("test/Noice Reduction/ksize:(3,3),sigmax:5,sigmay:20.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()