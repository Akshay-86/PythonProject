import cv2
import numpy as np

def normalize(src: np.ndarray, dst: np.ndarray | None = None,dtype: int = -1,mask: np.ndarray | None = None)-> np.ndarray:

    '''
        0-255 normalization scales the pixel intensity values of an image to a floating-point range of 0.0,255.0, where the minimum value in the original image becomes 0.0 and the maximum value becomes 255.0. 

        Paramaters with * at end are mandatory parameters.

        It is recomended not to change the alpha and beta values.

        Parameters:
            src (ndarray): inpit image ndarray you want to 0-1 normalize.*
            dst (ndarray): outupt array ndarray.
            alpha (float): The lower boundary of the range.
            beta (float): The upper boundary of the range.
            norm_type (int): An integer flag representing the normalization method.
            dtype (int): An integer flag representing the desired output array data type.
            mask (ndarray): An optional operation mask to normalize only a specific region of the image.

    '''


    alpha = 0.0
    beta = 255
    norm_type = cv2.NORM_MINMAX 

    if src is None:
        raise ValueError("Source is NONE")
    if not len(src.shape) == 2:
        raise ValueError("Image must be gray scale")
    if not norm_type == cv2.NORM_MINMAX:
        raise ValueError("Norm Must be cv2.NORM_MINMAX")
        
    outimg = cv2.normalize(src,dst,alpha,beta,norm_type,dtype,mask)
    return outimg

if __name__ == "__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
    

    res = normalize(img)
    # cv2.putText(res,f"alpha:{alpha},beta:{beta},normtype:{norm_type}",(0,15),5,1,(0,0,255),1)
    cv2.imshow("",res)
    cv2.imwrite("test/normalization/0-1normalization/5.jpg",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()