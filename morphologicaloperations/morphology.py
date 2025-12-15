import time
import cv2
import numpy as np

def morphology(src: np.ndarray, op: int = cv2.MORPH_OPEN, kernal: np.ndarray | None = None, dst: np.ndarray | None = None, anchor: tuple[int,int] = (-1,-1),iterations: int = 1,borderType: int =  cv2.BORDER_CONSTANT, borderValue: int = 0) -> np.ndarray:
    '''
    Use this funtion to perform Different types of morphologicaloperations.
    
    Parameters:
        src (np.ndarray) : The input image (source array) to be eroded.Grayscale image recomended.
        op (int): To deside the morphological operation in your image. PossibleValues 0-6.
            0 -> cv2.MORPH_ERODE, 
            1 -> cv2.MORPH_DILATE, 
            2 -> cv2.MORPH_OPEN, 
            3 -> cv2.MORPH_CLOSE, 
            4 -> cv2.MORPH_GRADIENT, 
            5 -> cv2.MORPH_TOPHAT, 
            6 -> cv2.MORPH_BLACKHAT
        kernal (np.ndarray) : The structuring element used to define the shape and size of the neighborhood for the operation.Must be odd number DefaultVal = (3,3)
        dst (np.ndarray) : The output image where the result of the erosion will be stored.
        anchor (tuple) : The position of the anchor within the structuring element.DefaultVal = (3,3)
        iterations (int) : The number of times the erosion operation is applied sequentially.Must be a positive integer (>= 1).
        borderType (int) : The method used for pixel extrapolation when the kernel goes outside image boundaries.
        borderValue (int) : The specific value to use for the border if borderType is cv2.BORDER_CONSTANT

    Returns: 
        image numpy nd array

    '''

    start_time = time.time()

    if src is None:
        raise ValueError("Source image is None.")
    
    if kernal is None:
        kernal = np.ones((3,3), np.uint8)
    
    if kernal.ndim != 2:
        raise ValueError("kernel must be 2D")

    h, w = kernal.shape
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError("kernel dimensions must be odd")

    
    if iterations < 1:
        raise ValueError("Itarations Must be positive number")
    
    if op > 6 or op < 0:
        raise ValueError("op value muist be bewteen 0-6")
    
    out_img = cv2.morphologyEx(src,op,kernal,dst,anchor,iterations,borderType,borderValue)
    end_time = time.time()
    print(f"Exicution Time: {end_time-start_time:.6f}s")
    return out_img

if __name__ == "__main__":
    src = "inputs/2.jpg"
    img = cv2.imread(src)
    res = morphology(img,op=-4,iterations=0)
    res1 = np.hstack((img,res))
    cv2.imshow("",res1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
