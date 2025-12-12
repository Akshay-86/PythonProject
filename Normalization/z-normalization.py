import cv2
import time
import numpy as np

def znormalize(src: np.ndarray) -> np.ndarray:
    starttime = time.time()
    """
    Docstring for znormalize
    
    Parameters:
        src (ndarray): inpit image ndarray you want to 0-1 normalize.*
    
    Returns: 
        Image_ndarray: Result z_score normalized image.
    """

    if src is None:
        raise ValueError("No image received")

    img_float = src.astype(np.float32)

    mean, std = cv2.meanStdDev(img_float)

    mean = mean.flatten()
    std = std.flatten()

    std[std == 0] = 1.0

    normalized_img = (img_float - mean) / std

    endtime = time.time()
    print("Runtime: ",endtime-starttime)
    return normalized_img


if __name__ == "__main__":
    src = "inputs/8.png"
    img = cv2.imread(src)

    res = znormalize(img)

    disp = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow("",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
