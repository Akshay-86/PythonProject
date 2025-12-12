import cv2
import time
import numpy as np 

def resize(src: np.ndarray, dsize: tuple[int,int], dst: np.ndarray | None = None,fx: float = 0.0, fy: float = 0.0, interpolation: int = cv2.INTER_LINEAR):
    '''
        Adding or removing pixels through various interpolation methods to fit the new size

        Parameters:
            src (np.ndarray) : Actual image you want to resize.
            dsize (tuple) : Desired size of output image.
            fx (float) : Horizontal scaling factor.efaultVal = 0
            fy (float) : Vertical scaling factor.DefaultVal = 0
            interpolation (int) : interpolation method.DefaultVal = cv2.INTER_LINEAR 
    '''
    start_time = time.time()

    w, h = int(dsize[0]), int(dsize[1])

    # Validate dsize vs fx/fy:
    using_dsize = (w > 0 and h > 0)
    using_scale = (fx > 0 and fy > 0)

    if using_dsize and using_scale:
        raise ValueError("Provide either valid `dsize` (both >0) OR both `fx` and `fy` (>0), not both.")
    if not using_dsize and not using_scale:
        raise ValueError("You must provide either `dsize` (width,height) or both `fx` and `fy` (>0).")
    
    out_img = cv2.resize(src,dsize,dst,fx,fy,interpolation)
    end_time = time.time()
    print(f"Exicuted Time: {end_time-start_time:.4f}")
    return out_img

if __name__ == "__main__":
    src = "inputs/3.jpg"
    img = cv2.imread(src)

    res = resize(img,(0,0),fx=5,fy=1,interpolation=cv2.INTER_CUBIC)

    cv2.imshow("",res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()