import cv2
import time
import numpy as np
from brisque import BRISQUE


def gabor_filter(src: np.ndarray, ksize: tuple[int, int] = (31,31), sigma: float = 3.0, theta: float = 45, lambd: float =10, gamma: float = 0.4, psi: float = 0, dst: np.ndarray | None = None, borderType: int = cv2.BORDER_DEFAULT) -> np.ndarray:

    '''
        Apply Gabor filter for texture and directional feature extraction.

        Paramaters with * at end are mandatory parameters.

        Parameters:
            src (Image ndarray): Input image (grayscale or BGR).*
            ksize (tuple[int,int]): Size of the Gabor kernel (odd, positive).* Recomended(21,21)-(51,51)
            sigma (float): Standard deviation of Gaussian envelope.* Recomended 2.0 - 6.0
            theta (float): Orientation of the filter in radians.* Recomended 0 - π
            lambd (float): Wavelength of the sinusoidal factor.* Recomended 4 - 20
            gamma (float): Spatial aspect ratio (ellipticity).* Recomended 0.3 - 0.8 
            psi (float): Phase offset. Default = 0
            dst (Image ndarray): Destination image.
            borderType (int): Pixel extrapolation method.

        Returns:
            Image_ndarray: Gabor filtered image.

        Raises:
            ValueError: If invalid parameters are provided.
    '''
    start_time = time.time()

    # Validation
    if src is None:
        raise ValueError("Source image is NONE")

    if not isinstance(ksize, tuple) or ksize[0] % 2 == 0 or ksize[1] % 2 == 0:
        raise ValueError("ksize must be a tuple of odd positive integers")

    if sigma <= 0 or lambd <= 0:
        raise ValueError("sigma and lambd must be > 0")

    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    
    if ksize[1] > 51 :
        raise Warning("Ksize value is too high,results may not be good")

    
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    # Create Gabor kernel
    kernel = cv2.getGaborKernel(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi, ktype=cv2.CV_32F)

    #  Apply filter
    out_img = cv2.filter2D(gray, ddepth=cv2.CV_8U, kernel=kernel, dst=dst, borderType=borderType)

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

    return out_img


if __name__ == "__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src)

    # Gabor parameters
    ksize = (53, 53)
    sigma = 4.0
    theta = np.pi / 2 
    lambd = 10.0
    gamma = 0.5
    psi = 90

    res = gabor_filter(img,ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=psi)

    cv2.imshow("Gabor Output", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
