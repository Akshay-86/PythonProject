import cv2
import numpy as np

def znormalize(src: np.ndarray) -> np.ndarray:
    """
    Z-score normalization:
    (x - mean) / std
    
    Parameters:
        src (ndarray): Input image ndarray.
    
    Returns:
        ndarray: Z-score normalized image (float32).
    """

    if src is None:
        raise ValueError("No image received")

    img_float = src.astype(np.float32)

    # mean, std returns column vectors per channel
    mean, std = cv2.meanStdDev(img_float)

    # Flatten them to shape (3,) or (1,)
    mean = mean.flatten()
    std = std.flatten()

    # If any channel has std = 0, avoid division error
    std[std == 0] = 1.0

    # Z-NORM using broadcasting
    normalized_img = (img_float - mean) / std

    return normalized_img


if __name__ == "__main__":
    src = "inputs/1.jpg"
    img = cv2.imread(src)

    res = znormalize(img)

    # Convert for display (Z-score has neg values)
    disp = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    disp = disp.astype(np.uint8)

    cv2.imshow("Z-score Normalized", disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
