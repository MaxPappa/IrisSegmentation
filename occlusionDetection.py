import numpy as np
import cv2

# input normRed is the one 2D tensor containing only red channel of the normalized iris
# output is a binary mask
def upperEyelidDetection(normRed: np.ndarray) -> np.ndarray:
    pass


def lowerEyelidDetection(normRed: np.ndarray) -> np.ndarray:
    lowEyelidMask = np.ones_like(normRed)
    rows, cols = normRed.shape
    # sectionX, sectionY = slice(cols//4, 3*cols//4, 1), slice(0, rows//2, 1)
    # mean,stdev = cv2.meanStdDev(normRed[sectionY, sectionX])
    mean, stdev = cv2.meanStdDev(normRed[0:rows//2, cols//4, 3*cols//4])
    mean, stdev = float(mean), float(stdev)

    #threshold = int(mean+stdev)
    if stdev > mean/4:
        threshold = np.uint8(np.round(mean+stdev/2))
        Y,X = np.where(normRed[0:rows//2, :] > threshold)
        lowEyelidMask[Y,X] = 0
    
    return lowEyelidMask


def reflectionDetection(imgBlue: np.ndarray) -> np.ndarray:

    
    
    pass