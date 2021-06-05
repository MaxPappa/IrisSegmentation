import numpy as np
import cv2

def drawRays(img: np.ndarray, numRays: int):
    rows,cols = img.shape[:2]
    xStart,yStart = cols//2, 0
    mask = np.zeros_like(img, dtype=bool)
    rayLength = rows    # maybe I should check if cols < rows, then this assignment should change. Need to handle this.
    rayLens = np.arange(0,rayLength, 1, dtype=int).reshape(1,-1)
    angles = np.arange(0, np.pi, (np.pi+np.pi/numRays)/numRays).reshape(-1,1)
    X = (xStart+np.cos(angles)*rayLens).astype(int)
    Y = (yStart+np.sin(angles)*rayLens).astype(int)
    mask[Y,X] = 1
    return mask, (X,Y)

# input normRed is the one 2D tensor containing only red channel of the normalized iris
# output is a binary mask
def upperEyelidDetection(normRed: np.ndarray) -> np.ndarray:
    rows, cols = normRed.shape
    mask = np.zeros_like(normRed)
    upEyelid = np.zeros_like(normRed)
    # just swapping two parts of the normalized image, because here we want the upperEyelid at the middle of image
    upEyelid[:, cols//2:] = normRed[:, 0:cols//2 ]
    upEyelid[:, 0:cols//2] = normRed[:, cols//2:]

    blurred = cv2.GaussianBlur(upEyelid, (41,41), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    rayMask, rayCoords = drawRays(blurred)
    
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



# 'rly? maybe can do better than this.
def reflectionDetection(normBlue: np.ndarray) -> np.ndarray:
    #reflectionMask = np.ones_like(normBlue)
    reflectionMask = cv2.threshold(normBlue, 200, 255, cv2.THRESH_BINARY)
    return reflectionMask