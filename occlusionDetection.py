import numpy as np
import cv2
from scipy.signal.signaltools import convolve2d


def drawRays(img: np.ndarray, numRays: int):
    rows,cols = img.shape[:2]
    xStart,yStart = cols//2, 0
    mask = np.zeros_like(img, dtype=bool)
    rayLength = rows    # maybe I should check if cols < rows, then this assignment should change. Need to handle this.
    rayLens = np.arange(0,rayLength, 1, dtype=int).reshape(1,-1)
    angles = np.arange(0, np.pi, (np.pi+np.pi/numRays)/numRays).reshape(-1,1)
    X = (xStart+np.cos(angles)*rayLens).astype(int)
    X = np.clip(np.column_stack([X-1, X, X+1]), 0, cols-1)
    Y = (yStart+np.sin(angles)*rayLens).astype(int)
    Y = np.clip(np.column_stack([Y-1, Y, Y+1]), 0, rows-1)
    mask[Y,X] = 1
    return mask, (X,Y)

# input normRed is a 2D tensor containing only red channel of the normalized iris
# output is a binary mask
def upperEyelidDetection(normRed: np.ndarray) -> np.ndarray:
    rows, cols = normRed.shape
    mask = np.zeros_like(normRed)
    upEyelid = np.zeros_like(normRed)
    # just swapping two parts of the normalized image, because here we want the upperEyelid at the middle of image
    upEyelid[:, cols//2:] = normRed[:, 0:cols//2 ]
    upEyelid[:, 0:cols//2] = normRed[:, cols//2:]

    blurred = cv2.GaussianBlur(upEyelid, (41,41), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    rayMask, (rayXs, rayYs) = drawRays(blurred, numRays=15)

    rayedImg = normRed.copy()
    rayedImg[rayYs,rayXs] = 160 # grey rays

    sob = cv2.Sobel(rayedImg, cv2.CV_8U, 1, 1, ksize=5)
    maxIDs = np.argmax(sob[rayYs, rayXs][:,::-1], axis=1)   # with the argmax I want the LAST max on each ray, not the first ones
    maxY = np.take_along_axis(rayYs[:,::-1], np.expand_dims(maxIDs, axis=-1), axis=-1)
    maxX= np.take_along_axis(rayXs[:,::-1], np.expand_dims(maxIDs, axis=-1), axis=-1)


    # below lines are useless, just a reminder for what to do next.
    sob[maxY,maxX] = 255
    normRed[maxY, maxX] = 0 



    # maxXs, maxYs = list(), list()
    # for ray in normRed[rayYs, rayXs]:
    #     maxId = np.argmax(convolve2d(ray, SCHARR, mode='valid'))
    #     maxXs.append(rayYs[maxId]), maxYs.append(rayXs[maxId])
    # maxXs, maxYs = np.array(maxXs), np.array(maxYs)
    # normRed[maxYs,maxXs] = 255
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