import numba
import numpy as np
import cv2
from scipy.signal.signaltools import convolve2d
from scipy.signal import argrelextrema
from numpy.polynomial.polynomial import polyfit
from typing import Tuple

prewittKerX = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittKerY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

def drawRays(
    img: np.ndarray, numRays: int
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """drawRays, as the name says, draw a number of rays into the image.

    These drawn rays are needed to detect boundaries of upper eyelid on the normalized image.
    After this, maxima are detected, outliers removed, and a polynomial fitting algorithm is used to
    draw these boundaries.

    Parameters
    ----------
    img : np.ndarray
        normalized image (only the red channel)
    numRays : int
        number of rays to draw into the normalized image

    Returns
    ----------
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        mask in which all 1s are pixels of rays.
        Second Tuple contains 2 arrays which are all the coordinates of 1s in the mask np.ndarray
    """

    rows,cols = img.shape[:2]
    xStart,yStart = cols//2, 0
    mask = np.zeros_like(img, dtype=np.uint8)
    rayLength = cols//4    # maybe I should check if cols < rows, then this assignment should change. Need to handle this.
    rayLens = np.arange(0,rayLength, 1, dtype=int).reshape(1,-1)
    angles = np.arange(0, np.pi, (np.pi+np.pi/numRays)/numRays).reshape(-1,1)
    X = (xStart+np.cos(angles)*rayLens).astype(int)
    #X = np.clip(np.column_stack([X-1, X, X+1]), 0, cols-1)
    Y = np.clip((yStart+np.sin(angles)*rayLens).astype(int), 0, rows-1)
    #Y = np.clip(np.column_stack([Y-1, Y, Y+1]), 0, rows-1)
    mask[Y,X] = 255
    return mask, (X,Y)



# input normRed is a 2D tensor containing only red channel of the normalized iris
# output is a binary mask
def upperEyelidDetection(normRed: np.ndarray, numRays: int) -> np.ndarray:
    """upperEyelidDetection detects upper eyelid on the red channel of a normalized iris image

    Parameters
    ----------
    normRed : np.ndarray
        red channel of the normalized (iris) image
    numRays : int
        number of rays to draw and use to detect upper eyelid

    Returns
    ----------
    np.ndarray
        binary mask 
    """

    rows, cols = normRed.shape
    upEyelid = np.zeros_like(normRed)
    # just swapping two parts of the normalized image, because here we want the upperEyelid at the middle of image
    upEyelid[:, cols // 2 :] = normRed[:, 0 : cols // 2]
    upEyelid[:, 0 : cols // 2] = normRed[:, cols // 2 :]

    #cv2.imwrite('./out/swapped_norm.jpg',upEyelid)

    swappedMask = np.full_like(upEyelid, fill_value=255)

    # blurred = cv2.GaussianBlur(
    #     upEyelid, (3, 3), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT
    # )
    rayMask, (rayXs, rayYs) = drawRays(upEyelid, numRays=numRays)

    rayedImg = upEyelid.copy()
    rayedImg[rayYs, rayXs] = 160  # grey rays

    #cv2.imwrite('./out/rayedImg.jpg', rayedImg)

    #sob = cv2.Sobel(rayedImg, cv2.CV_8U, 1, 1, ksize=5)
    blurred = cv2.GaussianBlur(rayedImg,(3,3),0)
    prewittedX = cv2.filter2D(blurred, -1, prewittKerX)
    prewittedY = cv2.filter2D(blurred, -1, prewittKerY)
    prewitt = prewittedX + prewittedY

    #cv2.imwrite('./out/prewitted.jpg', prewitt)

    maxIDs = np.argmax(
        prewitt[rayYs, rayXs][:, ::-1], axis=1
    )  # with the argmax I want the LAST max on each ray, not the first ones
    maxY = np.take_along_axis(rayYs[:, ::-1], np.expand_dims(maxIDs, axis=-1), axis=-1)
    maxX = np.take_along_axis(rayXs[:, ::-1], np.expand_dims(maxIDs, axis=-1), axis=-1)

    # outliers detection
    coordMask = np.ones_like(maxY).astype(bool)
    coordMask[argrelextrema(prewitt[maxY, maxX], np.less)] = False
    coordMask[argrelextrema(maxY, np.less)] = False
    coordMask[argrelextrema(maxX, np.less)] = False

    maxX, maxY = maxX[coordMask], maxY[coordMask]  # removing outliers

    rayedImg[maxY, maxX] = 0
    #cv2.imwrite('./out/rayedImg_removeOutliers.jpg', rayedImg)

    xPoly = np.arange(0, cols, 1, dtype=int)
    coeffs = polyfit(maxX, maxY, deg=2)
    coeffsIDs = np.arange(0, coeffs.shape[0], 1)
    yPoly = np.round(
        np.sum(coeffs * np.power(xPoly[:, None], coeffsIDs[None, :]), axis=1)
    ).astype(int)
    maskCoords = np.ones_like(yPoly, dtype=bool)
    maskCoords[np.where(yPoly < 0)] = False
    # maskCoords[
    #     np.where(yPoly >= rows)
    # ] = False  # But with degree == 2 this shouldn't happen.

    xPoly, yPoly = (
        xPoly[maskCoords],
        yPoly[maskCoords],
    )  # filtering out useless/noisy poly-fitted points

    # I literally don't know how to do this in a numpiest way. Tried with np.indices (which returns 2 grids of indices)
    # but can't figure out how to do it. So I surrended and used a for loop. \_('-')_/
    swappedMask = doit(swappedMask, xPoly, yPoly)
    #cv2.imwrite('./out/swappedMask_upperEyelid.jpg', swappedMask)

    mask = swappedMask.copy()
    mask[:, cols // 2 :] = swappedMask[:, 0 : cols // 2]
    mask[:, 0 : cols // 2] = swappedMask[:, cols // 2 :]

    #cv2.imwrite('./out/mask_upperEyelid.jpg', mask)

    return mask


@numba.jit(nopython=True)
def doit(swappedMask: np.ndarray, xPoly: np.ndarray, yPoly: np.ndarray) -> np.ndarray:
    for i in numba.prange(0, yPoly.shape[0]):
        swappedMask[0 : yPoly[i], xPoly[i]] = 0
    return swappedMask


def lowerEyelidDetection(normRed: np.ndarray) -> np.ndarray:
    """lowerEyelidDetection detects the lower eyelid of the normalized iris

    Parameters
    ----------
    normRed : np.ndarray
        red channel of the normalized iris image

    Returns
    ----------
    np.ndarray
        Binary mask of the normalized iris. All black pixels are those evalued as part of lower eyelid.
    """

    lowEyelidMask = np.full_like(normRed, fill_value=255)
    rows, cols = normRed.shape
    # sectionX, sectionY = slice(cols//4, 3*cols//4, 1), slice(0, rows//2, 1)
    # mean,stdev = cv2.meanStdDev(normRed[sectionY, sectionX])
    mean, stdev = cv2.meanStdDev(normRed[0 : rows // 2, cols // 4 : 3 * cols // 4])
    mean, stdev = float(mean), float(stdev)

    # newNormRed = normRed.copy()
    # newNormRed[0 : rows // 2, cols//4] = 0
    # newNormRed[rows//2, cols//4 : 3*cols//4] = 0
    # newNormRed[0 : rows // 2, 3*cols//4] = 0
    #cv2.imwrite('./out/lowerEyelidDistributionSection.jpg', newNormRed)

    # threshold = int(mean+stdev)
    if stdev > mean / 4:
        threshold = np.uint8(np.round(mean + stdev / 2))
        Y, X = np.where(normRed[0 : rows // 2, :] > threshold)
        lowEyelidMask[Y, X] = 0

    return lowEyelidMask


# 'rly? maybe can do better than this.
def reflectionDetection(normBlue: np.ndarray) -> np.ndarray:
    """reflectionDetection detects the pixels been part of reflection into the blue channel of normalized iris image.

    Parameters
    ----------
    normBlue : np.ndarray
        blue channel of normalized iris image

    Returns
    ----------
    np.ndarray
        Binary mask in which all white pixels are those of reflection.

    Notes
    ----------
    In the output binary mask, here, noisy pixels are those with value 255 (white ones), while in the
    other mask-detectors is the opposite.
    """

    # reflectionMask = np.ones_like(normBlue)
    _, reflectionMask = cv2.threshold(normBlue, 200, 255, cv2.THRESH_BINARY)
    return reflectionMask
