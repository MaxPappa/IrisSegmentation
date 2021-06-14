import cv2
import numpy as np
from pathlib import Path
from typing import DefaultDict, Final, Tuple
from scipy.signal import convolve
from utils import lineInt
from enum import Enum


DELTA_I: Final = 7
DELTA_P: Final = 5
SIGMA: Final = -1
gaussK_Iris = cv2.getGaussianKernel(DELTA_I, SIGMA).flatten()
gaussK_Pupil = cv2.getGaussianKernel(DELTA_P, SIGMA).flatten()

NORM_HEIGHT: Final = 100
NORM_WIDTH: Final = 600


class EyeSection(Enum):
    iris = 0
    pupil = 1


def daugman(
    img: np.ndarray, color: bool, sec: EyeSection, scale: int
) -> Tuple[float, Tuple[int, int], int]:
    """reproduction with numpy of the daugman integro-differential operator

    Parameters
    ----------
    img : np.ndarray
        image that should contain an eye.
    color : bool
        True if image is an RGB type (color image), False if is a grayscale one.
    sec : EyeSection
        sec can be an EyeSection.iris or EyeSection.pupil, respectively when daugman operator should search for iris or pupil.
    scale : int
        scale factor to resize the original image (and restrict amount of pixels to use as center of circumference)

    Returns
    ----------
    Tuple[float, Tuple[int, int], int]
        first value is the value of daugman operator computed for the candidate correct circumference, second value is the corresponding center,
        third and last value is the corresponding ray.
    """

    delta = DELTA_I if sec == EyeSection.iris else DELTA_P
    gaussK = gaussK_Iris if sec == EyeSection.iris else gaussK_Pupil
    full = (
        False if sec == EyeSection.iris else True
    )  # compute the whole circle boundary only in the pupil case

    img = cv2.resize(
        img, (img.shape[1] // scale, img.shape[0] // scale)
    )  # if sec == EyeSection.iris else cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))

    rows, cols = img.shape[:2]

    if sec == EyeSection.iris:
        rMin, rMax = rows // 6, rows // 2
    elif sec == EyeSection.pupil:
        rMin, rMax = rows // 8, rows // 2

    maxValue = 0.0
    candidateVal = 0.0
    candidateRay = 0
    candidateCenter = (0, 0)

    radii = np.arange(rMin, rMax, 1, dtype=int)

    for y in np.arange(
        rows // 3, rows - rows // 3, 1
    ):  # maybe these ranges are too restrictive
        for x in np.arange(cols // 3, cols - cols // 3, 1):

            center = (x, y)

            lInt, flag = lineInt(img, center, radii, color, full)
            if not flag or lInt.shape[0] < delta:
                continue
            elif not np.all(lInt):
                lInt = lInt[: (np.where(lInt == 0)[0][0])]

            d1 = convolve(lInt[:-1], gaussK, "same")
            d2 = convolve(lInt[1:], gaussK, "same")
            val = np.abs(d1 - d2)

            if sec == EyeSection.pupil:
                pupDivider, _ = lineInt(img, center, radii - 2, color, full)
                # pupDivider = convolve(pupDivider, gaussK, 'same')  # this line is pretty useless
                val = val / pupDivider[: val.shape[0]]

                # notice that we're

                # pupDivider[delta//2 : -delta//2]

            if np.max(val) > maxValue:
                maxValue, index = np.max(val), np.argmax(val)
                candidateVal = maxValue
                candidateCenter = center
                candidateRay = radii[
                    index
                ]  # [delta//2 : - delta//2][index] #if sec == EyeSection.iris else radii[index]
                # print(f"cCenter: {candidateCenter}, cRay: {candidateRay}, cVal: {candidateVal}")

    return (
        candidateVal,
        (candidateCenter[0] * scale, candidateCenter[1] * scale),
        candidateRay * scale,
    )


def irisSegmentation(p: Path, color: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """irisSegmentation detect, crop and normalize the eye contained in image at path p.

    Parameters
    ----------
    p: Path
        Path (object) of image to read.
    color: bool, optional
        if images at Path p is an RGB image, bool will be True, if image is a grayscale one, then color should be set to False.

    Returns
    ----------
    Tuple[np.ndarray, np.ndarray]
        normalized image and original image with circles around iris and pupil.
    """

    img = (
        cv2.imread(str(p.absolute()), cv2.IMREAD_COLOR)
        if color
        else cv2.imread(str(p.absolute()), cv2.IMREAD_GRAYSCALE)
    )
    rows, cols = img.shape[:2]

    scaleIris = 12
    iVal, iCenter, iRay = daugman(
        img, color, EyeSection.iris, scaleIris
    )  # iris value, center and ray

    # print(f'iVal:{iVal}, iCenter:{iCenter}, iRay:{iRay}')

    # Daugman-pupil-operator only needs red spectrum (can be seen as grayscale image if original image has 3 channels)
    yStart = iCenter[1] - iRay if iCenter[1] - iRay >= 0 else 0
    yEnd = iCenter[1] + iRay if iCenter[1] + iRay < rows else rows - 1
    xStart = iCenter[0] - iRay if iCenter[0] - iRay >= 0 else 0
    xEnd = iCenter[0] + iRay if iCenter[0] + iRay < cols else cols - 1

    roiSliceX, roiSliceY = slice(xStart, xEnd), slice(yStart, yEnd)

    scalePupil = 8
    pVal, pCenter, pRay = daugman(
        img[roiSliceY, roiSliceX, 2] if color else img[roiSliceY, roiSliceX],
        False,
        EyeSection.pupil,
        scalePupil,
    )
    pCenter = xStart + pCenter[0], yStart + pCenter[1]
    # print(f'pVal:{pVal}, pCenter:{pCenter}, pRay:{pRay}')

    normImg = normalize(img, iRay, iCenter, pRay, pCenter)

    # cv2.imwrite(f'{dstNorm.as_posix()}norm_{p.name}', normImg)

    # draw circles around iris and pupil (respectively green and red circles)
    cv2.circle(img, iCenter, int(iRay), (0, 255, 0))
    cv2.circle(img, pCenter, int(pRay), (0, 0, 255))
    # cv2.imwrite(f'{dst.as_posix()}/{p.name}', img)

    return normImg, img


def normalize(
    img: np.ndarray, iRay: int, iCenter: Tuple, pRay: int, pCenter: Tuple
) -> np.ndarray:
    """normalize, as the name says, normalizes the iris using the rubber sheet model.

    Parameters
    ----------
    img : np.ndarray
        image containing an eye.
    iRay : int
        detected iris ray.
    iCenter : Tuple
        detected center (x,y) of iris.
    pRay : int
        detected pupil ray.
    pCenter : Tuple
        detected center (x,y) of pupil.

    Returns
    ----------
    np.ndarray
        normalized iris with a dimension of 100x600.
    """

    height = int(round(iRay * 2))
    width = int(round(iRay * 2 * np.pi))
    normImg = np.zeros((height, width, 3), np.uint8)

    thetaStep = 2 * np.pi / width
    angles = np.arange(3 * np.pi / 2, 2 * np.pi + 3 * np.pi / 2, thetaStep)
    sinAngles = np.sin(angles)
    cosAngles = np.cos(angles)

    xPup = pCenter[0] + pRay * cosAngles
    yPup = pCenter[1] + pRay * sinAngles
    xIris = iCenter[0] + iRay * cosAngles
    yIris = iCenter[1] + iRay * sinAngles

    for j in np.arange(0, height, 1):
        r = j / height
        xNorm = np.round((1 - r) * xIris + r * xPup).astype(int)
        yNorm = np.round((1 - r) * yIris + r * yPup).astype(int)
        normImg[int(j)] = img[yNorm, xNorm]

    normImg = cv2.resize(normImg, (NORM_WIDTH, NORM_HEIGHT))

    return normImg
