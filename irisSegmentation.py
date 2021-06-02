import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from typing import Final
from scipy.signal import convolve
from utils import lineInt
from enum import Enum

DELTA_I: Final = 7
DELTA_P: Final = 5
SIGMA: Final = -1
gaussK_Iris = cv2.getGaussianKernel(DELTA_I, SIGMA).flatten()
gaussK_Pupil = cv2.getGaussianKernel(DELTA_P, SIGMA).flatten()

class EyeSection(Enum):
    iris = 0
    pupil = 1


def daugman(img: np.ndarray, color: bool, sec: EyeSection, scale: int):
    delta = DELTA_I if sec==EyeSection.iris else DELTA_P
    gaussK = gaussK_Iris if sec==EyeSection.iris else gaussK_Pupil
    full = False if sec==EyeSection.iris else True  # compute the whole circle boundary only in the pupil case

    img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))# if sec == EyeSection.iris else cv2.resize(img, (img.shape[1]//3, img.shape[0]//3))

    rows,cols = img.shape[:2]

    if sec == EyeSection.iris: rMin, rMax = rows//6, rows//2
    elif sec == EyeSection.pupil: rMin, rMax = rows//8, rows//4 #25, 50

    maxValue = 0.0
    candidateVal = 0.0
    candidateRay = 0
    candidateCenter = (0,0)

    radii = np.arange(rMin, rMax, 1, dtype=int)

    for y in np.arange(rows//3, rows-rows//3, 1):   # maybe these ranges are too restrictive
        for x in np.arange(cols//3, cols-cols//3, 1):

            center = (x,y)

            lInt, flag = lineInt(img, center, radii, color, full)
            if not flag or lInt.shape[0] < delta: continue
            elif not np.all(lInt):
                lInt = lInt[:(np.where(lInt == 0)[0][0])]
            
            d1 = convolve(lInt[:-1], gaussK, 'valid')
            d2 = convolve(lInt[1:], gaussK,'valid')
            val = np.abs(d1-d2)

            if sec == EyeSection.pupil:
                pupDivider, _ = lineInt(img, center, radii-2, color, full)
                val = val / pupDivider[delta//2 : -delta//2]
            
            if np.max(val) > maxValue:
                maxValue, index = np.max(val), np.argmax(val)
                candidateVal = maxValue
                candidateCenter = center
                candidateRay = radii[delta//2 : - delta//2][index] #if sec == EyeSection.iris else radii[index]
                #print(f"cCenter: {candidateCenter}, cRay: {candidateRay}, cVal: {candidateVal}")

    return candidateVal, (candidateCenter[0]*scale,candidateCenter[1]*scale), candidateRay*scale


def irisSegmentation(p: Path, dst: Path, color: bool = True):
    print(p.name)
    img = cv2.imread(str(p.absolute()), cv2.IMREAD_COLOR) if color else cv2.imread(str(p.absolute()), cv2.IMREAD_GRAYSCALE)

    scaleIris = 8
    iVal, iCenter, iRay = daugman(img, color, EyeSection.iris, scaleIris)  # iris value, center and ray

    #print(f'iVal:{iVal}, iCenter:{iCenter}, iRay:{iRay}')

    # Daugman-pupil-operator only needs red spectrum (can be seen as grayscale image if original image has 3 channels)
    roiSliceX, roiSliceY = slice(iCenter[0]-iRay, iCenter[0]+iRay), slice(iCenter[1]-iRay, iCenter[1]+iRay)
    
    scalePupil = 6
    pVal, pCenter, pRay = daugman(img[roiSliceY, roiSliceX, 2] if color else img[roiSliceY, roiSliceX], False, EyeSection.pupil, scalePupil)

    #print(f'pVal:{pVal}, pCenter:{pCenter}, pRay:{pRay}')
    
    cv2.imwrite(f'{dst.name}/iris_{p.name}', img[roiSliceY, roiSliceX, :])
    cv2.circle(img, iCenter, int(iRay), (0,0,0))
    cv2.circle(img[roiSliceY, roiSliceX, :], pCenter, int(pRay), (255,0,0))
    cv2.imwrite(f'{dst.name}/{p.name}', img)

if __name__ == '__main__':
    path = '.'
    dir = Path(f'./{path}')
    with Path('./UTIRIS') as dataP:
        if not dataP.is_dir():
            from downloadDataset import downloadDataset
            utirisUrl = 'http://www.dsp.utoronto.ca/~mhosseini/UTIRIS%20V.1.zip'
            downloadDataset(dataP, utirisUrl)
    
    pList = list(dir.glob('./UTIRIS/RGB Images/*/*.JPG'))
    color = True
    color = [color for x in pList]
    dsts = [Path('./RES') for x in pList]

    with Pool(4) as pool:
        with tqdm(total=len(pList)) as pbar:
            pool.starmap(irisSegmentation, zip(pList,dsts, color))
            pbar.update()
    
    
    # irisSegmentation(Path('./UTIRIS/RGB Images/005/IMG_005_L_1.JPG'), Path('./RES'), True)

    #for p in pList:
    #    daugman(p, True)