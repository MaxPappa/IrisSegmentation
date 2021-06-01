import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from typing import Final
from scipy.signal import convolve
from utils import lineInt

DELTA: Final = 7
SIGMA: Final = -1
gaussK = cv2.getGaussianKernel(DELTA, SIGMA).flatten()

def daugman(p: Path, color: bool):
    print(p.name)
    img = cv2.imread(str(p.absolute()), cv2.IMREAD_COLOR) if color else cv2.imread(str(p.absolute()), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1]//8, img.shape[0]//8))
    # cv2.imshow(f'{color}',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    rows,cols = img.shape[:2]
    #print(f'rows:{rows}, cols:{cols}')
    rMin, rMax = 40, 70#rows//6, rows//3
    #print(f'rMin:{rMin}, rMax:{rMax}')
    maxValue = 0.0
    candidateVal = 0.0
    candidateRay = 0
    candidateCenter = (0,0)

    radii = np.arange(rMin, rMax, 1, dtype=np.int8)
    for y in np.arange(rows//3, rows-rows//3, 1):   # maybe these ranges are too restrictive
        for x in np.arange(cols//3, cols-cols//3, 1):
            center = (x,y)
            lInt, flag = lineInt(img, center, radii, color)
            if not flag or lInt.shape[0] < DELTA: continue
            elif not np.all(lInt):
                lInt = lInt[:(np.where(lInt == 0)[0][0])]
            
            d1 = convolve(lInt[:-1], gaussK, 'valid')
            d2 = convolve(lInt[1:], gaussK,'valid')
            val = np.abs(d1-d2)

            #diff = np.diff(lInt)
            #val = np.abs(convolve(diff, gaussK, 'valid'))
            
            if np.max(val) > maxValue:
                maxValue, index = np.max(val), np.argmax(val)
                candidateVal = maxValue
                candidateCenter = center
                candidateRay = radii[index]
                #print(f"cCenter: {candidateCenter}, cRay: {candidateRay}, cVal: {candidateVal}")
    cv2.circle(img, candidateCenter, int(candidateRay), (0,0,0))
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(f'./RESULTS/{p.name}', img)
    return candidateVal, candidateCenter, candidateRay

if __name__ == '__main__':
    path = '.'
    dir = Path(f'./{path}')
    with Path('./UTIRIS') as dataP:
        if not dataP.is_dir():
            from utils import downloadDataset
            utirisUrl = 'http://www.dsp.utoronto.ca/~mhosseini/UTIRIS%20V.1.zip'
            downloadDataset(dataP, utirisUrl)
    
    pList = list(dir.glob('./UTIRIS/RGB Images/*/*.JPG'))
    color = True
    color = [color for x in pList]

    with Pool(4) as pool:
        with tqdm(total=len(pList)) as pbar:
            pool.starmap(daugman, zip(pList,color))
            pbar.update()
    
    # for p in pList:
    #     daugman(p, True)