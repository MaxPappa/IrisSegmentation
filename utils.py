import math
from numpy.core.arrayprint import _make_options_dict
import scipy
import numpy as np
from math import sqrt, pi, sin, cos
import cv2


def boundMask(img: np.ndarray, n: int, center: tuple, ray: int, color: bool):
    #circ = 2*np.pi*ray
    theta = (2*np.pi) / n
    mask = np.zeros_like(img)
    #print(mask.shape, img.shape)
    rows,cols = img.shape[:2]

    angle_1 = np.arange(theta, (2*np.pi)/8, theta)
    angle_2 = np.arange((2*np.pi*3)/8, (2*np.pi*5)/8, theta)
    angle_3 = np.arange((2*np.pi*7)/8, 2*np.pi, theta)
    angles = np.concatenate([angle_1, angle_2, angle_3])
    #angles = np.arange(theta, 2*np.pi, theta)

    X = np.round(ray * np.cos(angles) + center[0]).astype(np.uint8)
    Y = np.round(ray * np.sin(angles) + center[1]).astype(np.uint8)

    if np.any(X >= cols) or np.any(Y >= rows) or np.any(X < 0) or np.any(Y < 0):
        # print(f'rows:{rows},cols:{cols}, Y:{Y}, X:{X}')
        # print(Y >= rows)
        # print(np.any(Y>=rows))
        return None, 0
    else:
        if color:
            mask[Y, X, :] = 1
        else: mask[Y,X] = 1
        return mask, np.count_nonzero(mask[:,:,0]) if color else np.count_nonzero(mask)

lumaF = lambda img, mask: np.sum((img[:,:,0] * mask[:,:,0])*0.114) + np.sum((img[:,:,1] * mask[:,:,1])*0.587) + np.sum((img[:,:,2] * mask[:,:,2])*0.299)

def lineInt(img: np.ndarray, center: tuple, radii: np.ndarray, color: bool):
    l = list()
    rows, cols = img.shape[:2]
    for r in radii:
        mask, numPx = boundMask(img=img, n=600, center=center, ray=r, color=color)
        if numPx == 0:
            l.append(0)
            return np.array(l), True
        if color: l.append(lumaF(img,mask))
        else: l.append(np.sum(img & mask))
    return np.array(l), True