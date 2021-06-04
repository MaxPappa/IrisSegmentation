import numpy as np
import cv2

def boundMask(img: np.ndarray, n: int, center: tuple, ray: int, color: bool, full: bool):
    #circ = 2*np.pi*ray
    theta = (2*np.pi) / n # / 2*np.pi*ray
    mask = np.zeros_like(img[:,:,0], dtype=bool) if color else np.zeros_like(img, dtype=bool)
    rows,cols = img.shape[:2]

    if not full:
        angle_1 = np.arange(theta, np.pi/4, theta)
        angle_2 = np.arange((np.pi*3)/4, (np.pi*5)/4, theta)
        angle_3 = np.arange((np.pi*7)/4, 2*np.pi, theta)
        angles = np.concatenate([angle_1, angle_2, angle_3])
    # else:
    #     angle_1 = np.arange(theta, (np.pi)/4, theta)
    #     angle_2 = np.arange((np.pi*3)/4, 2*np.pi, theta)
    #     angles = np.concatenate([angle_1, angle_2])
    else:
        angles = np.arange(theta, 2*np.pi, theta)

    X = np.round(ray * np.cos(angles) + center[0]).astype(int)
    Y = np.round(ray * np.sin(angles) + center[1]).astype(int)

    if np.any(X >= cols) or np.any(Y >= rows) or np.any(X < 0) or np.any(Y < 0):
        return None, 0
    else:
        mask[Y, X] = True
        return mask, np.count_nonzero(mask)

lumaF = lambda img, mask: np.sum(img[mask,0]*0.114) + np.sum(img[mask,1]*0.587) + np.sum(img[mask,2]*0.299)

def lineInt(img: np.ndarray, center: tuple, radii: np.ndarray, color: bool, full: bool):
    l = list()
    rows, cols = img.shape[:2]
    for r in radii:
        mask, numPx = boundMask(img=img, n=600, center=center, ray=r, color=color, full=full)
        if numPx == 0:
            l.append(0)
            return np.array(l), True
        if color: l.append(lumaF(img,mask) / numPx)
        else: l.append(np.sum(img[mask]) / numPx)
    return np.array(l), True



def polyFit():
    pass