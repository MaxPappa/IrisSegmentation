import cv2
import numpy as np


def searchReflection(img: np.ndarray, ksize: int, c: float) -> np.ndarray:
    #mask = np.zeros(img.shape[:2])     # idk about channels, just need a (rows,cols,1) tensor
    # img[:,:,0] is just the blue channel of img (cv2 use the BGR order instead of RGB)
    mask = cv2.adaptiveThreshold(img[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ksize, c)
    mask = cv2.dilate(mask, np.zeros((3,3)), anchor=(-1,-1), iterations=2)
    return mask


def inpaintRefl(img: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    imgInp = cv2.inpaint(img, mask, iterations, cv2.INPAINT_TELEA)
    return imgInp


def blurImg(img: np.ndarray) -> np.ndarray:
    ksize = (3,3)
    return cv2.blur(img, ksize=ksize)



def preprocessing(img: np.ndarray):
    blurred = blurImg(img)      # should I blur before or after inpainting? really dk.
    ksize, c = 3, -20
    mask = searchReflection(blurred, ksize, c)
    imgInp = inpaintRefl(blurred, mask, 1)
    return imgInp
