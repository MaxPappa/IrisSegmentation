import numba
import numpy as np


def boundMask(
    img: np.ndarray, n: int, center: tuple, ray: int, color: bool, full: bool
):
    # circ = 2*np.pi*ray
    theta = (2 * np.pi) / n  # / 2*np.pi*ray
    mask = (
        np.zeros_like(img[:, :, 0], dtype=bool)
        if color
        else np.zeros_like(img, dtype=bool)
    )
    rows, cols = img.shape[:2]

    if not full:
        angles = np.concatenate(
            [
                np.arange(theta, np.pi / 4, theta),
                np.arange((np.pi * 3) / 4, (np.pi * 5) / 4, theta),
                np.arange((np.pi * 7) / 4, 2 * np.pi, theta),
            ]
        )
    else:
        angles = np.arange(theta, 2 * np.pi, theta)

    X = np.round(ray * np.cos(angles) + center[0]).astype(int)
    Y = np.round(ray * np.sin(angles) + center[1]).astype(int)

    if np.any(X >= cols) or np.any(Y >= rows) or np.any(X < 0) or np.any(Y < 0):
        return None, 0
    else:
        mask[Y, X] = True
        return mask, np.count_nonzero(mask)


def lumaF(img: np.ndarray, mask: np.ndarray):
    return (
        np.sum(img[mask, 0] * 0.114)
        + np.sum(img[mask, 1] * 0.587)
        + np.sum(img[mask, 2] * 0.299)
    )


def lineInt(img: np.ndarray, center: tuple, radii: np.ndarray, color: bool, full: bool):
    rows, cols = img.shape[:2]
    l = np.zeros((len(radii),))
    for i, r in enumerate(radii):
        mask, numPx = boundMask(
            img=img, n=600, center=center, ray=r, color=color, full=full
        )
        if numPx == 0:
            return l[: i + 1], True
        if color:

            l[i] = lumaF(img, mask) / numPx
        else:
            l[i] = np.sum(img[mask]) / numPx
    return l, True
