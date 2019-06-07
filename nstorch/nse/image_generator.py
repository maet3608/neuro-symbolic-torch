"""
Randomly generate images with colored rectangles and circles.
"""

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skd

C_MASK = 1
C_RED = (255, 0, 0)
C_GREEN = (0, 255, 0)


def create_background(r, c):
    """Return blank RGB image with r rows, c cols and 3 channels"""
    return np.zeros((r, c, 3), dtype=np.uint8)


def draw(img, rr, cc, color):
    img = img.copy()
    msk = np.zeros(img.shape[:2])
    img[rr, cc, ...] = color
    msk[rr, cc] = C_MASK
    return img, msk


def draw_ball(img: np.array, rp, cp, s, color: tuple):
    """Draw filled circle in image
    s: relative size of circle, 1.0 fills entire image
    rp, cp:  row and col of circle center.
    color: RGB color tuple (r,g,b)
    """
    r, c, _ = img.shape
    rr, cc = skd.circle(rp, cp, s, img.shape)
    return draw(img, rr, cc, color)


def draw_box(img: np.array, rp, cp, s, color: tuple):
    sr,sc = (rp - s //2, cp - s //2)
    er,ec = (rp + s //2, cp + s //2)
    img = img.copy()
    msk = np.zeros(img.shape[:2])
    img[sr:er, sc:ec, ...] = color
    msk[sr:er, sc:ec] = C_MASK
    return img, msk


def gen_samples(config, ir, ic):
    for _ in range(config['samples']):
        img = create_background(ir, ic)

        sr, sc = rnd.uniform(0.7, 0.8), rnd.uniform(0.7, 0.8)
        rp, cp = int(ir * sr), int(ic * sc)
        img, ball_msk = draw_ball(img, rp, cp, 10, C_RED)

        sr, sc = rnd.uniform(0.1, 0.4), rnd.uniform(0.1, 0.4)
        rp, cp = int(ir * sr), int(ic * sc)
        img, box_msk = draw_box(img, rp, cp, 10, C_GREEN)

        yield (img, box_msk, ball_msk)


if __name__ == '__main__':
    from nutsflow import Consume, Map
    from nutsml import ViewImage, PrintColType

    ir, ic = 64, 64
    config = {'samples': 20,
              'objects': {'box': [1, 1], 'ball': [1, 1]}}
    gen_samples(config, ir, ic) >> ViewImage((0, 1, 2), pause=10) >> Consume()
