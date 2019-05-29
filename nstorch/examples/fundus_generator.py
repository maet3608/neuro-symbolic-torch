"""
Create synthetic fundus images and
"""

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as skd

# pathology RGB colors
C_PATHO = {
    'fu': (205, 89, 47),  # fundus
    'od': (200, 200, 0),  # optic disc
    'fo': (180, 80, 30),  # Fovea
    'ma': (139, 69, 19),  # microaneurysm
    'ex': (250, 250, 0),  # exudate
    'ha': (140, 70, 20),  # haemorrhage
}

C_MASK = 1


def create_background(r, c):
    """Return blank RGB image with r rows, c cols and 3 channels"""
    return np.zeros((r, c, 3), dtype=np.uint8)


def draw(img, rr, cc, color):
    img = img.copy()
    msk = np.zeros(img.shape[:2])
    img[rr, cc, ...] = color
    msk[rr, cc] = C_MASK
    return img, msk


def draw_circle(img: np.array, rp, cp, s, color: tuple):
    """Draw filled circle in image
    s: relative size of circle, 1.0 fills entire image
    rp, cp:  row and col of circle center.
    color: RGB color tuple (r,g,b)
    """
    r, c, _ = img.shape
    rf = int(min(r, c) * 0.5 * s)
    rr, cc = skd.circle(rp, cp, rf, img.shape)
    return draw(img, rr, cc, color)


def draw_fundus(img, rp, cp, s=0.9):
    """Draw fundus circle with percentage size s of image"""
    return draw_circle(img, rp, cp, s, color=C_PATHO['fu'])


def draw_opticdisc(img, rp, cp, s=0.2):
    """Draw optic disc"""
    return draw_circle(img, rp, cp, s, color=C_PATHO['od'])


def draw_fovea(img, rp, cp, s=0.07):
    """Draw fovea"""
    return draw_circle(img, rp, cp, s, color=C_PATHO['fo'])


def draw_microaneurysm(img, rp, cp):
    """Draw draw_microaneurysm"""
    rr, cc = skd.line(rp, cp, rp, cp)
    return draw(img, rr, cc, C_PATHO['ma'])


def draw_haemorrhage(img, inmsk, rp, cp, srr=0.04, src=0.06):
    """Draw haemorrhage"""
    r, c, _ = img.shape
    rra, cra = int(r * srr), int(c * src)
    rr, cc = skd.ellipse(rp, cp, rra, cra, img.shape)
    valid = inmsk[rr, cc] == C_MASK
    rr, cc = rr[valid], cc[valid]
    return draw(img, rr, cc, C_PATHO['ha'])


def draw_exudate(img, inmsk, rp, cp, srr=0.1, src=0.07, d=0.3):
    """Draw exudate: random yellow pixels in elliptic area with density d"""
    r, c, _ = img.shape
    rra, cra = int(r * srr), int(c * src)
    rr, cc = skd.ellipse(rp, cp, rra, cra, img.shape)
    n = len(rr)
    idx = np.random.choice(n, int(d * n))
    img = img.copy()
    msk = np.zeros((r, c))
    for rp, cp in zip(rr[idx], cc[idx]):
        if inmsk[rp, cp]:
            rrp, ccp = skd.line(rp, cp, rp, cp)
            img[rrp, ccp, ...] = C_PATHO['ex']
            msk[rrp, ccp] = C_MASK
    return img, msk


def sample_coord(inmsk, n):
    rr, cc = np.where(inmsk == C_MASK)
    idx = np.random.choice(len(rr), n, replace=False)
    for rp, cp in zip(rr[idx], cc[idx]):
        yield rp, cp


def draw_pathology(ptype, img, inmsk, rp, cp, sr, sc):
    if ptype == 'ha':
        return draw_haemorrhage(img, inmsk, rp, cp, sr, sc)
    if ptype == 'ex':
        return draw_exudate(img, inmsk, rp, cp, sr, sc)
    if ptype == 'ma':
        return draw_microaneurysm(img, rp, cp)
    raise ValueError('Unknown pathology type:' + ptype)


def hemifield(mask, isupper, h=0.5):
    """ Filter input mask with upper or lower hemifield at row (h)"""
    r, _ = mask.shape
    hemi = np.zeros_like(mask)
    if isupper:
        hemi[:int(r * h), :] = 1
    else:
        hemi[int(r * (1 - h)):, :] = 1
    return mask * hemi


def calc_grade(samples):
    """Return DR severity grade"""
    num = {t: n for t, n, _, _ in samples}
    if not num['ex'] and not num['ha']:
        if 0 < num['ma'] < 5:
            return 1
        if num['ma'] >= 5:
            return 2
    if num['ex'] or num['ha']:
        return 3

    return 0


def gen_images(config, ir, ic):
    for _ in range(config['samples']):
        img = create_background(ir, ic)

        s = rnd.uniform(0.9, 1.1)
        img, fundus_msk = draw_fundus(img, ir // 2, ic // 2, s)

        sr, sc = rnd.uniform(0.4, 0.6), rnd.uniform(0.2, 0.3)
        img, fovea_msk = draw_fovea(img, int(ir * sr), int(ic * sc))

        sr, sc = rnd.uniform(0.4, 0.6), rnd.uniform(0.7, 0.8)
        img, opticdisc_msk = draw_opticdisc(img, int(ir * sr), int(ic * sc))

        inmsk = fundus_msk - fovea_msk - opticdisc_msk

        for ptype, (n_min, n_max) in config['pathologies'].items():
            n = rnd.randint(n_min, n_max)
            for rp, cp in sample_coord(inmsk, n):
                sr, sc = rnd.uniform(0.03, 0.15), rnd.uniform(0.03, 0.15)
                img, msk = draw_pathology(ptype, img, inmsk, rp, cp, sr, sc)

        yield img



def gen_samples(config, ir, ic):
    for _ in range(config['samples']):
        samples = []

        # draw fundus
        img = create_background(ir, ic)
        s = rnd.uniform(0.9, 1.1)
        img, fundus_msk = draw_fundus(img, ir // 2, ic // 2, s)

        # draw fovea
        sr, sc = rnd.uniform(0.4, 0.6), rnd.uniform(0.2, 0.3)
        rp, cp = int(ir * sr), int(ic * sc)
        img, fovea_msk = draw_fovea(img, rp, cp)
        fovea_imks = np.zeros_like(fundus_msk)
        fovea_imks[rp, cp] = C_MASK
        samples.append(('fo', 1, fovea_msk, fovea_imks))

        # draw optic disc
        sr, sc = rnd.uniform(0.4, 0.6), rnd.uniform(0.7, 0.8)
        rp, cp = int(ir * sr), int(ic * sc)
        img, od_msk = draw_opticdisc(img, rp, cp)
        od_imks = np.zeros_like(fundus_msk)
        od_imks[rp, cp] = C_MASK
        samples.append(('od', 1, od_msk, od_imks))

        inmsk = fundus_msk - fovea_msk - od_msk

        # draw pathologies
        for ptype, (n_min, n_max) in config['pathologies'].items():
            smks = np.zeros_like(inmsk)  # segmentation
            imks = np.zeros_like(inmsk)  # instance
            n = rnd.randint(n_min, n_max)
            for rp, cp in sample_coord(inmsk, n):
                sr, sc = rnd.uniform(0.03, 0.15), rnd.uniform(0.03, 0.15)
                img, msk = draw_pathology(ptype, img, inmsk, rp, cp, sr, sc)
                smks += msk
                imks[rp, cp] = C_MASK
            smks = smks.clip(0, C_MASK)
            samples.append((ptype, n, smks, imks))

        # emit samples
        # grade = calc_grade(samples)
        for ptype, n, mask, imask in samples:
            # fp = 'segment_%s(x)' % ptype
            # yield (fp, img, mask)

            fp = 'hemifield_up(segment_%s(x), segment_fo(x))' % ptype
            yield (fp, img, hemifield(mask, True))

            # fp = 'hemifield_lo(segment_{0}(x))'.format(ptype)
            # yield (fp, img, hemifield(mask, False))

            # if ptype == 'ma':
            #     yield ('count_ma(segment_ma(x))', img, n)
            #     #yield ('segment_ma(x)', img, mask)


def find_blobs(sample):
    fp, img, mask = sample
    from skimage.feature import blob_log
    blobs_log = blob_log(mask, min_sigma=3, max_sigma=20, threshold=.1)
    blobs = np.zeros_like(mask)
    for r, c, _ in blobs_log:
        blobs[int(r), int(c)] = C_MASK
    return fp, img, mask, blobs


if __name__ == '__main__':
    from nutsflow import Consume, Map
    from nutsml import ViewImage

    ir, ic = 64, 64
    config = {'samples': 5,
               'pathologies': {'ha': [1, 1], 'ex': [2, 2], 'ma': [3, 3]}}
    gen_images(config, ir, ic) >> ViewImage(None, pause=10) >> Consume()

    # sg_conf = {'samples': 5,
    #            'pathologies': {'ha': [1, 1], 'ex': [2, 2], 'ma': [3, 3]}}
    # gen_samples(config, ir, ic) >> ViewImage((1, 2), pause=10) >> Consume()

    # sg_conf = {'samples': 100, 'pathologies': {'ex': [1, 5]}}
    # (gen_samples(sg_conf, ir, ic) >> Map(find_blobs) >>
    #  ViewImage((1, 2, 3), pause=100) >> Consume())
