"""
.. module:: losses
   :synopsis: Common loss functions
"""

import torch


def mse_loss(pred, tar):
    iflat = pred.view(-1)
    tflat = tar.view(-1)
    diff = (iflat - tflat) ** 2
    return diff.mean()


def dice_loss(pred, tar):
    smooth = 1.0
    iflat = pred.view(-1)
    tflat = tar.view(-1)
    intersection = (iflat * tflat).sum()
    denom = (2. * intersection + smooth)
    nom = (iflat.sum() + tflat.sum() + smooth)
    return torch.abs(1 - denom / nom)
