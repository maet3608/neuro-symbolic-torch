"""
Train segmentation network.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import nutsflow as nf
import nutsml as nm
import numpy as np

from nstorch.models import NSModule
from nstorch.base import BuildBatch, Train, Predict
from nstorch.losses import mse_loss, dice_loss

IC, IH, IW = 3, 128, 128  # image dimensions
EPOCHS = 1
BATCHSIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@nf.nut_function
def AdjChannels(sample):
    fp, inp, out = sample
    inp = np.moveaxis(inp, -1, 0)  # move channel axis to front
    outp = np.expand_dims(out, 0)  # add channel axis
    return fp, inp, outp


def train(model, samples, epochs):
    """Train network for given number of epochs"""
    model.optimizer = optim.Adam(model.parameters(), amsgrad=False,
                                 lr=0.00057, weight_decay=1.55e-05)
    # plot = nm.PlotLines(None)
    for epoch in range(epochs):
        losses = (samples >> AdjChannels() >>
                  BuildBatch(BATCHSIZE) >> nf.Shuffle(100) >>
                  Train(model) >> nf.Collect())
        loss = np.mean(losses)
        print(epoch, 'train loss:', loss)
        # plot(loss)
        model.save_best(loss, True, 'best_weights.pt')
        if loss > 1.1 * model.best_score:  # reset weights
            model.load_weights('best_weights.pt')
        if loss < 0.001:
            break

    return model.best_score
