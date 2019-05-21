"""
A simple example for counting pixels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nutsflow as nf
import nutsml as nm

from nstorch.losses import mse_loss, dice_loss
from nstorch.models import NSModule
from nstorch.base import BuildBatch, Train, Validate, Predict


def gen_samples(n=100, s=6):
    for i in range(n):
        img = np.zeros((1, s, s), dtype='float32')
        img[0, 0, 0] = 1
        img[0, 1, 1] = 1
        num = np.float32(2)
        mask = img.copy()
        yield ('count(filter(_))', img, num)
        yield ('filter(_)', [img], mask)
        # yield ('gt(_)', 10, 1)
        # yield ('gt(_)', 5, 0)


def view_samples(samples):
    samples >> nm.PrintColType() >> nm.ViewImage(1, pause=1) >> nf.Consume()


class CountModule(nn.Module):
    def __init__(self):
        super(CountModule, self).__init__()
        self.name = 'count'
        self.loss = mse_loss

    def forward(self, batch):
        return batch.sum((1, 2, 3))


class WeightSumModule(nn.Module):
    def __init__(self):
        super(WeightSumModule, self).__init__()
        self.name = 'wsum'
        self.loss = mse_loss
        self.c = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return (x * self.c).sum((1, 2, 3))


class GTSumModule(nn.Module):
    def __init__(self, device):
        super(GTSumModule, self).__init__()
        self.name = 'gt'
        self.loss = mse_loss
        self.threshold = nn.Parameter(torch.Tensor([1.0]))
        self.one = torch.ones(1, requires_grad=True, device=device)
        self.zero = torch.zeros(1, requires_grad=True, device=device)

    def forward(self, x):
        # return self.one if (x > self.threshold).all() else self.zero
        return torch.where(x > self.threshold, self.one, self.zero)


class FilterModule(nn.Module):
    def __init__(self):
        super(FilterModule, self).__init__()
        self.name = 'filter'
        self.loss = mse_loss
        self.layers = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=1, bias=True),
        )

    def forward(self, img_batch):
        return self.layers(img_batch)


if __name__ == '__main__':
    device = 'cuda'
    modules = [CountModule(), FilterModule(), GTSumModule(device)]
    model = NSModule(modules, device=device)
    model.optimizer = optim.Adam(model.parameters())

    # add_channel = nf.MapCol(1, lambda img: np.expand_dims(img, 0))

    samples = sorted(gen_samples(), key=lambda s: s[2].count('('))

    print('training...')
    for epoch in range(100):
        print('epoch', epoch)
        t_losses = (samples >> BuildBatch(2) >>
                    Train(model) >> nf.Collect())
        print('t_loss', np.mean(t_losses))
        v_losses = (gen_samples(10) >> BuildBatch(1) >>
                    Validate(model) >> nf.Collect())
        print('v_loss', np.mean(v_losses))

    print('predicting...')
    round = nf.MapCol(1, lambda x: torch.round(x + 0.4))
    (gen_samples(2) >> BuildBatch(1, outcol=None) >>
     Predict(model) >> round >> nf.Print() >> nf.Consume())
