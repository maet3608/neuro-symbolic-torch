"""
A simple example for counting pixels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nutsflow as nf
import nutsml as nm

from nstorch.modules import NSModule
from nstorch.batcher import BuildBatch
from nutsml.network import PytorchNetwork


def gen_samples(n=10, s=6):
    for i in range(n):
        img = np.zeros((1, s, s))
        img[0, 0, 0] = 1
        img[0, 0, 2] = 1
        mask = img.copy()
        yield (img, 2, 'count(filter(x))', )
        yield (img, mask, 'filter(x)',)


def mse_loss(input, target):
    iflat = input.view(-1)
    tflat = target.view(-1)
    diff = (iflat - tflat) ** 2
    return diff.mean()


def view_samples(samples):
    samples >> nm.PrintColType() >> nm.ViewImage(1, pause=1) >> nf.Consume()


class CountModule(nn.Module):
    def __init__(self):
        super(CountModule, self).__init__()
        self.name = 'count'

    def forward(self, batch):
        return batch.sum((1, 2, 3))


class WeightSumModule(nn.Module):
    def __init__(self):
        super(WeightSumModule, self).__init__()
        self.name = 'wsum'
        self.c = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return (x * self.c).sum(((1,2,3)))


class FilterModule(nn.Module):
    def __init__(self):
        super(FilterModule, self).__init__()
        self.name = 'filter'
        self.layers = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=1, bias=True),
        )

    def forward(self, img_batch):
        return self.layers(img_batch)


@nf.nut_processor
def Train(batches, network):
    model = network.model
    device = model.device
    for inputs, outputs, fp in batches:
        model.optimizer.zero_grad()
        inputs = [torch.as_tensor(b, device=device).float() for b in inputs]
        outputs = [torch.as_tensor(b, device=device).float() for b in outputs]
        y_pred = network.model(fp, inputs)
        y_true = outputs[0]
        loss = model.losses(y_pred, y_true)
        loss.backward()
        model.optimizer.step()
        yield loss.item()


@nf.nut_processor
def Predict(batches, network):
    model = network.model
    device = model.device
    with torch.no_grad():
        for inputs, outputs, fp in batches:
            model.optimizer.zero_grad()
            inputs = [torch.as_tensor(b, device=device).float() for b in inputs]
            y_pred = network.model.forward(fp, inputs)
            yield y_pred


if __name__ == '__main__':
    modules = [CountModule(), FilterModule(), WeightSumModule()]
    model = NSModule(modules)
    model.device = 'cuda'
    # model.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.optimizer = optim.Adam(model.parameters())
    model.losses = mse_loss
    network = PytorchNetwork(model, 'weights.pt')

    build_batch = BuildBatch(0,1,2, 2)
    # add_channel = nf.MapCol(1, lambda img: np.expand_dims(img, 0))

    samples = sorted(gen_samples(), key=lambda s: s[2].count('('))

    print('training...')
    for epoch in range(100):
        print('epoch', epoch)
        losses = (samples >> build_batch >>
                  Train(network) >> nf.Collect())
        loss = np.mean(losses)
        print('loss', loss)
        if loss < 1e-6: break

    print('predicting...')
    (samples >> build_batch >> Predict(network) >> nf.Flatten() >>
      nf.Tail(3) >> nf.Print() >> nf.Consume())
