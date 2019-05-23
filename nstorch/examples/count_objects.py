"""
A simple example for counting pixels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nutsflow as nf
import nutsml as nm

from random import randint
from nstorch.losses import mse_loss, dice_loss
from nstorch.models import NSModule, Gt, Xor, Or, And, Not
from nstorch.base import BuildBatch, Train, Validate, Predict


def gen_samples(n=100, s=5):
    for i in range(n):
        img = np.zeros((1, s, s), dtype='float32')
        num_ma = randint(0, 4)
        for _ in range(num_ma):
            r, c = randint(0, s - 1), randint(0, s - 1)
            img[0, r, c] = 1
        num_ma = np.float32(np.sum(img > 0))
        mask = img.copy()
        lower, upper = count_hemifield(img)
        # yield ('CountObj(FilterObj(_))', img, num_ma)
        # yield ('FilterObj(_)', img, mask)
        yield 'CountObj(Hemifield_up(FilterObj(_)))', img, upper
        # yield 'Gt2(CountObj(Hemifield_up(FilterObj(_))))', img, int(upper > 2)


def count_hemifield(img):
    _, h, w = img.shape
    _, rr, cc = np.where(img > 0.5)
    upper = np.sum(rr < h / 2)
    lower = np.sum(rr >= h / 2)
    return lower, upper


def view_samples(samples):
    samples >> nm.PrintColType() >> nm.ViewImage(1, pause=1) >> nf.Consume()


class Hemifield(nn.Module):
    def __init__(self, isupper):
        super(Hemifield, self).__init__()
        self.name = 'Hemifield' + ('_up' if isupper else '_lo')
        self.loss = mse_loss
        self.isupper = isupper

    def forward(self, x):
        _, _, r, c = x.shape
        hemi = torch.zeros_like(x)
        if self.isupper:
            hemi[:, :, :r // 2, :] = 1
        else:
            hemi[:, :, r // 2:, :] = 1
        return (x * hemi)


class CountObj(nn.Module):
    def __init__(self):
        super(CountObj, self).__init__()
        self.name = 'CountObj'
        self.loss = mse_loss

    def forward(self, x):
        return x.sum((1, 2, 3))


class SumObj(nn.Module):
    def __init__(self):
        super(SumObj, self).__init__()
        self.name = 'SumObj'
        self.loss = mse_loss
        self.c = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return (x * self.c).sum((1, 2, 3))


class FilterObj(nn.Module):
    def __init__(self):
        super(FilterObj, self).__init__()
        self.name = 'FilterObj'
        self.loss = mse_loss
        self.layers = nn.Sequential(
            nn.Conv2d(1, 1, (3, 3), padding=1, bias=True),
            nn.Sigmoid(),
            # nn.Tanh(),
        )

    def forward(self, img_batch):
        return self.layers(img_batch)


def train(model):
    model.optimizer = optim.Adam(model.parameters(), lr=0.1)

    samples = sorted(gen_samples(), key=lambda s: s[0].count('('))
    print('training...')
    for epoch in range(100):
        print('epoch', epoch)
        t_losses = (samples >> BuildBatch(4) >>
                    Train(model) >> nf.Collect())
        print('t_loss', np.mean(t_losses))
        # v_losses = (gen_samples(10) >> BuildBatch(1) >>
        #             Validate(model) >> nf.Collect())
        # print('v_loss', np.mean(v_losses))

def predict(model):
    print('predicting...')
    samples = list(gen_samples(2))
    (samples >> BuildBatch(1, outcol=None) >> Predict(model) >>
     nf.Zip(samples >> nf.Get(1,3)) >> nf.Print() >> nf.Consume())

def demo(model):
    print('speaking...')
    from speech import text2speech, speech2text

    samples = list(gen_samples(1))
    preds = samples >> BuildBatch(1, outcol=None) >> Predict(
        model) >> nf.Collect()
    #print(samples[0][1])
    n = round(preds[0][1].numpy() + 0.4)
    #print(n)
    text2speech('I am listening')
    while True:
        recognized = speech2text()
        if 'how many' in recognized:
            print(n)
            text2speech('there are %d objects' % n)
        elif 'show me' in recognized:
            print(samples[0][1])
            text2speech('please have a look')
        elif 'quit' in recognized:
            text2speech('Thank you')
            break
        else:
            text2speech('I do not understand ' + recognized)


if __name__ == '__main__':
    torch.set_printoptions(precision=1, sci_mode=False)
    device = 'cpu'
    modules = [CountObj(), FilterObj(), Gt(2), Xor(), Hemifield(True)]
    model = NSModule(modules, device=device)

    train(model)
    predict(model)
    #demo(model)


