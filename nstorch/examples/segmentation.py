"""
(instance) segmentation of pathologies in synthetic fundus images.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
import nutsflow as nf
import nutsml as nm
import numpy as np

from nstorch.models import NSModule
from nstorch.base import BuildBatch, Train, Validate, Predict
from nstorch.losses import mse_loss, dice_loss
from fundus_generator import gen_samples, C_PATHO, C_MASK

C, H, W = 3, 64, 64  # image dimensions
EPOCHS = 0
BATCHSIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEGNETS = ['od', 'fo', 'ha', 'ex', 'ma']
SGCONF = {'samples': 100,
          'pathologies': {
              'ha': [0, 3],
              'ex': [0, 5],
              'ma': [0, 10]
          }}


class Gauss(nn.Module):
    def __init__(self):
        super(Gauss, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([1.0]))
        self.std = nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        return torch.exp((-(x - self.mean) ** 2) / (2 * self.std ** 2))


class Segment(nn.Module):
    """Fake, perfect pathology  segmentation based on color"""

    def __init__(self, name):
        super(Segment, self).__init__()
        self.name = 'segment_' + name
        self.loss = dice_loss

        color = C_PATHO[name]
        grid = [[color for _ in range(W)] for _ in range(H)]
        self.color = torch.Tensor(grid).permute(2, 0, 1).to(DEVICE)
        self.c = nn.Parameter(torch.Tensor([C_MASK]))

    def forward(self, x):
        ret = (x == self.color).sum(1).gt(2).float() * self.c
        # ret = (x == self.color).sum(1).gt(2).float()
        return ret.unsqueeze(1)  # add color channel


class __Segment(nn.Module):
    """Model for pathology segmentation"""

    def __init__(self, name):
        super(Segment, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3),
            nn.Conv2d(1, 1, kernel_size=5, padding=2),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.name = 'segment_' + name
        self.loss = dice_loss

    def forward(self, x):
        return self.layers(x)


class Instance(nn.Module):
    """Model for instance segmentation"""

    def __init__(self, name):
        super(Instance, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.name = 'instance_' + name
        self.loss = dice_loss

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return F.sigmoid(x)


class __Hemifield(nn.Module):
    """Filter given segmentation for upper or lower hemifield"""

    def __init__(self, isupper):
        super(__Hemifield, self).__init__()
        self.name = 'hemifield_' + ('up' if isupper else 'lo')
        self.loss = dice_loss
        self.isupper = isupper
        self.h = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x):
        hemi = torch.zeros_like(x)
        r = int(H * self.h)
        if self.isupper:
            hemi[:, :, :r, :] = 1
        else:
            hemi[:, :, r:, :] = 1
        return x * hemi


class Hemifield(nn.Module):
    """Filter given segmentation for upper or lower hemifield
    where hemifield is defined as by horizontal line through fovea"""

    def __init__(self, isupper):
        super(Hemifield, self).__init__()
        self.name = 'hemifield_' + ('up' if isupper else 'lo')
        self.loss = dice_loss
        self.isupper = isupper

    def forward(self, x, od_mask):
        hemi = torch.zeros_like(x, requires_grad=False)
        row_sum = od_mask.sum(dim=3)
        r_max = row_sum.argmax(dim=2).int()
        for r, b in zip(r_max, range(BATCHSIZE)):
            if self.isupper:
                hemi[b, :, :r, :] = 1
            else:
                hemi[:, :, r:, :] = 1
        return x * hemi


class __Count(nn.Module):
    def __init__(self, name):
        super(Count, self).__init__()
        self.name = 'count_' + name
        self.loss = mse_loss
        self.c = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return (x * self.c).sum((1, 2, 3))


class Count(nn.Module):
    def __init__(self, name):
        super(Count, self).__init__()
        self.name = 'count_' + name
        self.loss = mse_loss

    def forward(self, x):
        from skimage.feature import blob_log
        counts = []
        s = 1 if self.name == 'count_ma' else 3  # smaller sigma for microano
        for mask in x:
            mask = np.squeeze(mask.cpu().detach().numpy())
            blobs = blob_log(mask, min_sigma=s, max_sigma=20, threshold=.1)
            cnt = len(blobs)
            counts.append(cnt)
        ret = torch.FloatTensor(counts).to(DEVICE)
        return ret


@nf.nut_function
def AdjChannels(sample):
    fp, inp, out = sample
    inp = np.moveaxis(inp, -1, 0)  # move channel axis to front
    outp = np.expand_dims(out, 0)  # add channel axis
    return fp, inp, outp


def train(model, samples, epochs):
    """Train network for given number of epochs"""
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


def predict(model, samples):
    """Compute network outputs and view segmentation"""
    diff = lambda s: (s[1], s[2], s[4], np.abs(s[2] - s[4]))
    isimage = lambda s: hasattr(s[2], 'shape')
    tensor2img = lambda t: np.squeeze(t.cpu().numpy())
    preds = (samples >> nf.PrintProgress(samples) >> AdjChannels() >>
             BuildBatch(BATCHSIZE, outcol=None) >> Predict(model) >>
             nf.Collect())

    view = nm.ViewImage((0, 1, 2, 3), pause=1000, figsize=(17, 5))
    (samples >> nm.PrintColType() >> nf.Append(preds) >> nf.Filter(isimage) >>
     nf.MapCol(4, tensor2img) >> nf.Map(diff) >> view >> nf.Consume())

    # (samples >> nf.Append(preds) >> nf.FilterFalse(isimage) >> nf.Get(2, 5) >>
    #  nf.Print() >> nf.Consume())


def print_parameters(model):
    print('model parameters')
    for n, p in model.named_parameters():
        print(n, p.shape, p.requires_grad)


if __name__ == '__main__':
    samples = sorted(gen_samples(SGCONF, H, W), key=lambda s: s[0])
    # samples >> nm.ViewImage((0, 1), pause=1) >> nf.Consume()

    print('creating vocabulary...')
    submods = []
    for name in SEGNETS:
        submods.append(Segment(name))
        # submods.append(Instance(name))
        submods.append(Count(name))
    submods.append(Hemifield(True))

    print('creating model...')
    model = NSModule(submods, device=DEVICE)
    model.print_layers()
    print_parameters(model)

    model.optimizer = optim.Adam(model.parameters(), amsgrad=False,
                                  lr=0.00057, weight_decay=1.55e-05)

    print('training ...')
    # model.load_weights('best_weights.pt')
    train(model, samples, epochs=EPOCHS)

    print('predicting ...')
    # model.load_weights('best_weights.pt')
    random.shuffle(samples)
    predict(model, samples)
