"""
(instance) segmentation of pathologies in synthetic fundus images.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim

import nutsflow as nf
import nutsml as nm
import numpy as np

from nstorch.models import NSModule, Gt, St, Xor, Or, Not
from nstorch.base import BuildBatch, Train, Predict
from nstorch.losses import mse_loss, dice_loss
from fundus_generator import gen_samples, C_PATHO, C_MASK

IC, IH, IW = 3, 64, 64  # image dimensions
EPOCHS = 1
BATCHSIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OBJS = ['fu', 'od', 'fo', 'ha', 'ex', 'ma']
CONFIG = {'samples': 100,
          'pathologies': {
              'ha': [0, 2],
              'ex': [0, 2],
              'ma': [0, 5]
          }}


class Segment(nn.Module):
    """Fake, perfect pathology segmentation based on color"""

    def __init__(self, name):
        super(Segment, self).__init__()
        self.name = 'seg_' + name
        self.loss = dice_loss

        color = C_PATHO[name]
        grid = [[color for _ in range(IW)] for _ in range(IH)]
        self.color = torch.Tensor(grid).permute(2, 0, 1).to(DEVICE)
        self.c = nn.Parameter(torch.Tensor([C_MASK]))

    def forward(self, x):
        ret = (x == self.color).sum(1).gt(2).float() * self.c
        # ret = (x == self.color).sum(1).gt(2).float()
        return ret.unsqueeze(1)  # add color channel


class Hemifield(nn.Module):
    """Filter given segmentation for upper or lower hemifield
    where hemifield is defined as by horizontal line through fovea"""

    def __init__(self, isupper):
        super(Hemifield, self).__init__()
        self.name = 'hem_' + ('up' if isupper else 'lo')
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


class Count(nn.Module):
    def __init__(self, name):
        super(Count, self).__init__()
        self.name = 'cnt_' + name
        self.loss = mse_loss

    def forward(self, x):
        from skimage.feature import blob_log
        sigmas = {'cnt_ma': 1, 'cnt_fu':20}
        s = sigmas.get(self.name, 3)
        counts = []
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


def predict_one(model, fn, img):
    img = np.moveaxis(img, -1, 0)  # move channel axis to front
    build_batch = BuildBatch(1, outcol=None)
    preds = [(fn, img)] >> build_batch >> Predict(model) >> nf.Collect()
    pred = preds[0][1]
    return pred.cpu().numpy()


def print_parameters(model):
    print('model parameters')
    for n, p in model.named_parameters():
        print(n, p.shape, p.requires_grad)


def complexity(sample):
    """Sort key for samples according to functional program.
    By number of function calls first and then alphabetical"""
    fn = sample[0]
    return fn.count('('), fn


def create_samples():
    samples = sorted(gen_samples(CONFIG, IH, IW), key=complexity)
    # samples >> nm.ViewImage((0, 1), pause=1) >> nf.Consume()
    return samples


def create_model():
    mods = []
    for name in OBJS:
        mods.append(Segment(name))
        mods.append(Count(name))
    mods.append(Hemifield(True))
    mods.append(Hemifield(False))
    mods.append(Gt(0))
    mods.append(Gt(2))
    mods.append(Xor())
    mods.append(Or())
    mods.append(Not())
    return NSModule(mods, device=DEVICE)


if __name__ == '__main__':
    print('creating model...')
    model = create_model()
    model.print_layers()
    print_parameters(model)

    samples = create_samples()

    print('training ...')
    # model.load_weights('best_weights.pt')
    train(model, samples, epochs=EPOCHS)
    #
    # print('predicting ...')
    # # model.load_weights('best_weights.pt')
    # random.shuffle(samples)
    # predict(model, samples)

    # print('predicting one...')
    # model.load_weights('best_weights.pt')
    # fn, img, _ = samples[0]
    # print(fn)
    # print(predict_one(model, fn, img))
