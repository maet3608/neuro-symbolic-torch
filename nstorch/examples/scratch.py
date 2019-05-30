import torch
import torch.nn.functional as F
import torch.nn as nn

from nstorch.losses import mse_loss, dice_loss


class Gauss(nn.Module):
    def __init__(self):
        super(Gauss, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([1.0]))
        self.std = nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        return torch.exp((-(x - self.mean) ** 2) / (2 * self.std ** 2))


class _Count(nn.Module):
    def __init__(self, name):
        super(_Count, self).__init__()
        self.name = 'count_' + name
        self.loss = mse_loss
        self.c = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        return (x * self.c).sum((1, 2, 3))


class _Hemifield(nn.Module):
    """Filter given segmentation for upper or lower hemifield"""

    def __init__(self, isupper):
        super(_Hemifield, self).__init__()
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


class _Segment(nn.Module):
    """Model for pathology segmentation"""

    def __init__(self, name):
        super(_Segment, self).__init__()
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
