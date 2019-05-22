"""
.. module:: modules
   :synopsis: Neuro-symbolic Pytorch modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from nstorch.losses import mse_loss


class NSModule(nn.Module):
    """A neuro-symbolic module"""

    def __init__(self, modules, device):
        """
        Constructor.

        :param list[nn.Module] modules: Iterable over modules
           Note that modules must have 'name' attribute and names
           must be unique.
        :param str device: Device, eg. 'gpu', 'cuda' or 'cuda:0'
        """
        super(NSModule, self).__init__()
        for m in modules:
            assert hasattr(m, 'name'), 'Module has no name: ' + str(m)
            assert hasattr(m, 'loss'), 'Module has no loss fn: ' + str(m)
            self.add_module(m.name, m)
        self.locals = {m.name: m for m in modules}
        self.device = device
        self.to(device)

    def loss(self, fp):
        """Return loss function for given functional program"""
        modname = fp.split('(')[0]  # last fn call is output module
        return self._modules[modname].loss

    def forward(self, fp, inputs):
        """
        Compute forward pass of network.

        :param str fp: Functional program.
        :param inputs inputs: Network inputs (batches)
        :return: Output of network
        """
        inputs = inputs[0] if len(inputs) == 1 else inputs
        self.locals['_'] = inputs
        for m in self._modules.values():
            m.context = inputs
        return eval(fp, None, self.locals)


class Gt(nn.Module):
    def __init__(self, threshold, trainable=False):
        super(Gt, self).__init__()
        self.name = 'Gt' + str(threshold)
        self.loss = mse_loss
        self.threshold = torch.Tensor([threshold])
        self.slope = torch.Tensor([1.0])
        if trainable:
            self.threshold = nn.Parameter(self.threshold)
            self.slope = nn.Parameter(self.slope)

    def forward(self, x):
        return torch.sigmoid(x * self.slope - self.threshold)


class Not(nn.Module):
    def __init__(self):
        super(Not, self).__init__()
        self.name = 'Not'
        self.loss = mse_loss

    def forward(self, x):
        return 1 - x


class And(nn.Module):
    def __init__(self, trainable=False):
        super(And, self).__init__()
        self.name = 'And'
        self.loss = mse_loss
        self.threshold = torch.Tensor([7.6])
        self.slope = torch.Tensor([5.5])
        if trainable:
            self.threshold = nn.Parameter(self.threshold)
            self.slope = nn.Parameter(self.slope)

    def forward(self, a, b):
        ret = torch.sigmoid((a + b) * self.slope - self.threshold)
        return ret


class Or(nn.Module):
    def __init__(self, trainable=False):
        super(Or, self).__init__()
        self.name = 'Or'
        self.loss = mse_loss
        self.threshold = torch.Tensor([3.2])
        self.slope = torch.Tensor([5.5])
        if trainable:
            self.threshold = nn.Parameter(self.threshold)
            self.slope = nn.Parameter(self.slope)

    def forward(self, a, b):
        ret = torch.sigmoid((a + b) * self.slope - self.threshold)
        return ret


class Xor(nn.Module):
    def __init__(self, trainable=True):
        super(Xor, self).__init__()
        self.name = 'Xor'
        self.loss = mse_loss
        self.threshold = torch.Tensor([-26.3])
        self.slope = torch.Tensor([-78])
        self.w1 = torch.Tensor([2.7])
        self.w2 = torch.Tensor([0.7])
        if trainable:
            self.threshold = nn.Parameter(self.threshold)
            self.slope = nn.Parameter(self.slope)
            self.w1 = nn.Parameter(self.w1)
            self.w2 = nn.Parameter(self.w2)

    def forward(self, a, b):
        sig = torch.sigmoid
        w1, w2, s, t = self.w1, self.w2, self.slope, self.threshold
        b1 = sig(a + b - w1)
        b2 = sig(-a - b - w2)
        return sig((b1 + b2) * s - t)
