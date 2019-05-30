"""
.. module:: modules
   :synopsis: Neuro-symbolic Pytorch modules.
"""

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter
from torchsummary import summary
from nstorch.losses import mse_loss


class NSModule(nn.Module):
    """A neuro-symbolic module that is composed of sub-modules"""

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
        self.best_score = None  # score of best scoring network so far

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
        self.locals['x'] = inputs
        for m in self._modules.values():
            m.context = inputs
        return eval(fp, None, self.locals)

    def save_weights(self, filepath='weights.pt'):
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath='weights.pt'):
        self.load_state_dict(torch.load(filepath))

    def save_best(self, score, isloss=True, filepath='weights.pt'):
        """
        Save weights of best network

        :param float score: Score of the network, e.g. loss, accuracy
        :param bool isloss: True means lower score is better, e.g. loss
          and the network with the lower score score is saved.
        :param str filepath: Path to weights file.
        """

        if (not self.best_score or
                (isloss is True and score <= self.best_score) or
                (isloss is False and score >= self.best_score)):
            self.best_score = score
            self.save_weights(filepath)

    def print_layers(self, input_shape=None):
        """
        Print network architecture (and layer dimensions).

        :param tuple|None input_shape: (C, H, W) or None
               If None, layer dimensions and param numbers are not printed.
        """
        if input_shape:
            device = self.device[:4]  # remove GPU id, e.g. cuda:0
            summary(self, input_shape, device=device)
        else:
            print(str(self))


class Gt(nn.Module):
    """Greater Than"""

    def __init__(self, threshold, trainable=False):
        super(Gt, self).__init__()
        self.name = 'G_' + str(threshold)
        self.loss = mse_loss
        self.threshold = Parameter(Tensor([threshold]), requires_grad=trainable)
        self.slope = Parameter(Tensor([1.0]), requires_grad=trainable)

    def forward(self, x):
        return torch.sigmoid(x * self.slope - self.threshold)


class St(nn.Module):
    """Smaller Than"""

    def __init__(self, threshold, trainable=False):
        super(St, self).__init__()
        self.name = 'St_' + str(threshold)
        self.loss = mse_loss
        self.threshold = Parameter(Tensor([threshold]), requires_grad=trainable)
        self.slope = Parameter(Tensor([1.0]), requires_grad=trainable)

    def forward(self, x):
        return torch.sigmoid(-x * self.slope + self.threshold)


class Not(nn.Module):
    """Not"""

    def __init__(self):
        super(Not, self).__init__()
        self.name = 'Not'
        self.loss = mse_loss

    def forward(self, x):
        return 1 - x


class And(nn.Module):
    """And"""

    def __init__(self, trainable=False):
        super(And, self).__init__()
        self.name = 'And'
        self.loss = mse_loss
        self.threshold = Parameter(Tensor([7.6]), requires_grad=trainable)
        self.slope = Parameter(Tensor([5.5]), requires_grad=trainable)

    def forward(self, a, b):
        ret = torch.sigmoid((a + b) * self.slope - self.threshold)
        return ret


class Or(nn.Module):
    """Or"""

    def __init__(self, trainable=False):
        super(Or, self).__init__()
        self.name = 'Or'
        self.loss = mse_loss
        self.threshold = Parameter(Tensor([3.2]), requires_grad=trainable)
        self.slope = Parameter(Tensor([5.5]), requires_grad=trainable)

    def forward(self, a, b):
        ret = torch.sigmoid((a + b) * self.slope - self.threshold)
        return ret


class Xor(nn.Module):
    """Exclusive Or"""

    def __init__(self, trainable=True):
        super(Xor, self).__init__()
        self.name = 'Xor'
        self.loss = mse_loss
        self.threshold = Parameter(Tensor([-26.3]), requires_grad=trainable)
        self.slope = Parameter(Tensor([-78]), requires_grad=trainable)
        self.w1 = Parameter(Tensor([2.7]), requires_grad=trainable)
        self.w2 = Parameter(Tensor([0.7]), requires_grad=trainable)

    def forward(self, a, b):
        sig = torch.sigmoid
        w1, w2, s, t = self.w1, self.w2, self.slope, self.threshold
        b1 = sig(a + b - w1)
        b2 = sig(-a - b - w2)
        return sig((b1 + b2) * s - t)
