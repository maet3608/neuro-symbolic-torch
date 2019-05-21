"""
.. module:: modules
   :synopsis: Neuro-symbolic Pytorch modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim


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

