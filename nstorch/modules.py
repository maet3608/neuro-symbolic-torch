"""
.. module:: modules
   :synopsis: Neuro-symbolic Pytorch modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class NSModule(nn.Module):
    """A neuro-symbolic module"""

    def __init__(self, modules):
        """
        Constructor.

        :param list[nn.Module] modules: Iterable over modules
           Note that modules must have 'name' attribute and names
           must be unique.
        """
        super(NSModule, self).__init__()
        for m in modules:
            assert hasattr(m, 'name'), 'Module has no name: ' + str(m)
            self.add_module(m.name, m)
        self.locals = {m.name: m for m in modules}

    def forward(self, fp, inputs):
        """
        Compute forward pass of network.

        :param str fp: Functional program.
        :param inputs inputs: Network inputs (batches)
        :return: Output of network
        """

        # print('NSModule, fp', fp)
        # print('NSModule, inputs', inputs)
        inputs = inputs[0] if len(inputs) == 1 else inputs
        self.locals['x'] = inputs
        for m in self._modules.values():
            m.context = inputs
        return eval(fp, None, self.locals)

