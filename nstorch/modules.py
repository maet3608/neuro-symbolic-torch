"""
.. module:: modules
   :synopsis: Neuro-symbolic Pytorch modules.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import nstorch.parser as nsp

from collections import OrderedDict
from nutsflow import Window


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

        A functional program is described by an edge list of modules that
        must define a directed acyclic graph (DAG) of the execution flow, e.g.
        'i0|A A|B i1|B B|O' describes the DAG below

               i0 - A
                     \
                      B - O
                     /
               i1 --

        Edges can be chained, e.g. the following definitions are valid and
        equivalent:
        'i0|A A|B i1|B B|O'
        'i0|A|B|O i1|B'
        'i0|A|B i1|B|O'

        Order is preserved and modules are executed from left-to-right and
        top-to-bottom (for a DAG as displayed in the orientation above)

        :param str fp: Description of a functional program.
        :param inputs x: Network inputs (batches)
        :return: Output of network
        """
        # fp, inputs = x[-1], x[:-1]

        # print('NSModule, fp', fp)
        # print('NSModule, inputs', inputs)
        inputs = inputs[0] if len(inputs) == 1 else inputs
        self.locals['x'] = inputs
        for m in self._modules.values():
            m.context = inputs
        return eval(fp, None, self.locals)
        # return nsp.forward(fp, *inputs, modules)
