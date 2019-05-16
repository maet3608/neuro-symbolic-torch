"""
.. module:: batcher
   :synopsis: Batch generator for neuro-symbolic computing.
"""

import numpy as np
import nutsflow as nf

from nutsflow.common import as_list
from nutsml.datautil import shapestr


def to_str(batch):
    return '[' + ', '.join(shapestr(b, True) for b in batch) + ']'

@nf.nut_processor
def BuildBatch(samples, incols, outcols, fpcol, max_batchsize, verbose=False):
    for group in samples >> nf.GroupBySorted(fpcol, nokey=True):
        for chunk in group >> nf.Chunk(max_batchsize):
            bc = list(zip(*chunk))  # batch columns
            fp = bc[fpcol][0]  # functional program
            assert isinstance(fp, str), 'No fp found in fpcol: ' + str(fp)
            inputs = [np.stack(bc[c]) for c in as_list(incols)]
            outputs = [np.stack(bc[c]) for c in as_list(outcols)]
            if verbose:
                print("batch in:{} out:{}, fp:{}".format(to_str(inputs),
                                                           to_str(outputs), fp))
            yield inputs, outputs, fp


