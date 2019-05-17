"""
.. module:: batcher
   :synopsis: Batch generator for neuro-symbolic computing.
"""

import numpy as np
import nutsflow as nf

from nutsflow.common import console
from nutsml.datautil import batchstr


def to_list(x):
    return x if isinstance(x, list) else [x]





def to_array(cols):
    return [np.stack(c) for c in zip(*map(to_list, cols))]


@nf.nut_processor
def BuildBatch(samples, batchsize, incol=0, outcol=1, fpcol=2, verbose=True):
    for group in samples >> nf.GroupBySorted(fpcol, nokey=True):
        for batch in group >> nf.Chunk(batchsize):
            bc = list(zip(*batch))  # batch columns
            fp = bc[fpcol][0]  # functional program
            assert isinstance(fp, str), 'No fp found in fpcol: ' + str(fp)
            inputs = to_array(bc[incol])
            outputs = to_array(bc[outcol])
            if verbose:
                fmtstr = "batch in:{} out:{}, fp:{}"
                console(fmtstr.format(batchstr(inputs), batchstr(outputs), fp))
            yield inputs, outputs, fp


if __name__ == '__main__':
    i1 = np.expand_dims(np.array([[1, 1], [1, 1]]), 0)
    i2 = np.expand_dims(np.array([[2, 2], [2, 2]]), 0)
    i3 = np.expand_dims(np.array([[3, 3], [3, 3]]), 0)
    # samples = [ (1,2,'myfp'), (1,2,'myfp')]
    samples = [([i1, i2], i3, 'myfp'), ([i1, i2], i3, 'myfp')]
    samples >> BuildBatch(2) >> nf.Consume()
