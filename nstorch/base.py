"""
.. module:: base
   :synopsis: Basic functions/nuts for neuro-symbolic computing.
"""
import torch

import numpy as np
import nutsflow as nf

from nutsflow.common import console
from nutsml.datautil import batchstr


def to_list(x):
    """
    Wraps x in a list if it is not already a list.

    :param object x: Any object.
    :return: x wrapped in list
    :rtype: list
    """
    return x if isinstance(x, list) else [x]


def to_array(cols):
    """Convert data columns in batch to numpy arrays"""
    return [np.stack(c) for c in zip(*map(to_list, cols))]


def to_tensor(elements, device):
    return [torch.as_tensor(e, device=device).float() for e in elements]


@nf.nut_processor
def BuildBatch(samples, batchsize, fpcol=0, incol=1, outcol=2, verbose=False):
    """
    Build batches for neuro-symbolic networks.

    :param iterable samples: Iterable over samples
    :param int batchsize: Maximum batchsize. Actual batchsize might be smaller.
    :param int incol: Column in sample that contain network inputs.
    :param int|None outcol: Column in sample that contain network outputs.
           Set to None to generate batches for prediction
    :param int fpcol: Column in sample that contain functional program.
    :param bool verbose: If True, print batch format
    :return: iterator over batches of format [[inputs], [outputs], fp]
    :rtype: list
    """
    for group in samples >> nf.GroupBySorted(fpcol, nokey=True):
        for batch in group >> nf.Chunk(batchsize):
            bc = list(zip(*batch))  # batch columns
            fp = bc[fpcol][0]  # functional program
            assert isinstance(fp, str), 'No fp found in fpcol: ' + str(fp)
            assert all(f == fp for f in bc[fpcol]), 'Expect same fp in batch'
            inputs = to_array(bc[incol])
            if outcol is not None:
                outputs = to_array(bc[outcol])
                if verbose:
                    fmtstr = "batch in:{} out:{}, fp:{}"
                    console(fmtstr.format(batchstr(inputs), batchstr(outputs), fp))
                yield fp, inputs, outputs
            else:
                if verbose:
                    fmtstr = "batch in:{} fp:{}"
                    console(fmtstr.format(batchstr(inputs), fp))
                yield fp, inputs


@nf.nut_processor
def Train(batches, model):
    device = model.device
    for fp, inputs, outputs in batches:
        model.optimizer.zero_grad()
        inputs = to_tensor(inputs, device)
        outputs = to_tensor(outputs, device)
        y_pred = model(fp, inputs)
        y_true = outputs[0] if len(outputs) == 1 else outputs
        loss = model.loss(fp)(y_pred, y_true)
        loss.backward()
        model.optimizer.step()
        yield loss.item()


@nf.nut_processor
def Validate(batches, model):
    device = model.device
    with torch.no_grad():
        for fp, inputs, outputs in batches:
            inputs = to_tensor(inputs, device)
            outputs = to_tensor(outputs, device)
            y_pred = model(fp, inputs)
            y_true = outputs[0] if len(outputs) == 1 else outputs
            loss = model.loss(fp)(y_pred, y_true)
            yield loss.item()


@nf.nut_processor
def Predict(batches, model):
    device = model.device
    with torch.no_grad():
        for fp, inputs in batches:
            inputs = to_tensor(inputs, device)
            preds = model(fp, inputs)
            for pred in preds:  # flatten batch of predictions
                yield fp, pred


if __name__ == '__main__':
    i1 = np.expand_dims(np.array([[1, 1], [1, 1]]), 0)
    i2 = np.expand_dims(np.array([[2, 2], [2, 2]]), 0)
    i3 = np.expand_dims(np.array([[3, 3], [3, 3]]), 0)
    # samples = [ (1,2,'myfp'), (1,2,'myfp')]
    samples = [('myfp', [i1, i2], i3), ('myfp', [i1, i2], i3)]
    samples >> BuildBatch(2) >> nf.Consume()
