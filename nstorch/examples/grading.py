import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nutsflow as nf
import nutsml as nm

from nstorch.models import NSModule, Gt, Xor, Or, And, Not
from nstorch.base import BuildBatch, Train, Validate, Predict


def gen_samples():
    for ha_up in range(0, 5):
        for ha_lo in range(0, 5):
            is_grade0 = ha_up + ha_lo > 0
            is_grade1 = (ha_up > 2 and ha_lo <= 2) or (ha_up <= 2 and ha_lo > 2)
            yield 'Gt1(x[0] + x[1])', [ha_up, ha_lo], 0.9 if is_grade0 else 0.1
            yield 'Xor(Gt2(x[0]), Gt2(x[1]))', [ha_up, ha_lo], 0.9 if is_grade1 else 0.1


if __name__ == '__main__':
    device = 'cpu'
    modules = [Gt(1), Gt(2), Xor()]
    model = NSModule(modules, device=device)
    model.optimizer = optim.Adam(model.parameters(), lr=0.1)

    samples = sorted(gen_samples(), key=lambda s: s[0].count('('))

    print('training...')
    for epoch in range(100):
        print('epoch', epoch)
        t_losses = (samples >> BuildBatch(4) >>
                    Train(model) >> nf.Collect())
        print('t_loss', np.mean(t_losses))

    print('predicting...')
    samples = list(gen_samples())
    (samples >> BuildBatch(1, outcol=None) >> Predict(model) >>
     nf.Zip(samples >> nf.Get(1)) >> nf.Print() >> nf.Consume())
