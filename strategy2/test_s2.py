"""Checks for train_s2.py: the altered labels and the loss, against the authors' own code.

usage: python test_s2.py
"""
import os
import sys

import torch
import torch.nn.functional as F

from train_s2 import (ALPHA, MIN_GAIN, REDUCTION, TEMPERATURE, VARIANTS, altered_labels, is_new_best, s2_loss,
                      weights_above)
from train import NCLS

UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'upstream', 'ResourceForecasting')
torch.manual_seed(0)
CLS = torch.tensor([0, 1, 7, 20, 31, 38, 39])
OUT = torch.randn(len(CLS), NCLS)


def authors_label(c, variant):
    """One sample's altered label, written as the authors' data types write it."""
    w = torch.zeros(NCLS)
    if variant == 'single':
        w[c] = 1
    elif variant == 'binary':
        w[c:] = 1
    else:
        if variant == 'proportional':
            proportion = torch.arange(1, 0, REDUCTION[variant] / NCLS, dtype=torch.float)
        else:
            proportion = torch.ones(NCLS, dtype=torch.float)
            for index in range(1, NCLS):
                proportion[index:] /= REDUCTION[variant]
        n = len(w[c:(c + proportion.shape[0])])
        w[c:(c + proportion.shape[0])] = proportion[:(NCLS - c)][:n]
    return w


def test_labels_match_the_authors_construction():
    for variant in VARIANTS:
        mine = altered_labels(CLS, variant)
        theirs = torch.stack([authors_label(int(c), variant) for c in CLS])
        assert torch.allclose(mine, theirs, atol=1e-6), variant


def test_labels_put_nothing_below_the_true_class():
    for variant in VARIANTS:
        w = altered_labels(CLS, variant)
        for row, c in zip(w, CLS):
            assert not row[:c].any() and row[c] == 1, variant


def test_loss_equals_the_authors_cdce():
    sys.path.insert(0, UPSTREAM)
    from app.auxiliary_files.model_methods.loss_functions import CdCE
    theirs = CdCE({'alpha': ALPHA, 'temperature': TEMPERATURE}, 'cpu')
    for variant in VARIANTS:
        w = altered_labels(CLS, variant)
        assert torch.isclose(s2_loss(OUT, CLS, w), theirs.run(OUT, (CLS, w)), rtol=1e-5), variant


def test_the_summed_loss_is_the_mean_loss_times_the_batch():
    w = altered_labels(CLS, 'proportional')
    assert torch.isclose(s2_loss(OUT, CLS, w, reduction='sum'), s2_loss(OUT, CLS, w) * len(CLS), rtol=1e-6)


def test_the_softmaxed_labels_are_the_ones_in_the_adr():
    c = torch.tensor([10])
    p = {v: F.softmax(altered_labels(c, v) / TEMPERATURE, dim=1)[0] for v in VARIANTS}
    assert round(p['single'][10].item(), 3) == 0.998
    assert round(p['exponential'][10].item(), 3) == 0.973 and round(p['exponential'][11].item(), 3) == 0.023
    assert round(p['proportional'][10].item(), 3) == 0.632 and round(p['proportional'][11].item(), 3) == 0.232
    assert torch.allclose(p['binary'][10:], p['binary'][10]) and p['binary'][:10].sum() < 0.002


def test_the_settings_are_the_ones_in_the_adr():
    assert ALPHA == 0.99 and TEMPERATURE == 0.1 and REDUCTION == {'proportional': -4.0, 'exponential': 1.6}
    assert len(weights_above('proportional')) == 10
    assert torch.allclose(weights_above('proportional'), torch.linspace(1.0, 0.1, 10))


def test_a_new_best_needs_the_same_share_of_the_loss_as_train_py():
    assert abs(MIN_GAIN - 1e-4 / 0.77) < 1e-6
    assert is_new_best(0.5, float('inf'))
    assert is_new_best(0.036160, 0.036171)       # a gain of 1.1e-5, which a fixed 1e-4 ignores
    assert not is_new_best(0.036170, 0.036171)   # a gain of 1e-6, under 0.013% of the loss
    assert not is_new_best(0.036200, 0.036171)


def test_the_loss_falls_when_the_output_moves_towards_the_label():
    w = altered_labels(CLS, 'binary')
    towards = OUT + 2.0 * w
    assert s2_loss(towards, CLS, w) < s2_loss(OUT, CLS, w)


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
