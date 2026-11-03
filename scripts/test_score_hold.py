"""Checks for the ADR-010 hold and rule, on synthetic traces and hand-written bounds.

usage: python test_score_hold.py
"""
import os
import tempfile

import numpy as np
import pandas as pd

from baselines import proposed_target
from eval_seeds import BUDGETS
from score_hold import (COMPUTED, DELAYS, GROUPS, LEARNED, MARGIN, NO_HOLD, computed_hold, check_published, decide,
                        encoded, guard, region_masks, summarise)
from train import make_target

rng = np.random.default_rng(0)
X = rng.integers(0, 101, (5, 400)).astype(np.float64)
VALID = np.arange(6, 400 - 4)        # the steps where the paper's target exists


def test_the_split_is_exact():
    whole = proposed_target(X)[:, VALID]
    known = computed_hold(X)[:, VALID]
    future = proposed_target(X, past=1, future=5)[:, VALID]
    assert np.array_equal(whole, np.maximum(known, future))


def test_the_split_is_exact_in_bins():
    whole = make_target(X, 'proposed')[:, VALID]
    future = make_target(X, 'fwdmax')[:, VALID]
    known = encoded(computed_hold(X))[:, VALID]
    assert np.array_equal((whole + 1) * 2.5, np.maximum(known, (future + 1) * 2.5))


def test_the_hold_uses_no_future():
    t = 200
    changed = X.copy()
    changed[:, t:] = 100.0
    assert np.array_equal(computed_hold(X)[:, :t + 1], computed_hold(changed)[:, :t + 1])


def test_the_hold_reaches_back_six_steps():
    x = np.zeros((1, 50))
    x[0, 20] = 40.0
    hold = computed_hold(x)[0]
    assert hold[20] == 0 and hold[21] == 40 and hold[26] == 40 and hold[27] == 0


def test_encoding_matches_the_models_bins():
    got = encoded(np.array([0.0, 55.0, 56.0, 57.5, 100.0, 120.0]))
    assert got.tolist() == [2.5, 57.5, 57.5, 60.0, 100.0, 100.0], got


def test_regions_split_every_step_once():
    after, runup, other = region_masks([[20], [20, 24]], (2, 40))
    assert (after.astype(int) + runup + other == 1).all()
    assert np.where(after[0])[0].tolist() == list(range(21, 27))
    assert np.where(runup[0])[0].tolist() == list(range(15, 21))
    assert not runup[1, 21:25].any() and after[1, 21:31].all()     # after wins where a burst overlaps


def test_the_rules_are_strict():
    fine = {1: 0.001, 3: 0.001}
    assert decide(fine, {1: -MARGIN + 1e-6, 3: -MARGIN + 1e-6})[1] == 'complete hypothesis supported'
    assert decide(fine, {1: -MARGIN, 3: -MARGIN + 1e-6})[1] != 'complete hypothesis supported'
    assert decide({1: 0.0, 3: 0.001}, {1: 0.0, 3: 0.0})[1] != 'complete hypothesis supported'


def test_the_outcomes():
    yes, no = {1: 0.001, 3: 0.001}, {1: -0.02, 3: -0.02}
    assert decide(yes, {1: 0.0, 3: 0.0})[1] == 'complete hypothesis supported'
    assert decide(yes, no)[1] == 'drop computed hold as a replacement for the learned hold'
    assert decide(no, no)[1] == 'drop computed hold as a replacement for the learned hold'
    per_delay, overall = decide(yes, {1: -0.02, 3: 0.0})
    assert overall == 'complete hypothesis not supported; the conclusion is delay-specific'
    assert per_delay[1] == 'not an acceptable replacement at this delay'
    assert per_delay[3] == 'both claims hold at this delay'
    assert 'not shown to be better overall' in decide({1: -0.001, 3: -0.001}, {1: 0.0, 3: 0.0})[1]


def fake_curves():
    shape = (len(BUDGETS), len(DELAYS), len(GROUPS), 6)
    return {(m, s): dict(cover=rng.random(shape), thr=rng.random((len(BUDGETS), len(DELAYS), 4, 6)) + 1)
            for m in (NO_HOLD, LEARNED, 'fwdmax6') for s in (0, 1)}


def published_tables(curves):
    diff, mean, lo, hi = summarise(curves, NO_HOLD, LEARNED, 'cover', (0, 1))
    paired = pd.DataFrame([dict(setting='base', pair=f'{NO_HOLD} - {LEARNED}', budget=0.8, d=3, group='follow-on',
                                mean=mean[4, 1, 2, 0], seed_min=diff[:, 4, 1, 2, 0].min(), seed_max=diff[:, 4, 1, 2, 0].max(),
                                ci_lo=lo[4, 1, 2], ci_hi=hi[4, 1, 2])])
    c = curves[NO_HOLD, 1]['thr']
    matched = pd.DataFrame([{'mode': NO_HOLD, 'seed': 1, 'budget': 0.8, **{f'throttle_core_h@{d}': c[4, j, 0, 0] for j, d in enumerate(DELAYS)}}])
    return paired, matched


def test_the_published_check_reads_the_right_cells():
    assert BUDGETS[4] == 0.8 and DELAYS[1] == 3 and GROUPS[2] == 'follow-on'
    curves = fake_curves()
    paired, matched = published_tables(curves)
    check_published(curves, (0, 1), paired, matched)
    for table, column in ((paired, 'ci_lo'), (matched, 'throttle_core_h@3')):
        good = table[column].copy()
        table[column] = good * 1.01
        try:
            check_published(curves, (0, 1), paired, matched)
        except AssertionError:
            table[column] = good
        else:
            raise AssertionError(f'a wrong {column} was not caught')


def test_the_guard_refuses_a_second_run():
    with tempfile.TemporaryDirectory() as out:
        try:
            guard(out, 'HEAD', os.getcwd())
        except SystemExit as e:
            assert 'happens once' in str(e)
        else:
            raise AssertionError('an existing output folder was not refused')


def test_the_primary_forecaster_is_fwdmax_with_the_encoded_hold():
    assert COMPUTED == 'fwdmax + encoded hold' and MARGIN == 0.01


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
