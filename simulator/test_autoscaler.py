"""Checks for autoscaler.py on synthetic traces and hand-written cases.

usage: python test_autoscaler.py
"""
import os
import tempfile

import numpy as np
import pandas as pd

from autoscaler import (COMPUTED, JUDGED, LEARNED, MEASURES, METHODS, NO_HOLD, SETTINGS, THROTTLE, alloc_sums,
                        check_replay, controller, decide, guard, queued)
from eval_seeds import BUDGETS
from metrics import START, evaluate_trace
from score_types import DELAYS, prepare, spike_index

rng = np.random.default_rng(0)
N, T = 6, 1300
X = 20 + rng.normal(0, 1, (N, T)).clip(-3, 3)
for row in X:
    for k in rng.choice(np.arange(800, T - 10), 12, replace=False):
        row[k:k + 2] += 30
Y = np.zeros_like(X)
Y[:, 1:] = 1.2 * X[:, :-1]                                           # a forecast that repeats the last value
Y = prepare(Y)
REQ = rng.integers(1, 9, N).astype(float)


def one(values, d=1, window=0, tol=0.0):
    return controller(np.array([values], dtype=float), d, window, tol)[0].tolist()


def test_replay_is_the_published_allocation():
    for d in DELAYS:
        A = controller(Y, d, *SETTINGS['replay'])
        assert np.array_equal(A[:, d - 1:], Y[:, :T - d + 1]) and not A[:, :d - 1].any()


def test_replay_scores_as_the_published_metrics():
    tid, pos = spike_index(X)
    everyone = np.ones((len(tid), 1), bool)
    for j, d in enumerate(DELAYS):
        slack, sums = alloc_sums(X, controller(Y, d, 0, 0.0), REQ, tid, pos, everyone)
        for i in range(N):
            e = evaluate_trace(X[i], Y[i], REQ[i])
            assert sums[0, i] == e[f'cover@{d}']
            assert np.isclose(sums[1, i], e[f'throttle_core_h@{d}'])
            assert np.isclose(slack[i] / 100 * REQ[i] * 5 / 60, e[f'slack_core_h@{d}'])


def test_the_delay_shifts_the_forecast():
    assert one([1, 2, 3, 4, 5], d=3) == [0, 0, 1, 2, 3]


def test_the_window_delays_decreases_only():
    assert one([1, 5, 1, 1, 1], window=1) == [1, 5, 5, 1, 1]
    assert one([1, 5, 1, 1, 1], window=2) == [1, 5, 5, 5, 1]
    assert one([1, 1, 5, 6, 7], window=2) == [1, 1, 5, 6, 7]         # increases pass at once
    A = controller(Y, 3, 1, 0.0)
    assert (A >= controller(Y, 3, 0, 0.0)).all()


def test_the_tolerance_skips_small_changes():
    assert one([100, 105, 95, 111, 100], tol=0.10) == [100, 100, 100, 111, 111]
    assert one([100, 110, 121], tol=0.10) == [100, 100, 121]         # exactly 10% does not move
    assert one([0, 0, 5], tol=0.10) == [0, 0, 5]                     # from zero, any change moves


def test_the_controller_uses_no_future():
    t = 700
    changed = Y.copy()
    changed[:, t + 1:] = 99.0
    for d in DELAYS:
        for window, tol in SETTINGS.values():
            a, b = controller(Y, d, window, tol), controller(changed, d, window, tol)
            assert np.array_equal(a[:, :t + d], b[:, :t + d])       # step t + d - 1 uses the forecast for t


def test_changes_and_allocation_are_counted_from_the_first_scored_step():
    tid, pos = spike_index(X)
    A = np.full_like(X, 50.0)
    A[:, START + 10:] = 60.0
    A[:, :START] = 0.0                                               # the jump into START counts as a change
    _, sums = alloc_sums(X, A, REQ, tid, pos, np.ones((len(tid), 1), bool))
    changes, alloc = sums[MEASURES.index('allocation changes') - 1], sums[MEASURES.index('allocated core-hours') - 1]
    assert (changes == 2).all()
    assert np.allclose(alloc, (50.0 * 10 + 60.0 * (T - START - 10)) * REQ / 100 * 5 / 60)


def test_the_queue_carries_unserved_cpu_forward():
    x = np.zeros((1, START + 4))
    a = np.zeros((1, START + 4))
    x[0, START:] = [10, 0, 0, 0]
    a[0, START:] = [4, 4, 4, 4]
    assert queued(x, a)[0] == 6 + 2 + 0 + 0                          # 10 arrives, 4 served per step
    assert queued(x, a + 10)[0] == 0


def test_the_rule_reads_the_whole_interval():
    per_delay, overall = decide({1: -9.0, 3: -5.0}, {1: -1.0, 3: -0.1})
    assert overall.startswith("the replay's throttling gain carries over")
    per_delay, overall = decide({1: -9.0, 3: -5.0}, {1: -1.0, 3: 0.0})
    assert overall == 'the conclusion is reported per delay'
    assert per_delay[1].startswith('the throttling gain carries over') and per_delay[3].startswith('not shown')
    per_delay, _ = decide({1: 0.5, 3: -5.0}, {1: 2.0, 3: 4.0})
    assert per_delay[1].startswith('the result reverses') and per_delay[3].startswith('not shown')


def test_the_replay_check_catches_a_wrong_table():
    published = {(m, s): rng.random((len(DELAYS), len(BUDGETS), 3, 4)) + 0.5 for m in METHODS for s in (0, 1)}
    names = ('cover: all', 'cover: follow-on', 'throttle core-hours: all')
    table = pd.DataFrame([dict(method=m, seed=s, d=d, budget=b, measure=name, value=float(f'{v[j, BUDGETS.index(b), k, 0]:.6g}'))
                          for (m, s), v in published.items() for j, d in enumerate(DELAYS) for b in (0.6, 0.8, 1.0)
                          for k, name in enumerate(names)]
                         + [dict(method='fwdmax', seed=0, d=np.nan, budget=0.8, measure='slack share: other', value=0.5)])
    check_replay('test', published, (0, 1), table)
    table.loc[5, 'value'] *= 1.01
    try:
        check_replay('test', published, (0, 1), table)
    except AssertionError:
        pass
    else:
        raise AssertionError('a wrong published value was not caught')


def test_the_guard_refuses_a_second_run():
    with tempfile.TemporaryDirectory() as out:
        try:
            guard(out, 'HEAD', os.getcwd())
        except SystemExit as e:
            assert 'happens once' in str(e)
        else:
            raise AssertionError('an existing output folder was not refused')


def test_the_judged_setting_and_methods_are_the_ones_in_the_adr():
    assert JUDGED == 'controller' and SETTINGS == {'replay': (0, 0.0), 'controller': (1, 0.10)}
    assert METHODS[COMPUTED] == ('fwdmax', 'encoded') and METHODS[LEARNED] == ('proposed', None)
    assert METHODS[NO_HOLD] == ('fwdmax', None) and MEASURES[THROTTLE] == 'throttled core-hours'


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
