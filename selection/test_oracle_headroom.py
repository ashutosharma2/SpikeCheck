"""Checks for oracle_headroom.py on synthetic traces and hand-built sums.

usage: python test_oracle_headroom.py
"""
import numpy as np

from oracle_headroom import (best_fixed, build_parts, fixed_values, hindsight, mix_curve, range_sums, read_at,
                             split_step, POLICIES)
from eval_seeds import BUDGETS
from metrics import START, evaluate_trace
from score_types import DELAYS, at_budgets, point_sums, prepare, spike_index

rng = np.random.default_rng(0)
N, T, K = 8, 1300, 5
X = 20 + rng.normal(0, 1, (N, T)).clip(-3, 3)
for row in X:
    for k in rng.choice(np.arange(800, T - 10), 12, replace=False):
        row[k:k + 2] += 30
REQ = rng.integers(1, 9, N).astype(float)
LEVELS = (0.9, 1.1, 1.4, 1.8, 2.5)


def lagged(scale, lag):
    """A forecast that repeats the trace `lag` steps late, times `scale`."""
    Y = np.zeros_like(X)
    Y[:, lag:] = scale * X[:, :-lag]
    return prepare(Y)


FORECASTS = {name: [lagged(s, 1 + i % 3) for s in LEVELS] for i, name in enumerate(POLICIES)}
PARTS = build_parts(X, REQ, FORECASTS)
ONES = np.ones((N, 1))


def toy(cover, slack):
    """Sums for hand-built cases: cover and slack are (P, K, N); one delay, one spike and unit usage per trace."""
    cover, slack = np.asarray(cover, float), np.asarray(slack, float)
    n = cover.shape[2]
    return dict(slack=slack, cover=cover[:, :, None], thr=np.zeros(cover.shape)[:, :, None],
                usage=np.ones(n), spikes=np.ones(n))


def test_read_at_is_the_published_interpolation():
    slack = np.array([0.3, 0.5, 0.9, 1.4, 2.0])
    vals = np.array([0.2, 0.4, 0.55, 0.7, 0.9])
    published = at_budgets(slack[:, None], vals[:, None, None])[:, 0, 0]
    mine = np.array([read_at(slack, vals, b) for b in BUDGETS])
    assert np.allclose(mine, published, equal_nan=True)
    assert np.isnan(read_at(slack, vals, 0.2)) and np.isnan(read_at(slack, vals, 2.5))


def test_sums_match_the_published_scoring():
    tid, pos = spike_index(X)
    Y = FORECASTS['fwdmax'][2]
    slack, cover, thr = range_sums(X, Y, REQ, tid, pos, [(START, T)])[0]
    pub_slack, pub_cover = point_sums(X, Y, tid, pos, np.ones((len(tid), 1, 1), bool))
    assert np.allclose(slack, pub_slack) and np.allclose(cover, pub_cover[:, 0, 0])
    for i in range(N):
        e = evaluate_trace(X[i], Y[i], REQ[i])
        for j, d in enumerate(DELAYS):
            assert cover[j, i] == e[f'cover@{d}']
            assert np.isclose(thr[j, i], e[f'throttle_core_h@{d}'])


def test_the_halves_add_up_to_the_whole():
    for key in ('slack', 'cover', 'thr', 'usage', 'spikes'):
        assert np.allclose(PARTS['all'][key], PARTS['first'][key] + PARTS['second'][key]), key
    assert PARTS['first']['spikes'].sum() > 0 and PARTS['second']['spikes'].sum() > 0


def test_the_first_half_ignores_later_steps():
    cut = split_step(T)
    changed = X.copy()
    changed[:, cut:] = 90.0
    forecasts = {name: [np.where(np.arange(T) < cut, Y, 0.0) for Y in ys] for name, ys in FORECASTS.items()}
    other = build_parts(changed, REQ, forecasts)
    for key in ('slack', 'cover', 'thr', 'usage', 'spikes'):
        assert np.allclose(PARTS['first'][key], other['first'][key]), key


def test_a_fixed_policy_is_a_mix_with_one_policy():
    p = 2
    mix = mix_curve(PARTS['all'], np.full(N, p), ONES)
    part = PARTS['all']
    slack_r = part['slack'][p].sum(axis=1, keepdims=True) / part['usage'].sum()
    cover_r = part['cover'][p].sum(axis=2, keepdims=True) / part['spikes'].sum()
    assert np.allclose(mix[:, :len(DELAYS)], at_budgets(slack_r, cover_r), equal_nan=True)
    for j in range(len(DELAYS)):
        for bi, b in enumerate(BUDGETS):
            assert np.isclose(mix[bi, j, 0], fixed_values(part, j, b)[p], equal_nan=True)


def test_hindsight_finds_a_known_split():
    # policy 0 covers traces 0-4, policy 1 covers traces 5-9; both spend the same slack
    up = np.array([0.5, 1.0, 1.5])
    slack = np.tile(up[None, :, None], (2, 1, 10))
    cover = np.zeros((2, 3, 10))
    cover[0, 1:, :5] = 1
    cover[1, 1:, 5:] = 1
    part = toy(cover, slack)
    assert np.isclose(fixed_values(part, 0, 1.0), 0.5).all()
    a = hindsight(part, 0, 1.0, start=0)
    assert a.tolist() == [0] * 5 + [1] * 5
    assert np.isclose(mix_curve(part, a, np.ones((10, 1)))[BUDGETS.index(1.0), 0, 0], 1.0)


def test_hindsight_moves_slack_to_where_it_covers():
    # policy 1 spends nothing; traces 5-9 have no spike to cover, so they should take it
    slack = np.zeros((2, 3, 10))
    slack[0] = np.array([0.5, 1.0, 1.5])[:, None]
    cover = np.zeros((2, 3, 10))
    cover[0, 2, :5] = 1                                              # covered only at the top quantile
    part = toy(cover, slack)
    part['spikes'] = np.r_[np.ones(5), np.zeros(5)]
    fixed = best_fixed(part, 0, 0.7)
    a = hindsight(part, 0, 0.7, fixed)
    assert fixed == 0 and a.tolist() == [0] * 5 + [1] * 5
    ones = np.ones((10, 1))
    bi = BUDGETS.index(0.7)
    assert mix_curve(part, a, ones)[bi, 0, 0] > mix_curve(part, np.zeros(10, int), ones)[bi, 0, 0]


def test_hindsight_never_scores_below_the_fixed_policy():
    for j in range(len(DELAYS)):
        for b in (0.6, 0.8, 1.0):
            fixed = best_fixed(PARTS['all'], j, b)
            a = hindsight(PARTS['all'], j, b, fixed)
            bi = BUDGETS.index(b)
            assert mix_curve(PARTS['all'], a, ONES)[bi, j, 0] >= fixed_values(PARTS['all'], j, b)[fixed] - 1e-12


def test_ties_stay_with_the_fixed_policy():
    same = np.tile(np.array([0.5, 1.0, 1.5])[None, :, None], (3, 1, 6))
    part = toy(same / 2, same)
    assert hindsight(part, 0, 1.0, start=1).tolist() == [1] * 6


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
