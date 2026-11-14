"""Checks for trained_selector.py on synthetic traces and hand-built sums.

usage: python test_trained_selector.py
"""
import numpy as np

from trained_selector import DECIDE, FEATURES, FOLDS, cross_fit, decide, features, fold_ids, subset
from oracle_headroom import BUDGETS, mix_curve
from metrics import GAMMA, find_spikes

rng = np.random.default_rng(1)
N, T = 6, 900
X = np.abs(rng.normal(20, 2, (N, 8, T)))
X[0, 0, 400:402] += 40                                               # one early spike on trace 0
X[1, 0, 300:302] += 40                                               # two close ones on trace 1
X[1, 0, 305:307] += 40
REQ = np.arange(1, N + 1, dtype=float)
LIM = np.linspace(1.0, 1.5, N)
F = features(X, REQ, LIM)


def toy(n, cover_on):
    """Two policies with the same slack; policy p covers trace i's one spike where cover_on[p, i]."""
    slack = np.tile(np.array([0.5, 1.0, 1.5])[None, :, None], (2, 1, n))
    cover = np.zeros((2, 3, n))
    cover[:, 1:] = np.asarray(cover_on, float)[:, None, :]
    return dict(slack=slack, cover=cover[:, :, None], thr=np.zeros((2, 3, 1, n)), usage=np.ones(n), spikes=np.ones(n))


def test_features_have_the_declared_shape():
    assert F.shape == (N, len(FEATURES)) and np.isfinite(F).all()


def test_features_ignore_everything_from_the_decision_step():
    changed = X.copy()
    changed[:, :, DECIDE:] = 99.0
    assert np.array_equal(F, features(changed, REQ, LIM))


def test_features_match_hand_values():
    col = {name: i for i, name in enumerate(FEATURES)}
    cpu = X[2, 0, :DECIDE]
    assert np.isclose(F[2, col['cpu_mean']], cpu.mean()) and np.isclose(F[2, col['cpu_std']], cpu.std())
    assert np.isclose(F[2, col['mem_mean']], X[2, 2, :DECIDE].mean())
    assert np.isclose(F[2, col['net_mean']], (X[2, 5, :DECIDE] + X[2, 6, :DECIDE]).mean())
    assert F[3, col['cpu_request']] == REQ[3] and F[3, col['limit_ratio']] == LIM[3]
    for i in range(N):
        assert F[i, col['early_spikes']] == len(find_spikes(X[i, 0, :DECIDE].astype(float), start=GAMMA))
    assert F[0, col['early_spikes']] >= 1 and F[0, col['close_share']] == 0
    assert F[1, col['early_spikes']] >= 2 and F[1, col['close_share']] > 0


def test_a_flat_trace_has_no_spread_and_no_spikes():
    flat = np.full((1, 8, T), 10.0)
    f = features(flat, [1.0], [1.0])[0]
    col = {name: i for i, name in enumerate(FEATURES)}
    assert f[col['cpu_std']] == 0 and f[col['acf_1']] == 0 and f[col['acf_day']] == 0 and f[col['early_spikes']] == 0


def test_every_trace_is_in_one_fold():
    fold = fold_ids(103)
    assert sorted(np.unique(fold)) == list(range(FOLDS))
    assert np.bincount(fold).max() - np.bincount(fold).min() <= 1
    assert np.array_equal(fold, fold_ids(103))


def test_subset_keeps_the_chosen_traces():
    part = toy(10, rng.integers(0, 2, (2, 10)))
    keep = np.arange(10) % 2 == 0
    sub = subset(part, keep)
    assert sub['slack'].shape == (2, 3, 5) and sub['cover'].shape == (2, 3, 1, 5)
    assert np.array_equal(sub['cover'], part['cover'][..., keep])


def test_the_selector_learns_a_rule_the_features_carry():
    n = 200
    f = rng.normal(0, 1, (n, 2))
    right = f[:, 0] >= 0                                             # policy 1 covers these, policy 0 the rest
    part = toy(n, np.stack([~right, right]))
    a, fixed = cross_fit(part, f, fold_ids(n), 0, 1.0)
    assert (a == right).mean() > 0.95
    ones, bi = np.ones((n, 1)), BUDGETS.index(1.0)
    assert mix_curve(part, a, ones)[bi, 0, 0] > 0.95 > 0.6 > mix_curve(part, fixed, ones)[bi, 0, 0]


def test_no_trace_is_scored_by_a_tree_that_saw_it():
    n = 100
    f = rng.normal(0, 1, (n, 2))
    fold = fold_ids(n)
    part = toy(n, np.stack([f[:, 0] < 0, f[:, 0] >= 0]))
    a, fixed = cross_fit(part, f, fold, 0, 1.0)
    other = {k: v.copy() for k, v in part.items()}
    other['cover'][..., fold == 0] = 1 - other['cover'][..., fold == 0]      # flip fold 0's own outcomes
    a2, fixed2 = cross_fit(other, f, fold, 0, 1.0)
    assert np.array_equal(a[fold == 0], a2[fold == 0]) and np.array_equal(fixed[fold == 0], fixed2[fold == 0])


def test_identical_policies_leave_every_trace_on_the_fixed_one():
    n = 60
    part = toy(n, np.ones((2, n)))
    a, fixed = cross_fit(part, rng.normal(0, 1, (n, 2)), fold_ids(n), 0, 1.0)
    assert np.array_equal(a, fixed)


def test_the_rule_is_strict():
    assert decide(0.0001).startswith('continue') and decide(0.0).startswith('stop') and decide(-0.01).startswith('stop')


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
