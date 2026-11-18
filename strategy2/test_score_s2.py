"""Checks for score_s2.py, on synthetic traces and hand-written intervals.

usage: python test_score_s2.py
"""
import tempfile

import numpy as np
import pandas as pd

from score_s2 import (ALTERED_OUTCOMES, BUDGETS, DELAYS, LOW_QS, MARGIN, SINGLE_OUTCOMES, UNREAD, at_budgets, at_slack,
                      base_at_argmax, check_published, context, decide_altered, decide_single, forecast, guard,
                      predictions, read_model)
import metrics as M                                                     # noqa: E402  (score_s2 sets the path)
from eval_seeds import QS                                               # noqa: E402
from train import LAG, NCLS                                             # noqa: E402

rng = np.random.default_rng(0)
T = 1100
X = 10 + 3 * rng.random((8, T))
for row in X:
    row[rng.choice(np.arange(800, T), 12, replace=False)] = 70.0
REQ = np.full(8, 4.0)
CTX = context(X, REQ, np.ones((len(np.concatenate([M.find_spikes(x) for x in X])), 1), bool), n_boot=20)


def fake_probs(shift=0):
    """A forecast centred on the true class, moved up by `shift` classes."""
    cls = np.clip((X[:, LAG:] / 2.5).astype(int) + shift, 0, NCLS - 1)
    p = np.full((*cls.shape, NCLS), 0.002)
    np.put_along_axis(p, cls[..., None], 0.5, axis=-1)
    for k in (1, 2, 3):
        np.put_along_axis(p, np.clip(cls + k, 0, NCLS - 1)[..., None], 0.1, axis=-1)
    return (p / p.sum(-1, keepdims=True)).astype(np.float16)


BASE = read_model(fake_probs(), CTX)


def test_reading_at_a_slack_equals_reading_at_a_budget():
    R = BASE['slack'].shape[1]
    vals = BASE['curve']
    want = at_budgets(BASE['slack'], vals)
    for j, b in enumerate(BUDGETS):
        got = at_slack(BASE['slack'], vals, np.full(R, b))
        assert np.array_equal(np.isnan(got), np.isnan(want[j]))
        assert np.allclose(got, want[j], equal_nan=True), b


def test_reading_outside_the_curve_is_nan():
    R = BASE['slack'].shape[1]
    assert np.isnan(at_slack(BASE['slack'], BASE['curve'], np.full(R, 1e6))).all()
    assert np.isnan(at_slack(BASE['slack'], BASE['curve'], np.full(R, 1e-6))).all()


def test_the_scores_equal_the_published_metrics():
    q = 0.9
    k = QS.index(q)
    m = M.evaluate(X, forecast(fake_probs(), q), REQ)
    assert np.isclose(BASE['slack'][k, 0], m['slack_ratio'])
    for j, d in enumerate(DELAYS):
        assert np.isclose(BASE['curve'][k, j, 0], m[f'cover@{d}'])


def test_the_most_likely_class_reading_equals_the_published_metrics():
    m = M.evaluate(X, forecast(fake_probs()), REQ)
    assert np.isclose(BASE['argmax_slack'][0], m['slack_ratio'])
    for j, d in enumerate(DELAYS):
        assert np.isclose(BASE['argmax_cover'][j, 0, 0], m[f'cover@{d}'])
        assert np.isclose(BASE['argmax_thr'][j, 0], m[f'throttle_core_h@{d}'])


def test_a_model_identical_to_cbase_differs_by_nothing():
    same = read_model(fake_probs(), CTX)
    assert np.allclose(same['cover'], BASE['cover'], equal_nan=True)
    assert np.allclose(same['thr'], BASE['thr'], equal_nan=True)


def test_cbase_read_at_its_own_most_likely_slack_stays_on_its_curve():
    got = base_at_argmax(BASE, BASE)
    assert got.shape == BASE['argmax_cover'].shape
    ok = ~np.isnan(got[0, 0])
    assert ok.any()
    assert (got[:, 0, ok] >= 0).all() and (got[:, 0, ok] <= 1).all()


def test_lower_quantiles_leave_the_curve_above_them_unchanged():
    wide = read_model(fake_probs(), CTX, LOW_QS + QS)
    read = ~np.isnan(BASE['cover'])
    assert read.any() and np.allclose(wide['cover'][read], BASE['cover'][read])
    assert wide['slack'][0, 0] <= BASE['slack'][0, 0]


def test_a_forecast_moved_up_costs_more_slack():
    assert read_model(fake_probs(2), CTX)['argmax_slack'][0] > BASE['argmax_slack'][0]


def test_the_single_variants_rows_are_read_from_the_top():
    assert decide_single(-0.005, 0.005) == SINGLE_OUTCOMES['match']
    assert decide_single(0.002, 0.008) == SINGLE_OUTCOMES['match']       # inside the margin, and above 0
    assert decide_single(0.002, 0.02) == SINGLE_OUTCOMES['differ']
    assert decide_single(-0.02, -0.002) == SINGLE_OUTCOMES['differ']
    assert decide_single(-0.02, 0.005) == SINGLE_OUTCOMES['open']
    assert decide_single(-MARGIN, 0.005) == SINGLE_OUTCOMES['open']      # the margin itself is outside
    assert decide_single(np.nan, np.nan) == UNREAD


def test_the_altered_variants_rows():
    assert decide_altered(0.001, 0.02) == ALTERED_OUTCOMES['adds']
    assert decide_altered(-0.02, 0.001) == ALTERED_OUTCOMES['none']
    assert decide_altered(0.0, 0.02) == ALTERED_OUTCOMES['none']
    assert decide_altered(-0.02, -0.001) == ALTERED_OUTCOMES['worse']
    assert decide_altered(np.nan, 0.1) == UNREAD


def test_the_predictions():
    none = ALTERED_OUTCOMES['none']
    out = {'single': SINGLE_OUTCOMES['match'], 'binary': none, 'proportional': none, 'exponential': none}
    trails = dict.fromkeys(out, True)
    assert all(predictions(out, trails).values())
    wrong = predictions({**out, 'single': SINGLE_OUTCOMES['differ'], 'binary': ALTERED_OUTCOMES['adds']}, {**trails, 'single': False})
    assert not any(wrong.values())
    open_ = predictions({**out, 'single': SINGLE_OUTCOMES['open'], 'binary': UNREAD}, trails)
    assert open_['single is no different from cbase'] is None
    assert open_['no altered variant beats cbase at equal slack'] is None


def test_the_published_check_catches_a_changed_number():
    curves = {('cbase', 0): BASE}
    bi = BUDGETS.index(0.8)
    row = {'mode': 'cbase', 'seed': 0, 'budget': 0.8}
    for j, d in enumerate(DELAYS):
        row[f'cover@{d}'], row[f'throttle_core_h@{d}'] = BASE['cover'][bi, j, 0, 0], BASE['thr'][bi, j, 0]
    f1 = {('cbase', 0): {'spikes_f1': 0.4}}
    replication = pd.DataFrame({'paper_spikes_f1': [0.4]}, index=['cbase argmax'])
    check_published(curves, (0,), pd.DataFrame([row]), f1, replication)
    changed_cover = pd.DataFrame([{**row, 'cover@1': row['cover@1'] + 0.01}])
    changed_f1 = {('cbase', 0): {'spikes_f1': 0.5}}
    for table, scores in ((changed_cover, f1), (pd.DataFrame([row]), changed_f1)):
        try:
            check_published(curves, (0,), table, scores, replication)
        except AssertionError:
            continue
        raise RuntimeError('the check let a changed number through')


def test_the_test_run_refuses_an_existing_folder():
    with tempfile.TemporaryDirectory() as d:
        try:
            guard(d, 'HEAD')
        except SystemExit as e:
            assert 'already exists' in str(e)
            return
    raise RuntimeError('the guard let the run start')


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
