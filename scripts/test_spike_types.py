"""Checks for the ADR-009 labelling rules, on hand-written spike positions and one synthetic trace.

usage: python test_spike_types.py
"""
import numpy as np

from metrics import GAMMA
from spike_types import DAY, DETECT_START, label_spikes, label_trace, previous_peak, previous_spike

K = 1400                        # a spike with a full day of history


def label_last(spikes, ref=None, ref_start=DETECT_START, **kw):
    spikes = np.asarray(spikes)
    ref = spikes if ref is None else np.asarray(ref)
    labels, close = label_spikes(spikes, ref, ref_start, **kw)
    return labels[-1], bool(close[-1])


def test_seasonal_beats_follow_on():
    assert label_last([K - DAY, K - 5, K]) == ('seasonal', True)


def test_seasonal_tolerance_edges():
    assert label_last([K - DAY - 1, K])[0] == 'seasonal'
    assert label_last([K - DAY + 1, K])[0] == 'seasonal'
    assert label_last([K - DAY - 2, K])[0] == 'isolated'
    assert label_last([K - DAY - 2, K], tol=2)[0] == 'seasonal'


def test_follow_on_window():
    assert label_last([K - 5, K]) == ('follow-on', True)
    assert label_last([K - 6, K]) == ('follow-on', True)
    assert label_last([K - 7, K]) == ('isolated', False)
    assert label_last([K - 12, K]) == ('isolated', False)
    assert label_last([K - 12, K], follow=12) == ('follow-on', True)


def test_only_the_previous_spike_counts_for_follow_on():
    assert label_last([K - 10, K - 7, K])[0] == 'isolated'


def test_unknown_without_a_full_day():
    first_known = DETECT_START + DAY + 1                       # 1051
    # one step short: a daily match at 762 and a spike 5 steps back still give unknown
    assert label_last([DETECT_START, first_known - 6, first_known - 1]) == ('unknown', True)
    assert label_last([DETECT_START + 1, first_known])[0] == 'seasonal'
    assert label_last([first_known])[0] == 'isolated'
    assert label_last([first_known], tol=2)[0] == 'unknown'


def test_seasonal_looks_past_other_spikes():
    assert label_last([K - DAY, K - 100, K - 50, K])[0] == 'seasonal'


def test_ref250_looks_back_before_detection_start():
    # daily predecessor at 512 is only in the step-250 run
    assert label_last([800])[0] == 'unknown'
    assert label_last([800], ref=[800 - DAY, 800], ref_start=GAMMA)[0] == 'seasonal'
    # previous spike at 760, before the published spikes start
    assert label_last([765])[0] == 'unknown'
    assert label_last([765], ref=[760, 765], ref_start=GAMMA) == ('follow-on', True)


def test_previous_spike_and_peak():
    assert previous_spike([5, 10, 20], [3, 10, 15]).tolist() == [3, 3, 15]
    assert previous_spike([2], [3, 10]).tolist() == [-1]
    x = np.array([0, 9, 7, 0, 0, 0, 10, 0], dtype=float)
    peak = previous_peak(x, [6, 1], [1, -1])
    assert peak[0] == 9 and np.isnan(peak[1])           # max of x[1:6], not including the spike


def test_synthetic_trace_end_to_end():
    rng = np.random.default_rng(0)
    x = 20 + 0.1 * rng.standard_normal(2304)
    bumps = [900, 1100, 1105, 1100 + DAY, 1105 + DAY]
    x[bumps] += 20
    spikes, labels, close, prev = label_trace(x)
    assert spikes.tolist() == bumps, spikes
    assert labels.tolist() == ['unknown', 'isolated', 'follow-on', 'seasonal', 'seasonal']
    assert close.tolist() == [False, False, True, False, True]
    assert prev.tolist() == [-1, 900, 1100, 1105, 1100 + DAY]


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for f in tests:
        f()
        print('ok', f.__name__)
    print(f'all {len(tests)} tests passed')
