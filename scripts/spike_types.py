"""Spike types (docs/adr/009-spike-types.md): label every real spike on the test traces.

Labels, checked in this order (the first rule that fits sets the label):
  unknown    part of the +-tol window around DAY steps back falls before the reference
             detection start, so there isn't enough history to check seasonal
  seasonal   a reference spike starts within +-tol steps of DAY steps earlier
  follow-on  the previous reference spike started at most `follow` steps earlier
  isolated   none of the above
Every spike also gets a `close` flag (previous spike at most `follow` steps earlier),
whatever its label, so the overlap between seasonal and follow-on can be reported.

The labelled spikes are always the published ones (find_spikes from LAG + GAMMA). The
reference spikes used for looking back are the same set, except in the 'ref250' setting,
where both look-backs use a second find_spikes run from step GAMMA (same thresholds).

usage: python spike_types.py <data.npz> <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from metrics import GAMMA, LAG, MIN_RISE, SIGMA, _weighted_rolling_stats, find_spikes

DAY = 288                       # steps in 24 hours of 5-minute bins
DETECT_START = LAG + GAMMA      # where the published spikes start (step 762)
LABELS = ('unknown', 'seasonal', 'follow-on', 'isolated')
SETTINGS = {                    # baseline, then one change at a time (ADR-009)
    'base': dict(tol=1, follow=6, ref_start=DETECT_START),
    'tol2': dict(tol=2, follow=6, ref_start=DETECT_START),
    'tol3': dict(tol=3, follow=6, ref_start=DETECT_START),
    'follow8': dict(tol=1, follow=8, ref_start=DETECT_START),
    'follow12': dict(tol=1, follow=12, ref_start=DETECT_START),
    'ref250': dict(tol=1, follow=6, ref_start=GAMMA),
}
OFFSETS = sorted(set(range(240, 337, 12)) | {250, 270, 330})   # chance-match check around DAY


def label_spikes(spikes, ref, ref_start, tol=1, follow=6):
    """Label spike positions using reference spikes `ref`, detected from step ref_start.
    Returns (labels, close), one entry per spike."""
    ref = np.asarray(ref, dtype=np.int64)
    labels = np.empty(len(spikes), dtype=object)
    close = np.zeros(len(spikes), dtype=bool)
    for i, k in enumerate(spikes):
        earlier = ref[ref < k]
        close[i] = len(earlier) > 0 and k - earlier[-1] <= follow
        if k - DAY - tol < ref_start:
            labels[i] = 'unknown'
        elif np.any(np.abs(earlier - (k - DAY)) <= tol):
            labels[i] = 'seasonal'
        elif close[i]:
            labels[i] = 'follow-on'
        else:
            labels[i] = 'isolated'
    return labels, close


def previous_spike(spikes, ref):
    """Position of the reference spike before each spike, or -1 if there is none."""
    ref = np.asarray(ref, dtype=np.int64)
    i = np.searchsorted(ref, spikes) - 1
    if len(ref) == 0:
        return np.full(len(spikes), -1, dtype=np.int64)
    return np.where(i >= 0, ref[np.clip(i, 0, None)], -1)


def previous_peak(x, spikes, prev):
    """Highest CPU from the previous spike's start up to (not including) each spike; NaN without one."""
    return np.array([x[p:k].max() if p >= 0 else np.nan for k, p in zip(spikes, prev)])


def label_trace(x, tol=1, follow=6, ref_start=DETECT_START):
    """Detect the published spikes on one trace and label them.
    Returns (spikes, labels, close, prev), prev being the previous reference spike or -1."""
    spikes = find_spikes(x)
    ref = spikes if ref_start == DETECT_START else find_spikes(x, start=ref_start)
    labels, close = label_spikes(spikes, ref, ref_start, tol, follow)
    return spikes, labels, close, previous_spike(spikes, ref)


def above_threshold(x):
    """The detector's candidate condition at every step (before the refractory period)."""
    mean, sd = _weighted_rolling_stats(x, GAMMA)
    rise = x - mean
    return (rise > SIGMA * sd) & (rise > MIN_RISE)


def near(spikes, k, d, tol):
    return bool(np.any(np.abs(spikes - (k - d)) <= tol))


def main(data_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(data_path, allow_pickle=True)
    te = np.where(z['split'] == 'test')[0]
    Xte = z['X'][te, 0].astype(np.float64)

    labels = {s: [] for s in SETTINGS}
    close = []
    gaps = []
    plateau = ref250_diff = 0
    eligible = 0                                  # spikes with history for every offset
    offset_hits = {d: 0 for d in OFFSETS}
    tol_hits = {t: 0 for t in (1, 2, 3)}

    for x in Xte:
        spikes = find_spikes(x)
        ref250 = find_spikes(x, start=GAMMA)
        ref250_diff += len(set(ref250[ref250 >= DETECT_START].tolist()) ^ set(spikes.tolist()))
        for s, kw in SETTINGS.items():
            ref = ref250 if kw['ref_start'] == GAMMA else spikes
            labels[s].append(label_spikes(spikes, ref, kw['ref_start'], kw['tol'], kw['follow'])[0])
        close.append(label_spikes(spikes, spikes, DETECT_START)[1])

        g = np.diff(spikes)
        gaps.append(g)
        above = above_threshold(x)
        plateau += sum(bool(above[a:b + 1].all()) for a, b in zip(spikes[:-1], spikes[1:])
                       if b - a <= SETTINGS['base']['follow'])

        for k in spikes[spikes - max(OFFSETS) - 1 >= DETECT_START]:
            eligible += 1
            for d in OFFSETS:
                offset_hits[d] += near(spikes, k, d, 1)
            for t in tol_hits:
                tol_hits[t] += near(spikes, k, DAY, t)

    labels = {s: np.concatenate(v) for s, v in labels.items()}
    close = np.concatenate(close)
    gaps = np.concatenate(gaps)
    base = labels['base']
    n = len(base)

    counts = pd.DataFrame([dict(setting=s, label=l, n=int((v == l).sum()), share=(v == l).mean())
                           for s, v in labels.items() for l in LABELS])
    counts.to_csv(f'{out_dir}/counts.csv', index=False)
    changes = pd.DataFrame([dict(setting=s, base_label=a, label=b, n=int(((base == a) & (v == b)).sum()))
                            for s, v in labels.items() if s != 'base' for a in LABELS for b in LABELS])
    changes[changes.n > 0].to_csv(f'{out_dir}/label_changes.csv', index=False)
    g_val, g_n = np.unique(gaps, return_counts=True)
    pd.DataFrame(dict(gap=g_val, n=g_n)).to_csv(f'{out_dir}/gaps.csv', index=False)
    pd.DataFrame([dict(offset=d, matches=offset_hits[d], n=eligible, share=offset_hits[d] / eligible)
                  for d in OFFSETS]).to_csv(f'{out_dir}/offsets.csv', index=False)

    summary = dict(
        traces=len(Xte), spikes=n, gaps=len(gaps),
        gap_min=int(gaps.min()), gap_median=float(np.median(gaps)), gap_mean=float(gaps.mean()),
        gap_share_le6=float((gaps <= 6).mean()), gap_share_le12=float((gaps <= 12).mean()),
        first_known_step=DETECT_START + DAY + SETTINGS['base']['tol'],
        close=int(close.sum()),
        close_seasonal=int((close & (base == 'seasonal')).sum()),
        close_unknown=int((close & (base == 'unknown')).sum()),
        close_follow_on=int((close & (base == 'follow-on')).sum()),
        plateau_pairs=int(plateau),
        offset_eligible=eligible,
        day_match_share_by_tol={f'+-{t}': h / eligible for t, h in tol_hits.items()},
        ref250_spikes_differing_after_detect_start=int(ref250_diff),
    )
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(counts[counts.setting == 'base'].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:3])
