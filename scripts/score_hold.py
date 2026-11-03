"""Computed hold against the learned hold at matched over-provisioning (docs/adr/010-computed-hold.md).

proposed's target splits exactly: max x[t-6..t+4] = max(max x[t-6..t-1], max x[t..t+4]). The first
part is already in the forecast's input and the second is fwdmax's target. The split also holds in
train.py's classes, so a perfect proposed model would output max(encoded hold, fwdmax's forecast)
at every quantile. This script scores that computed hold against the trained proposed model.

Every forecaster is read along its quantile curve at the same overall slack budgets as
score_types.py, with the same paired bootstrap over whole traces. Intervals cover trace sampling
only; the seed columns show training variability. The checks in ADR-010 are read at slack 0.8.

usage:
  python score_hold.py val <data.npz> <probs_dir> <out_dir>
      dry run on the validation traces (seed 0, <probs_dir>/<mode>_s0.npy); must reproduce DRY_RUN
  python score_hold.py test <data.npz> <models_seeds_dir> <out_dir> <frozen_commit> <spike_types_dir> <horizon30_dir>
      the single test run: it won't start unless out_dir is new and the code equals the pushed
      frozen commit, and it checks plain fwdmax and proposed against the published tables before
      any computed hold is scored
"""
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from eval_seeds import BUDGETS, QS
from metrics import ALPHA, GAMMA, START, find_spikes
from score_types import DELAYS, N_BOOT, at_budgets, point_sums, prepare, spike_index
from spike_types import label_trace, previous_peak
from train import BIN_W, LAG, NCLS, decode

HOLD = 6                         # the paper's backward window: x[t-6..t-1]
MAIN_DELAYS = (1, 3)             # 5 and 15 minutes, the delays the four checks use
PRIMARY = 0.8                    # the slack budget the checks use
SHOW = (0.6, 0.8, 1.0)           # the budgets reported
MARGIN = 0.01                    # follow-on non-inferiority margin, absolute coverage
SIZE_CUTS = (10.0, 16.0, 30.0)   # CPU % of request: quartiles of the validation spikes, frozen
GROUPS = ('all', 'seasonal', 'follow-on', 'isolated', 'unknown', 'follow-on, previous peak covers',
          'follow-on, previous peak short', 'size q1', 'size q2', 'size q3', 'size q4')
REGIONS = ('after spike', 'run-up', 'other')

COMPUTED, EXACT, LEARNED, NO_HOLD = 'fwdmax + encoded hold', 'fwdmax + exact hold', 'proposed', 'fwdmax'
PLAIN = {LEARNED: (LEARNED, None), NO_HOLD: (NO_HOLD, None), 'fwdmax6': ('fwdmax6', None)}
HELD = {COMPUTED: (NO_HOLD, 'encoded'), EXACT: (NO_HOLD, 'exact'), 'fwdmax6 + encoded hold': ('fwdmax6', 'encoded')}
PAIRS = [(COMPUTED, LEARNED), (EXACT, LEARNED), (COMPUTED, NO_HOLD), (LEARNED, NO_HOLD),
         ('fwdmax6 + encoded hold', 'fwdmax6'), ('fwdmax6 + encoded hold', LEARNED)]

OUTCOMES = {(True, True): 'both claims hold at this delay',
            (True, False): 'not an acceptable replacement at this delay',
            (False, True): 'not shown to be meaningfully worse on follow-on, not shown to be better overall',
            (False, False): 'neither claim holds at this delay'}

# validation, seed 0, slack 0.8, computed minus learned, from the exploration: (mean, ci_lo, ci_hi)
DRY_RUN = {(1, 'all'): (0.0055, 0.0001, 0.0117), (1, 'follow-on'): (-0.0057, -0.0105, -0.0017),
           (3, 'all'): (0.0070, 0.0011, 0.0130), (3, 'follow-on'): (-0.0026, -0.0096, 0.0048)}
DRY_RUN_THROTTLE = {1: (-34.13, -51.18, -19.14), 3: (-56.88, -82.76, -34.99)}
DRY_RUN_SHARE = {COMPUTED: 0.271, LEARNED: 0.294}


def computed_hold(X, steps=HOLD):
    """Highest CPU over the steps before t, all of it in the forecast's input. (N, T); 0 before steps."""
    hold = np.zeros(X.shape)
    hold[:, steps:] = sliding_window_view(X, steps, axis=1)[:, :X.shape[1] - steps].max(axis=-1)
    return hold


def encoded(v):
    """What a model can output for a target of v: train.py's class, decoded to its upper edge."""
    return (np.clip(np.floor(v / BIN_W), 0, NCLS - 1) + 1) * BIN_W


def region_masks(spikes, shape):
    """After spike (the 6 steps after a spike), run-up (the spike and the 5 steps before) and other,
    from each trace's spike positions. A step in both of the first two counts as after spike."""
    after, runup = np.zeros(shape, bool), np.zeros(shape, bool)
    for i, sp in enumerate(spikes):
        for k in sp:
            runup[i, max(k - 5, 0):k + 1] = True
            after[i, k + 1:k + HOLD + 1] = True
    return after, runup & ~after, ~(after | runup)


def group_members(X, tid):
    """(n_spikes, len(GROUPS)) bool, with ADR-009's baseline labels."""
    lab, ok, level = [], [], []
    for x in X:
        sp, l, _, prev = label_trace(x)
        with np.errstate(invalid='ignore'):
            ok.append(previous_peak(x, sp, prev) >= (1 - ALPHA) * x[sp])
        lab.append(l)
        level.append(x[sp])
    lab, ok, level = np.concatenate(lab), np.concatenate(ok), np.concatenate(level)
    assert len(lab) == len(tid)
    fo, size = lab == 'follow-on', np.digitize(level, SIZE_CUTS, right=True)
    named = {'all': np.ones(len(lab), bool), 'follow-on, previous peak covers': fo & ok,
             'follow-on, previous peak short': fo & ~ok,
             **{f'size q{i + 1}': size == i for i in range(4)}}
    return np.stack([named.get(g, lab == g) for g in GROUPS], axis=1)


def trace_sums(X, Y, req, tid, pos, member, masks):
    """Per-trace sums for one forecaster: slack (1 + regions, N), covered spikes (D, G, N) and
    throttled core-hours (D, 1 + regions, N). Index 0 of a region axis is the whole trace."""
    _, cov = point_sums(X, Y, tid, pos, member[:, None])
    T = X.shape[1]
    slack = np.maximum(Y - X, 0)
    slack[:, :START] = 0
    thr = []
    for d in DELAYS:
        alloc = np.zeros_like(Y)
        alloc[:, d - 1:] = Y[:, :T - d + 1]
        t = np.maximum(X - alloc, 0)
        t[:, :START] = 0
        t *= req[:, None] / 100 * 5 / 60
        thr.append([t.sum(axis=1)] + [(t * m).sum(axis=1) for m in masks])
    return np.stack([slack.sum(axis=1)] + [(slack * m).sum(axis=1) for m in masks]), cov[:, 0], np.array(thr)


def decide(overall_lo, follow_lo):
    """ADR-010's outcomes from the lower bounds at each main delay. Both rules are strict."""
    better = {d: bool(overall_lo[d] > 0) for d in overall_lo}
    close = {d: bool(follow_lo[d] > -MARGIN) for d in follow_lo}
    per_delay = {d: OUTCOMES[better[d], close[d]] for d in overall_lo}
    if all(better.values()) and all(close.values()):
        return per_delay, 'complete hypothesis supported'
    if not any(close.values()):
        return per_delay, 'drop computed hold as a replacement for the learned hold'
    if len(set(per_delay.values())) > 1:
        return per_delay, 'complete hypothesis not supported; the conclusion is delay-specific'
    return per_delay, 'complete hypothesis not supported; ' + next(iter(per_delay.values()))


def guard(out_dir, frozen, repo):
    """The single test run: a new output folder, and code equal to a frozen commit that was pushed."""
    if os.path.exists(out_dir):
        sys.exit(f'{out_dir} already exists: the test run happens once')

    def git(*args):
        return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)

    paths = ['scripts', 'docs/adr/010-computed-hold.md', 'requirements.txt']
    if git('status', '--porcelain', '--', *paths).stdout.strip():
        sys.exit('uncommitted changes to the frozen files')
    if git('diff', '--quiet', frozen, '--', *paths).returncode:
        sys.exit(f'the frozen files differ from {frozen}')
    if not git('branch', '-r', '--contains', frozen).stdout.strip():
        sys.exit(f'{frozen} has not been pushed')
    return git('rev-parse', frozen).stdout.strip()


def load_probs(split, probs_dir, mode, seed):
    if split == 'val':
        return np.load(f'{probs_dir}/{mode}_s{seed}.npy')
    return np.load(f'{probs_dir}/{mode}_s{seed}/test_probs.npz')['probs']


def summarise(curves, a, b, key, seeds):
    """Paired difference a - b: per seed (seeds, ..., R), seed mean (..., R), and its 95% interval."""
    diff = np.stack([curves[a, s][key] - curves[b, s][key] for s in seeds])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(diff, axis=0)
        lo, hi = np.nanpercentile(mean[..., 1:], [2.5, 97.5], axis=-1)
    return diff, mean, lo, hi


def seed_row(summary, idx, seeds):
    """Mean, each seed's value, their range and the trace interval at one index of a summary."""
    diff, mean, lo, hi = summary
    per_seed = diff[(slice(None), *idx, 0)]
    return dict(mean=mean[(*idx, 0)], **{f'seed_{s}': v for s, v in zip(seeds, per_seed)},
                seed_min=per_seed.min(), seed_max=per_seed.max(), ci_lo=lo[idx], ci_hi=hi[idx])


def check_published(curves, seeds, paired, matched):
    """Plain fwdmax and proposed must equal the published tables, the bootstrap included."""
    worst = 0.0
    summary = summarise(curves, NO_HOLD, LEARNED, 'cover', seeds)
    rows = paired[(paired.setting == 'base') & (paired.pair == f'{NO_HOLD} - {LEARNED}') & paired.group.isin(GROUPS)]
    assert len(rows), 'no published rows to check against'
    for _, r in rows.iterrows():
        idx = (BUDGETS.index(r.budget), DELAYS.index(r.d), GROUPS.index(r.group))
        got = seed_row(summary, idx, seeds)
        worst = max(worst, *[abs(got[k] - r[k]) for k in ('mean', 'seed_min', 'seed_max', 'ci_lo', 'ci_hi')])
    assert worst < 1e-5, f'fwdmax minus proposed differs from paired_by_type.csv by {worst}'
    relative = 0.0
    for _, r in matched[matched['mode'].isin(PLAIN) & (matched.seed >= 0)].iterrows():
        c = curves[r['mode'], r.seed]
        for j, d in enumerate(DELAYS):
            a, e = c['thr'][BUDGETS.index(r.budget), j, 0, 0], r[f'throttle_core_h@{d}']
            if not (np.isnan(a) and np.isnan(e)):
                relative = max(relative, abs(a - e) / e) if not (np.isnan(a) or np.isnan(e)) else np.inf
    assert relative < 1e-6, f'throttled core-hours differ from matched_slack_by_seed.csv by {relative}'
    print(f'plain methods match the published tables (coverage {worst:.1e}, throttling {relative:.1e})', flush=True)


def check_dry_run(pair, throttle, share):
    """The validation run must reproduce the exploration: means exactly, intervals within noise."""
    for key, (m, lo, hi) in DRY_RUN.items():
        r = pair[key]
        assert round(float(r['mean']), 4) == m, (key, r['mean'], m)
        assert abs(r['ci_lo'] - lo) <= 0.002 and abs(r['ci_hi'] - hi) <= 0.002, (key, r['ci_lo'], r['ci_hi'])
    for d, (m, lo, hi) in DRY_RUN_THROTTLE.items():
        r = throttle[d]
        assert abs(r['mean'] - m) <= 0.01, (d, r['mean'], m)
        assert abs(r['ci_lo'] - lo) <= 0.05 * abs(lo) and abs(r['ci_hi'] - hi) <= 0.05 * abs(hi), (d, r['ci_lo'], r['ci_hi'])
    for name, s in DRY_RUN_SHARE.items():
        assert abs(share[name] - s) <= 0.002, (name, share[name], s)
    print('the dry run reproduces the exploration', flush=True)


def main(split, data_path, probs_dir, out_dir, frozen=None, spike_types_dir=None, horizon30_dir=None):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = guard(out_dir, frozen, repo) if split == 'test' else None
    seeds = (0, 1, 2) if split == 'test' else (0,)

    z = np.load(data_path, allow_pickle=True)
    ids = np.where(z['split'] == split)[0]
    X = z['X'][ids, 0].astype(np.float64)
    req = z['cpu_request'][ids] / 100.0
    N = len(X)

    tid, pos = spike_index(X)
    member = group_members(X, tid)
    n_group = np.stack([np.bincount(tid, weights=member[:, g], minlength=N) for g in range(len(GROUPS))])
    usage = X[:, START:].sum(axis=1)
    masks = region_masks([find_spikes(x, start=GAMMA) for x in X], X.shape)
    exact = computed_hold(X)
    holds = {'exact': exact, 'encoded': encoded(exact)}

    # trace weights: column 0 is the full set, the rest are bootstrap resamples, as in score_types.py
    rng = np.random.default_rng(0)
    W = np.ones((N, N_BOOT + 1))
    W[:, 1:] = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(N_BOOT)], axis=1)

    def score(name, mode, hold, seed):
        """One forecaster's curve read at every budget, under every trace weighting."""
        probs = load_probs(split, probs_dir, mode, seed)
        assert probs.shape[0] == N
        pad = np.full((N, LAG), np.nan)
        points = []
        for q in QS:
            Y = prepare(np.concatenate([pad, decode(probs, q)], axis=1))
            if hold:
                Y = prepare(np.maximum(Y, holds[hold]))
            points.append(trace_sums(X, Y, req, tid, pos, member, masks))
        slack, cov, thr = [np.stack([p[i] for p in points]) for i in range(3)]      # K first
        K, R = len(QS), N_BOOT + 1
        by_region = slack @ W                                                       # (K, 1 + regions, R)
        slack_r = by_region[:, 0] / (usage @ W)                                     # (K, R)
        with np.errstate(invalid='ignore', divide='ignore'):
            cover_r = (cov @ W) / (n_group @ W)                                     # (K, D, G, R)
        D, M = len(DELAYS), len(REGIONS)
        print(name, 'seed', seed, 'scored', flush=True)
        return dict(cover=at_budgets(slack_r, cover_r.reshape(K, -1, R)).reshape(-1, D, len(GROUPS), R),
                    thr=at_budgets(slack_r, (thr @ W).reshape(K, -1, R)).reshape(-1, D, 1 + M, R),
                    share=at_budgets(slack_r, by_region[:, 1:] / by_region[:, :1]))

    curves = {(name, s): score(name, *spec, s) for name, spec in PLAIN.items() for s in seeds}
    if split == 'test':
        check_published(curves, seeds, pd.read_csv(f'{spike_types_dir}/paired_by_type.csv'),
                        pd.read_csv(f'{horizon30_dir}/matched_slack_by_seed.csv'))
    curves.update({(name, s): score(name, *spec, s) for name, spec in HELD.items() for s in seeds})

    # the tables
    n_spikes = member.sum(axis=0)
    by_method, paired, throttle, share = [], [], [], []
    for (name, s), c in curves.items():
        for b in SHOW:
            bi = BUDGETS.index(b)
            by_method += [dict(method=name, seed=s, d='', budget=b, measure=f'slack share: {r}', value=c['share'][bi, ri, 0], n_spikes='')
                          for ri, r in enumerate(REGIONS)]
            for j, d in enumerate(DELAYS):
                by_method += [dict(method=name, seed=s, d=d, budget=b, measure=f'cover: {g}', value=c['cover'][bi, j, gi, 0], n_spikes=int(n_spikes[gi]))
                              for gi, g in enumerate(GROUPS)]
                by_method += [dict(method=name, seed=s, d=d, budget=b, measure=f'throttle core-hours: {r}', value=c['thr'][bi, j, ri, 0], n_spikes='')
                              for ri, r in enumerate(('all',) + REGIONS)]
    for a, b_ in PAIRS:
        cover_s, thr_s, share_s = [summarise(curves, a, b_, k, seeds) for k in ('cover', 'thr', 'share')]
        for b in SHOW:
            bi = BUDGETS.index(b)
            share += [dict(pair=f'{a} - {b_}', budget=b, region=r, **seed_row(share_s, (bi, ri), seeds)) for ri, r in enumerate(REGIONS)]
            for j, d in enumerate(DELAYS):
                paired += [dict(pair=f'{a} - {b_}', d=d, budget=b, group=g, **seed_row(cover_s, (bi, j, gi), seeds), n_spikes=int(n_spikes[gi]))
                           for gi, g in enumerate(GROUPS) if g != 'unknown']
                for ri, r in enumerate(('all',) + REGIONS):
                    base = np.mean([curves[b_, s]['thr'][bi, j, ri, 0] for s in seeds])
                    row = seed_row(thr_s, (bi, j, ri), seeds)
                    throttle.append(dict(pair=f'{a} - {b_}', d=d, budget=b, region=r, relative=row['mean'] / base, **row))

    # the checks in ADR-010
    bi = BUDGETS.index(PRIMARY)
    cover_s, thr_s, share_s = [summarise(curves, COMPUTED, LEARNED, k, seeds) for k in ('cover', 'thr', 'share')]
    primary = {(d, g): seed_row(cover_s, (bi, DELAYS.index(d), GROUPS.index(g)), seeds)
               for d in MAIN_DELAYS for g in ('all', 'follow-on')}
    per_delay, overall = decide({d: primary[d, 'all']['ci_lo'] for d in MAIN_DELAYS},
                                {d: primary[d, 'follow-on']['ci_lo'] for d in MAIN_DELAYS})
    thr_rows = {d: seed_row(thr_s, (bi, DELAYS.index(d), 0), seeds) for d in MAIN_DELAYS}
    after_row = seed_row(share_s, (bi, 0), seeds)
    checks = []
    for d in MAIN_DELAYS:
        checks.append(dict(check='overall coverage', minutes=5 * d, rule='ci_lo > 0', passed=bool(primary[d, 'all']['ci_lo'] > 0), **primary[d, 'all']))
        checks.append(dict(check='follow-on coverage', minutes=5 * d, rule=f'ci_lo > -{MARGIN}', passed=bool(primary[d, 'follow-on']['ci_lo'] > -MARGIN), **primary[d, 'follow-on']))
    for d in MAIN_DELAYS:
        checks.append(dict(check='throttled core-hours (prediction)', minutes=5 * d, rule='ci_hi < 0', passed=bool(thr_rows[d]['ci_hi'] < 0), **thr_rows[d]))
    checks.append(dict(check='after-spike share of slack (prediction)', minutes='', rule='ci_hi < 0', passed=bool(after_row['ci_hi'] < 0), **after_row))

    if split == 'val':
        check_dry_run(primary, thr_rows, {name: curves[name, 0]['share'][bi, 0, 0] for name in DRY_RUN_SHARE})

    os.makedirs(out_dir, exist_ok=split == 'val')
    for name, rows in (('checks', checks), ('paired', paired), ('throttle', throttle), ('slack_share', share), ('by_method', by_method)):
        pd.DataFrame(rows).to_csv(f'{out_dir}/{name}.csv', index=False, float_format='%.6g')
    with open(f'{out_dir}/conclusion.json', 'w') as f:
        json.dump(dict(split=split, frozen_commit=commit, seeds=list(seeds), traces=N, n_boot=N_BOOT, budget=PRIMARY, margin=MARGIN,
                       n_spikes={g: int(n) for g, n in zip(GROUPS, n_spikes)}, per_delay={f'{5 * d} min': v for d, v in per_delay.items()},
                       conclusion=overall), f, indent=2)

    print(pd.DataFrame(checks)[['check', 'minutes', 'mean', 'seed_min', 'seed_max', 'ci_lo', 'ci_hi', 'rule', 'passed']].to_string(index=False, float_format='%.4f'))
    print('\n'.join(f'{5 * d} min: {v}' for d, v in per_delay.items()))
    print('conclusion:', overall)
    print('wrote', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:])
