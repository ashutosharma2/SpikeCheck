"""The computed hold against the learned hold in a simulated autoscaler (docs/adr/012-autoscaler-simulation.md).

The published replay sets the allocation at step t to the forecast issued d steps earlier. The
simulated autoscaler adds two things a controller does on top of that:
  window     scaling down waits: the allocation follows the highest recommendation of the last
             `window` steps, so increases pass at once and decreases are delayed
  tolerance  a change is skipped while the recommendation is within `tol` of the allocation

Two settings are scored. 'replay' has no window and no tolerance, and is the published replay.
'controller' has a 1-step (5-minute) window and a 10% tolerance. Every method is read along its
quantile curve at the same overall slack budgets as score_hold.py, with slack measured on the
simulated allocation at each delay. The rule in ADR-012 reads throttled core-hours, computed hold
minus learned hold, in the controller setting at slack 0.8.

Before any controller setting is scored, the replay setting has to reproduce the published
numbers under their own budget convention (slack of the undelayed forecast): the ADR-010 dry
run on validation, results/computed_hold/by_method.csv on test.

usage:
  python autoscaler.py val <data.npz> <probs_dir> <out_dir>
      dry run on the validation traces (seed 0)
  python autoscaler.py test <data.npz> <models_seeds_dir> <out_dir> <frozen_commit> <computed_hold_dir>
      the single test run: it won't start unless out_dir is new and the code equals the pushed
      frozen commit
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from eval_seeds import BUDGETS, QS                                   # noqa: E402
from metrics import ALPHA, START                                     # noqa: E402
from score_hold import (COMPUTED, DRY_RUN, DRY_RUN_THROTTLE, GROUPS, LEARNED, NO_HOLD, SHOW, computed_hold,   # noqa: E402
                        encoded, group_members, load_probs, seed_row, summarise)
from score_types import DELAYS, N_BOOT, at_budgets, prepare, spike_index   # noqa: E402
from train import LAG, decode                                        # noqa: E402

SETTINGS = {'replay': (0, 0.0), 'controller': (1, 0.10)}   # name: (scale-down window in steps, tolerance)
JUDGED = 'controller'
MAIN_DELAYS = (1, 3)               # 5 and 15 minutes, the delays the rule reads
PRIMARY = 0.8                      # the slack budget the rule reads
METHODS = {NO_HOLD: (NO_HOLD, None), COMPUTED: (NO_HOLD, 'encoded'), LEARNED: (LEARNED, None),
           'fwdmax6': ('fwdmax6', None), 'fwdmax6 + encoded hold': ('fwdmax6', 'encoded')}
PAIRS = [(COMPUTED, LEARNED), (COMPUTED, NO_HOLD), (LEARNED, NO_HOLD),
         ('fwdmax6 + encoded hold', LEARNED), ('fwdmax6 + encoded hold', 'fwdmax6')]
SHOWN = ('all', 'follow-on')       # spike groups reported, from score_hold.GROUPS
MEASURES = ('cover: all', 'cover: follow-on', 'throttled core-hours', 'allocated core-hours',
            'allocation changes', 'queued core-hours')
THROTTLE = MEASURES.index('throttled core-hours')
FROZEN = ['scripts', 'simulator', 'docs/adr/012-autoscaler-simulation.md', 'requirements.txt']

OUTCOMES = {'below': 'the throttling gain carries over at this delay',
            'spans': 'not shown to carry over at this delay',
            'above': 'the result reverses at this delay'}


def controller(Y, d, window, tol):
    """Allocation in force at every step. Y: (N, T) forecasts, Y[:, t] being the forecast for step t.
    The recommendation in force at t is the forecast issued d steps earlier, Y[:, t - d + 1]."""
    N, T = Y.shape
    rec = np.zeros_like(Y)
    rec[:, d - 1:] = Y[:, :T - d + 1]
    held = rec.copy()
    for k in range(1, window + 1):
        held[:, k:] = np.maximum(held[:, k:], rec[:, :-k])
    if tol == 0:
        return held
    alloc = np.empty_like(held)
    alloc[:, 0] = held[:, 0]
    for t in range(1, T):
        prev = alloc[:, t - 1]
        alloc[:, t] = np.where(np.abs(held[:, t] - prev) > tol * prev, held[:, t], prev)
    return alloc


def queued(X, A):
    """CPU not served at a step joins the next step's demand. Returns the queue summed over the
    scored steps, (N,), in the units of X. An assumption about workloads the traces can't confirm."""
    b = np.zeros(len(X))
    total = np.zeros(len(X))
    for t in range(START, X.shape[1]):
        b = np.maximum(X[:, t] + b - A[:, t], 0)
        total += b
    return total


def alloc_sums(X, A, req, tid, pos, member):
    """Per-trace sums for one allocation: slack (N,) and the MEASURES (len(MEASURES), N), with
    coverage as counts of covered spikes. Steps before START never count."""
    N = len(X)
    core_h = req / 100 * 5 / 60
    cov = A[tid, pos] >= (1 - ALPHA) * X[tid, pos]
    counts = [np.bincount(tid, weights=(cov & member[:, g]).astype(float), minlength=N) for g in range(member.shape[1])]
    xs, a = X[:, START:], A[:, START:]
    return (np.maximum(a - xs, 0).sum(axis=1),
            np.stack(counts + [np.maximum(xs - a, 0).sum(axis=1) * core_h, a.sum(axis=1) * core_h,
                               (a != A[:, START - 1:-1]).sum(axis=1).astype(float), queued(X, A) * core_h]))


def guard(out_dir, frozen, repo):
    """The single test run: a new output folder, and code equal to a frozen commit that was pushed."""
    if os.path.exists(out_dir):
        sys.exit(f'{out_dir} already exists: the test run happens once')

    def git(*args):
        return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True)

    if git('status', '--porcelain', '--', *FROZEN).stdout.strip():
        sys.exit('uncommitted changes to the frozen files')
    if git('diff', '--quiet', frozen, '--', *FROZEN).returncode:
        sys.exit(f'the frozen files differ from {frozen}')
    if not git('branch', '-r', '--contains', frozen).stdout.strip():
        sys.exit(f'{frozen} has not been pushed')
    return git('rev-parse', frozen).stdout.strip()


def decide(ci_lo, ci_hi):
    """ADR-012's outcomes from the interval of computed minus learned throttling at each main delay."""
    per_delay = {d: OUTCOMES['below' if ci_hi[d] < 0 else 'above' if ci_lo[d] > 0 else 'spans'] for d in ci_lo}
    if all(ci_hi[d] < 0 for d in ci_lo):
        return per_delay, "the replay's throttling gain carries over to the simulated autoscaler"
    return per_delay, 'the conclusion is reported per delay'


def check_replay(split, published, seeds, by_method=None):
    """The replay setting under the published budget convention must match what is already published."""
    worst = 0.0
    if split == 'val':
        diff = published[COMPUTED, 0] - published[LEARNED, 0]            # (D, B, 3, R)
        bi = BUDGETS.index(PRIMARY)
        for (d, group), (mean, _, _) in DRY_RUN.items():
            got = diff[DELAYS.index(d), bi, SHOWN.index(group), 0]
            assert round(float(got), 4) == mean, (d, group, got, mean)
        for d, (mean, _, _) in DRY_RUN_THROTTLE.items():
            got = diff[DELAYS.index(d), bi, 2, 0]
            assert abs(got - mean) <= 0.01, (d, got, mean)
        print('the replay setting reproduces the computed-hold dry run', flush=True)
        return
    names = {'cover: all': 0, 'cover: follow-on': 1, 'throttle core-hours: all': 2}
    rows = by_method[by_method.method.isin(METHODS) & by_method.measure.isin(names) & by_method.seed.isin(seeds)]
    assert len(rows), 'no published rows to check against'
    for _, r in rows.iterrows():
        got = published[r.method, int(r.seed)][DELAYS.index(int(r.d)), BUDGETS.index(r.budget), names[r.measure], 0]
        worst = max(worst, abs(got - r.value) / max(abs(r.value), 1e-9))
    assert worst < 1e-5, f'the replay setting differs from by_method.csv by {worst}'
    print(f'the replay setting matches the published table ({len(rows)} rows, {worst:.1e})', flush=True)


def main(split, data_path, probs_dir, out_dir, frozen=None, computed_hold_dir=None):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = guard(out_dir, frozen, repo) if split == 'test' else None
    seeds = (0, 1, 2) if split == 'test' else (0,)

    z = np.load(data_path, allow_pickle=True)
    ids = np.where(z['split'] == split)[0]
    X = z['X'][ids, 0].astype(np.float64)
    req = z['cpu_request'][ids] / 100.0
    N = len(X)

    tid, pos = spike_index(X)
    member = group_members(X, tid)[:, [GROUPS.index(g) for g in SHOWN]]
    n_group = np.stack([np.bincount(tid, weights=member[:, g], minlength=N) for g in range(len(SHOWN))])
    usage = X[:, START:].sum(axis=1)
    hold = encoded(computed_hold(X))

    rng = np.random.default_rng(0)                                   # trace weights as score_types.py
    W = np.ones((N, N_BOOT + 1))
    W[:, 1:] = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(N_BOOT)], axis=1)
    G, R = len(SHOWN), N_BOOT + 1

    def read(slack, sums):
        """Per-trace sums at every quantile -> the measures at every budget, under every trace weighting."""
        slack_r = (slack @ W) / (usage @ W)                           # (K, R)
        vals = sums @ W                                               # (K, F, R)
        with np.errstate(invalid='ignore', divide='ignore'):
            vals[:, :G] /= (n_group @ W)
        return at_budgets(slack_r, vals)

    def score(name, mode, held, seed):
        probs = load_probs(split, probs_dir, mode, seed)
        assert probs.shape[0] == N
        pad = np.full((N, LAG), np.nan)
        slack0, points = [], {(s, d): [] for s in SETTINGS for d in DELAYS}
        for q in QS:
            Y = prepare(np.concatenate([pad, decode(probs, q)], axis=1))
            if held:
                Y = prepare(np.maximum(Y, hold))
            slack0.append(np.maximum(Y[:, START:] - X[:, START:], 0).sum(axis=1))
            for (s, d), rows in points.items():
                rows.append(alloc_sums(X, controller(Y, d, *SETTINGS[s]), req, tid, pos, member))
        slack0 = np.stack(slack0)
        sim = np.stack([[read(np.stack([p[0] for p in points[s, d]]), np.stack([p[1] for p in points[s, d]]))
                         for d in DELAYS] for s in SETTINGS])       # (settings, D, B, F, R)
        pub = np.stack([read(slack0, np.stack([p[1][:G + 1] for p in points['replay', d]])) for d in DELAYS])
        print(name, 'seed', seed, 'scored', flush=True)
        return sim, pub

    scored = {(name, s): score(name, *spec, s) for name, spec in METHODS.items() for s in seeds}
    check_replay(split, {k: v[1] for k, v in scored.items()}, seeds,
                 pd.read_csv(f'{computed_hold_dir}/by_method.csv') if split == 'test' else None)
    curves = {k: {'sim': v[0]} for k, v in scored.items()}

    by_method, paired = [], []
    for (name, s), c in curves.items():
        for si, setting in enumerate(SETTINGS):
            for j, d in enumerate(DELAYS):
                for b in SHOW:
                    by_method += [dict(method=name, seed=s, setting=setting, d=d, budget=b, measure=m,
                                       value=c['sim'][si, j, BUDGETS.index(b), mi, 0]) for mi, m in enumerate(MEASURES)]
    summaries = {}
    for a, b_ in PAIRS:
        summary = summaries[a, b_] = summarise(curves, a, b_, 'sim', seeds)
        for si, setting in enumerate(SETTINGS):
            for j, d in enumerate(DELAYS):
                for b in SHOW:
                    bi = BUDGETS.index(b)
                    for mi, m in enumerate(MEASURES):
                        base = np.mean([curves[b_, s]['sim'][si, j, bi, mi, 0] for s in seeds])
                        row = seed_row(summary, (si, j, bi, mi), seeds)
                        paired.append(dict(pair=f'{a} - {b_}', setting=setting, d=d, budget=b, measure=m,
                                           relative=row['mean'] / base if base else np.nan, **row))

    si, bi = list(SETTINGS).index(JUDGED), BUDGETS.index(PRIMARY)
    rows = {d: seed_row(summaries[COMPUTED, LEARNED], (si, DELAYS.index(d), bi, THROTTLE), seeds) for d in MAIN_DELAYS}
    per_delay, overall = decide({d: r['ci_lo'] for d, r in rows.items()}, {d: r['ci_hi'] for d, r in rows.items()})
    checks = [dict(check='throttled core-hours, computed minus learned', setting=JUDGED, minutes=5 * d, rule='ci_hi < 0',
                   passed=bool(r['ci_hi'] < 0), **r) for d, r in rows.items()]

    os.makedirs(out_dir, exist_ok=split == 'val')
    for name, table in (('checks', checks), ('paired', paired), ('by_method', by_method)):
        pd.DataFrame(table).to_csv(f'{out_dir}/{name}.csv', index=False, float_format='%.6g')
    with open(f'{out_dir}/conclusion.json', 'w') as f:
        json.dump(dict(split=split, frozen_commit=commit, seeds=list(seeds), traces=N, n_boot=N_BOOT, budget=PRIMARY,
                       settings={k: dict(window_steps=w, tolerance=t) for k, (w, t) in SETTINGS.items()}, judged=JUDGED,
                       per_delay={f'{5 * d} min': v for d, v in per_delay.items()}, conclusion=overall), f, indent=2)

    print(pd.DataFrame(checks)[['check', 'minutes', 'mean', 'seed_min', 'seed_max', 'ci_lo', 'ci_hi', 'passed']]
          .to_string(index=False, float_format='%.4f'))
    print('\n'.join(f'{5 * d} min: {v}' for d, v in per_delay.items()))
    print('conclusion:', overall)
    print('wrote', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:])
