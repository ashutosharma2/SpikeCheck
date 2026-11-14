"""How much could choosing a policy per trace gain? A hindsight bound on the validation traces.

Every trace gets one policy from a fixed library. All policies are read at the same quantile, the
mix is scored along its quantile curve and read at the same overall slack budgets as a single
policy (score_types.py), so a fixed policy is the mix that gives every trace the same policy.

Two numbers per delay and budget, each against the best fixed policy:
  ceiling    the best assignment found with hindsight on the whole scored period. No selector
             can know this; it estimates the most a per-trace selector could gain.
  follow     the assignment that hindsight picks on the first half of the scored period, scored
             on the second half against the fixed policy that was best on the first half. This
             is the part of the ceiling that carries over within a trace.

The hindsight assignment is a search, not a proof: a penalised pick per trace (covered spikes
minus a price on slack) at every quantile and price, then one-trace swaps until none helps. It
is a real assignment, so its score is achievable with hindsight; the true ceiling can be higher.

Validation traces and seed 0 only, so the test traces stay untouched. Intervals resample whole
traces with the assignment frozen: they cover trace sampling, not the choice of assignment.

usage: python oracle_headroom.py <data.npz> <val_probs_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from eval_seeds import BUDGETS, QS                                   # noqa: E402
from metrics import GAMMA, START                                     # noqa: E402
from score_hold import DRY_RUN, computed_hold, encoded, load_probs   # noqa: E402
from score_types import DELAYS, N_BOOT, at_budgets, covered, prepare, spike_index   # noqa: E402
from train import LAG, decode                                        # noqa: E402

SEED = 0
SHOW = (0.6, 0.8, 1.0)             # the budgets reported, as score_hold.py
POLICIES = {                       # name: (trained target, hold added to its forecast)
    'cbase': ('cbase', None),
    'proposed': ('proposed', None),
    'fwdmax': ('fwdmax', None),
    'fwdmax6': ('fwdmax6', None),
    'fwdmax + encoded hold': ('fwdmax', 'encoded'),
    'fwdmax6 + encoded hold': ('fwdmax6', 'encoded'),
}
PRICES = np.concatenate([[0.0], np.logspace(-2, 1, 31)])   # coverage given up per unit of slack ratio
MAX_PASSES = 20


def split_step(T):
    """Middle of the steps where spikes are detected (LAG + GAMMA to T)."""
    return (LAG + GAMMA + T) // 2


def range_sums(X, Y, req, tid, pos, ranges):
    """Per-trace sums of one forecast over each [lo, hi) step range: slack (N,), covered spikes
    (D, N) and throttled core-hours (D, N), as metrics.evaluate_trace. Steps before START never count."""
    N, T = X.shape
    slack = np.maximum(Y - X, 0)
    cov = covered(X, Y, tid, pos)                                     # (n, D)
    thr = []
    for d in DELAYS:
        alloc = np.zeros_like(Y)
        alloc[:, d - 1:] = Y[:, :T - d + 1]
        thr.append(np.maximum(X - alloc, 0) * (req[:, None] / 100 * 5 / 60))
    out = []
    for lo, hi in ranges:
        lo = max(lo, START)
        inside = (pos >= lo) & (pos < hi)
        out.append((slack[:, lo:hi].sum(axis=1),
                    np.stack([np.bincount(tid, weights=(cov[:, j] & inside).astype(float), minlength=N)
                              for j in range(len(DELAYS))]),
                    np.stack([t[:, lo:hi].sum(axis=1) for t in thr])))
    return out


def read_at(slack_ratio, vals, budget):
    """One curve read at one budget: eval_seeds.at_budget's log-linear interpolation; NaN outside."""
    with np.errstate(divide='ignore'):
        xs = np.log(slack_ratio)
    order = np.argsort(xs)
    xs, vals, b = xs[order], vals[order], np.log(budget)
    if not (xs[0] <= b <= xs[-1]):
        return np.nan
    i = int(np.clip((xs <= b).sum() - 1, 0, len(xs) - 2))
    lam = (b - xs[i]) / (xs[i + 1] - xs[i])
    return (1 - lam) * vals[i] + lam * vals[i + 1]


def mix_curve(part, assign, W):
    """The mix that gives trace i policy assign[i], read at every budget under every trace weighting.
    Returns (len(BUDGETS), 2 * len(DELAYS), R): coverage per delay, then throttled core-hours per delay."""
    n = np.arange(len(assign))
    slack = part['slack'][assign, :, n].T                              # (K, N)
    cover = np.moveaxis(part['cover'][assign, :, :, n], 0, -1)         # (K, D, N)
    thr = np.moveaxis(part['thr'][assign, :, :, n], 0, -1)
    slack_r = (slack @ W) / (part['usage'] @ W)
    with np.errstate(invalid='ignore', divide='ignore'):
        cover_r = (cover @ W) / (part['spikes'] @ W)
    return at_budgets(slack_r, np.concatenate([cover_r, thr @ W], axis=1))


def fixed_values(part, j, budget):
    """Coverage at delay index j and one budget for every policy used on all traces."""
    usage, spikes = part['usage'].sum(), part['spikes'].sum()
    return np.array([read_at(part['slack'][p].sum(axis=1) / usage, part['cover'][p, :, j].sum(axis=1) / spikes, budget)
                     for p in range(part['slack'].shape[0])])


def best_fixed(part, j, budget):
    v = fixed_values(part, j, budget)
    assert not np.all(np.isnan(v)), 'no policy reaches this budget'
    return int(np.nanargmax(v))


def hindsight(part, j, budget, start):
    """Search for the assignment with the highest coverage at delay index j and one budget, from the
    sums in `part` only. `start` is the fixed policy it has to match or beat."""
    slack, cover = part['slack'], part['cover'][:, :, j]              # (P, K, N)
    P, K, N = slack.shape
    usage, spikes = part['usage'].sum(), part['spikes'].sum()
    n = np.arange(N)

    def value(a):
        return read_at(slack[a, :, n].sum(axis=0) / usage, cover[a, :, n].sum(axis=0) / spikes, budget)

    best = np.full(N, start)
    best_v = value(best)
    for k in range(K):
        for price in PRICES:
            gain = cover[:, k] / spikes - price * slack[:, k] / usage  # (P, N)
            gain[start] += 1e-12                                       # ties stay with the fixed policy
            a = gain.argmax(axis=0)
            v = value(a)
            if v > best_v:
                best, best_v = a, v

    s_sum, c_sum = slack[best, :, n].sum(axis=0), cover[best, :, n].sum(axis=0)
    for _ in range(MAX_PASSES):
        changed = False
        for i in range(N):
            for p in range(P):
                if p == best[i]:
                    continue
                s2 = s_sum - slack[best[i], :, i] + slack[p, :, i]
                c2 = c_sum - cover[best[i], :, i] + cover[p, :, i]
                v = read_at(s2 / usage, c2 / spikes, budget)
                if v > best_v + 1e-12:
                    best[i], best_v, s_sum, c_sum, changed = p, v, s2, c2, True
        if not changed:
            break
    return best


def interval(diff):
    """Value on the full set and the 95% interval over the bootstrap columns."""
    lo, hi = np.nanpercentile(diff[1:], [2.5, 97.5])
    return dict(gain=diff[0], ci_lo=lo, ci_hi=hi)


def build_parts(X, req, forecasts):
    """forecasts: {policy name: [Y at every quantile]}. Returns the per-trace sums for the whole
    scored period, its first half and its second half."""
    N, T = X.shape
    tid, pos = spike_index(X)
    cut = split_step(T)
    ranges = {'all': (START, T), 'first': (START, cut), 'second': (cut, T)}
    parts = {r: dict(slack=[], cover=[], thr=[]) for r in ranges}
    for name in POLICIES:
        rows = [range_sums(X, Y, req, tid, pos, list(ranges.values())) for Y in forecasts[name]]
        for ri, r in enumerate(ranges):
            for ki, key in enumerate(('slack', 'cover', 'thr')):
                parts[r][key].append(np.stack([row[ri][ki] for row in rows]))
    for r, (lo, hi) in ranges.items():
        parts[r] = {k: np.stack(v) for k, v in parts[r].items()}      # slack (P, K, N); cover, thr (P, K, D, N)
        parts[r]['usage'] = X[:, lo:hi].sum(axis=1)
        parts[r]['spikes'] = np.bincount(tid[(pos >= lo) & (pos < hi)], minlength=N).astype(float)
    parts['split_step'] = cut
    return parts


def headroom(parts, W):
    """The two comparisons at every delay and reported budget. Returns the table rows and the picks."""
    names = list(POLICIES)
    N, D = len(parts['all']['usage']), len(DELAYS)
    rows, picks = [], []
    for j, d in enumerate(DELAYS):
        for b in SHOW:
            bi = BUDGETS.index(b)
            fixed = best_fixed(parts['all'], j, b)
            a = hindsight(parts['all'], j, b, fixed)
            mix, base = mix_curve(parts['all'], a, W)[bi], mix_curve(parts['all'], np.full(N, fixed), W)[bi]
            rows.append(dict(check='ceiling', d=d, budget=b, fixed_policy=names[fixed], fixed_cover=base[j, 0],
                             mix_cover=mix[j, 0], **interval(mix[j] - base[j]), fixed_throttle=base[D + j, 0],
                             mix_throttle=mix[D + j, 0], traces_moved=float((a != fixed).mean())))
            picks += [dict(check='ceiling', d=d, budget=b, policy=names[p], traces=int((a == p).sum())) for p in range(len(names))]

            fixed1 = best_fixed(parts['first'], j, b)
            a1 = hindsight(parts['first'], j, b, fixed1)
            mix, base = mix_curve(parts['second'], a1, W)[bi], mix_curve(parts['second'], np.full(N, fixed1), W)[bi]
            rows.append(dict(check='follow', d=d, budget=b, fixed_policy=names[fixed1], fixed_cover=base[j, 0],
                             mix_cover=mix[j, 0], **interval(mix[j] - base[j]), fixed_throttle=base[D + j, 0],
                             mix_throttle=mix[D + j, 0], traces_moved=float((a1 != fixed1).mean())))
            picks += [dict(check='follow', d=d, budget=b, policy=names[p], traces=int((a1 == p).sum())) for p in range(len(names))]
            print(f'{5 * d} min, slack {b}: done', flush=True)
    return rows, picks


def check_dry_run(parts):
    """The fixed policies must reproduce score_hold.py's validation dry run (computed minus learned, slack 0.8)."""
    names = list(POLICIES)
    for (d, group), (mean, _, _) in DRY_RUN.items():
        if group != 'all':
            continue
        v = fixed_values(parts['all'], DELAYS.index(d), 0.8)
        got = v[names.index('fwdmax + encoded hold')] - v[names.index('proposed')]
        assert round(float(got), 4) == mean, (d, got, mean)
    print('the fixed policies reproduce the computed-hold dry run', flush=True)


def load_parts(data_path, probs_dir):
    """The validation traces, every policy's forecasts at every quantile, and their per-trace sums.
    Returns (ids of the validation traces in data.npz, parts)."""
    z = np.load(data_path, allow_pickle=True)
    ids = np.where(z['split'] == 'val')[0]
    X = z['X'][ids, 0].astype(np.float64)
    req = z['cpu_request'][ids] / 100.0
    N = len(X)
    hold = encoded(computed_hold(X))

    forecasts = {name: [] for name in POLICIES}
    for mode in sorted({m for m, _ in POLICIES.values()}):
        probs = load_probs('val', probs_dir, mode, SEED)
        assert probs.shape[0] == N
        pad = np.full((N, LAG), np.nan)
        for q in QS:
            Y = prepare(np.concatenate([pad, decode(probs, q)], axis=1))
            for name, (m, h) in POLICIES.items():
                if m == mode:
                    forecasts[name].append(prepare(np.maximum(Y, hold)) if h else Y)
        print(mode, 'decoded', flush=True)
    parts = build_parts(X, req, forecasts)
    check_dry_run(parts)
    return ids, parts


def trace_weights(N):
    """Column 0 is the full set, the rest are bootstrap resamples of whole traces, as score_types.py."""
    rng = np.random.default_rng(0)
    W = np.ones((N, N_BOOT + 1))
    W[:, 1:] = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(N_BOOT)], axis=1)
    return W


def main(data_path, probs_dir, out_dir):
    _, parts = load_parts(data_path, probs_dir)
    N = len(parts['all']['usage'])
    W = trace_weights(N)

    rows, picks = headroom(parts, W)
    by_policy = [dict(d=d, budget=b, policy=name, cover=v)
                 for j, d in enumerate(DELAYS) for b in SHOW
                 for name, v in zip(POLICIES, fixed_values(parts['all'], j, b))]

    os.makedirs(out_dir, exist_ok=True)
    for name, table in (('headroom', rows), ('picks', picks), ('by_policy', by_policy)):
        pd.DataFrame(table).to_csv(f'{out_dir}/{name}.csv', index=False, float_format='%.6g')
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(dict(split='val', seed=SEED, traces=N, n_boot=N_BOOT, policies=list(POLICIES), budgets=list(SHOW),
                       split_step=parts['split_step'], spikes=int(parts['all']['spikes'].sum()),
                       traces_without_spikes=int((parts['all']['spikes'] == 0).sum())), f, indent=2)
    cols = ['check', 'd', 'budget', 'fixed_policy', 'fixed_cover', 'mix_cover', 'gain', 'ci_lo', 'ci_hi', 'traces_moved']
    print(pd.DataFrame(rows)[cols].to_string(index=False, float_format='%.4f'))
    print('wrote', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:4])
