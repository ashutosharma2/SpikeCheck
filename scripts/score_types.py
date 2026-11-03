"""Spike coverage per type (docs/adr/009-spike-types.md) at matched over-provisioning.

Over-provisioning is spent over the whole trace, so it is matched overall: every forecaster
is read at the same overall slack budgets as eval_seeds.py, and its coverage is then split by
spike type. Learned runs (<mode>_s<seed>/test_probs.npz) are read along their quantile curve.
Baselines (rolling max, seasonal max) are read along a frontier chosen per delay on the
validation traces by overall coverage, then frozen, so no setting is picked with test scores.

Paired differences come with the per-seed range and a 95% paired bootstrap over whole test
traces (the same resampled traces for both sides; baseline frontiers stay frozen). Unknown
spikes are scored but left out of the paired comparisons.

Two sub-groups split follow-on spikes by whether the previous peak alone (highest CPU from the
previous spike's start up to the spike) would cover them, i.e. is at least (1 - ALPHA) of the
spike. They test whether the hold explains proposed's lead on follow-on spikes.

usage: python score_types.py <data.npz> <models_seeds_dir> <out_dir> [<horizon30_dir> <replication_dir>]
The optional folders are checked: summed over types, coverage must match the published results.
"""
import glob
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines as B
from eval_seeds import BUDGETS, COLORS, QS
from metrics import ALPHA, START, find_spikes
from spike_types import SETTINGS, label_trace, previous_peak
from train import LAG, decode

DELAYS = (1, 3, 6)
GROUPS = ('all', 'seasonal', 'follow-on', 'isolated', 'unknown', 'seasonal, close', 'seasonal, not close',
          'follow-on, previous peak covers', 'follow-on, previous peak short')
PAIRED_GROUPS = [g for g in GROUPS if g != 'unknown']
FAMILIES = {                     # the same settings as run_eval.py and eval_seeds.py
    'rolling-max': [(f'w={w} m={m}', lambda X, w=w, m=m: B.rolling_max(X, w, m))
                    for w in (3, 6, 12, 24, 48, 96, 288) for m in (0.0, 0.1, 0.25)],
    'seasonal-max': [(f'w={w} m={m}', lambda X, w=w, m=m: B.seasonal_max(X, w, margin=m))
                     for w in (6, 12, 24) for m in (0.0, 0.1)],
}
LEARNED_PAIRS = [('fwdmax', 'proposed'), ('cbase', 'proposed'), ('fwdmax6', 'proposed'), ('fwdmax6', 'fwdmax')]
N_BOOT = 500
COLORS = {**COLORS, 'rolling-max': '#9D755D'}


def prepare(Y):
    """The same clean-up eval_seeds.score applies before scoring."""
    Y = np.where(np.isnan(Y), 0.0, Y)
    Y[:, :LAG] = 0.0
    return Y


def spike_index(X):
    """Published spikes of every trace, flattened: (trace id, position)."""
    sp = [find_spikes(x) for x in X]
    return np.repeat(np.arange(len(X)), [len(s) for s in sp]), np.concatenate(sp)


def covered(X, Y, tid, pos):
    """(n_spikes, len(DELAYS)) bool: allocation in effect at the spike >= (1 - ALPHA) x, as metrics.cover@d."""
    return np.stack([Y[tid, pos - d + 1] >= (1 - ALPHA) * X[tid, pos] for d in DELAYS], axis=1)


def point_sums(X, Y, tid, pos, member):
    """Per-trace sums for one forecaster: slack (N,) and covered counts (D, S, G, N)."""
    slack = np.maximum(Y[:, START:] - X[:, START:], 0).sum(axis=1)
    cov = covered(X, Y, tid, pos)                                     # (n, D)
    S, G = member.shape[1:]
    out = np.zeros((len(DELAYS), S, G, len(X)))
    for j in range(len(DELAYS)):
        for s in range(S):
            for g in range(G):
                out[j, s, g] = np.bincount(tid, weights=(cov[:, j] & member[:, s, g]).astype(float),
                                           minlength=len(X))
    return slack, out


def at_budgets(slack, vals):
    """slack: (K, R) pooled slack ratio of K curve points under R trace weightings; vals: (K, F, R).
    Log-linear interpolation at every budget, as eval_seeds.at_budget; NaN outside the curve.
    Returns (len(BUDGETS), F, R)."""
    K, F, R = vals.shape
    out = np.full((len(BUDGETS), F, R), np.nan)
    if K < 2:
        return out
    xs = np.log(slack)
    order = np.argsort(xs, axis=0)
    xs = np.take_along_axis(xs, order, axis=0)
    vals = np.take_along_axis(vals, order[:, None, :], axis=0)
    r = np.arange(R)
    for j, b in enumerate(np.log(BUDGETS)):
        i = np.clip((xs <= b).sum(axis=0) - 1, 0, K - 2)
        lam = (b - xs[i, r]) / (xs[i + 1, r] - xs[i, r])
        v = (1 - lam)[:, None] * vals[i, :, r] + lam[:, None] * vals[i + 1, :, r]   # (R, F)
        ok = (xs[0] <= b) & (b <= xs[-1])
        out[j][:, ok] = v.T[:, ok]
    return out


def frontier(X, tid, pos, settings):
    """Per delay, the settings that give the best overall coverage at each level of slack
    (validation traces only). Returns {d: [setting names]} and a table of the validation scores."""
    rows = []
    for name, f in settings:
        Y = prepare(f(X))
        slack = np.maximum(Y[:, START:] - X[:, START:], 0).sum() / X[:, START:].sum()
        cov = covered(X, Y, tid, pos).mean(axis=0)
        rows.append(dict(setting=name, val_slack=slack, **{f'val_cover@{d}': c for d, c in zip(DELAYS, cov)}))
    t = pd.DataFrame(rows).sort_values('val_slack')
    chosen = {}
    for d in DELAYS:
        best, keep = -np.inf, []
        for _, row in t.iterrows():
            if row[f'val_cover@{d}'] > best:
                best = row[f'val_cover@{d}']
                keep.append(row.setting)
        chosen[d] = keep
    return chosen, t


def main(data_path, models_dir, out_dir, horizon30_dir=None, replication_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(data_path, allow_pickle=True)
    Xte = z['X'][np.where(z['split'] == 'test')[0], 0].astype(np.float64)
    Xva = z['X'][np.where(z['split'] == 'val')[0], 0].astype(np.float64)
    N = len(Xte)

    # labels for every setting; member[n, s, g] says spike n is in group g under setting s
    tid, pos = spike_index(Xte)
    names = list(SETTINGS)
    member = np.zeros((len(tid), len(names), len(GROUPS)), dtype=bool)
    lab = {s: [] for s in names}
    cls = {s: [] for s in names}
    peak_ok = {s: [] for s in names}
    for x in Xte:
        for s in names:
            sp, l, c, prev = label_trace(x, **SETTINGS[s])
            lab[s].append(l)
            cls[s].append(c)
            with np.errstate(invalid='ignore'):
                peak_ok[s].append(previous_peak(x, sp, prev) >= (1 - ALPHA) * x[sp])
    for si, s in enumerate(names):
        l, c, ok = np.concatenate(lab[s]), np.concatenate(cls[s]), np.concatenate(peak_ok[s])
        assert len(l) == len(tid)
        sea, fo = l == 'seasonal', l == 'follow-on'
        for gi, g in enumerate(GROUPS):
            member[:, si, gi] = {'all': np.ones(len(l), bool), 'seasonal, close': sea & c,
                                 'seasonal, not close': sea & ~c,
                                 'follow-on, previous peak covers': fo & ok,
                                 'follow-on, previous peak short': fo & ~ok}.get(g, l == g)
    n_group = np.stack([[np.bincount(tid, weights=member[:, s, g], minlength=N)
                         for g in range(len(GROUPS))] for s in range(len(names))])   # (S, G, N)
    usage = Xte[:, START:].sum(axis=1)

    # every forecaster point: learned runs x readouts, then every baseline setting
    points, slack, cov = [], [], []
    for path in sorted(glob.glob(os.path.join(models_dir, '*_s*', 'test_probs.npz'))):
        mode, seed = re.match(r'(.+)_s(\d+)$', os.path.basename(os.path.dirname(path))).groups()
        probs = np.load(path)['probs']
        pad = np.full((probs.shape[0], LAG), np.nan)
        for q in QS:
            sl, cv = point_sums(Xte, prepare(np.concatenate([pad, decode(probs, q)], axis=1)), tid, pos, member)
            points.append(dict(kind='learned', mode=mode, seed=int(seed), readout=f'q={q}'))
            slack.append(sl)
            cov.append(cv)
        print(mode, seed, 'scored', flush=True)
    vtid, vpos = spike_index(Xva)
    frontiers, val_rows = {}, []
    for fam, settings in FAMILIES.items():
        frontiers[fam], t = frontier(Xva, vtid, vpos, settings)
        for d in DELAYS:
            val_rows += [dict(family=fam, d=d, setting=r.setting, val_slack=r.val_slack,
                              val_cover=r[f'val_cover@{d}'], on_frontier=r.setting in frontiers[fam][d])
                         for _, r in t.iterrows()]
        for name, f in settings:
            sl, cv = point_sums(Xte, prepare(f(Xte)), tid, pos, member)
            points.append(dict(kind='baseline', mode=fam, seed=-1, readout=name))
            slack.append(sl)
            cov.append(cv)
        print(fam, 'scored', flush=True)
    pd.DataFrame(val_rows).to_csv(f'{out_dir}/baseline_frontiers.csv', index=False, float_format='%.6g')
    pts = pd.DataFrame(points)
    slack, cov = np.stack(slack), np.stack(cov)                     # (P, N), (P, D, S, G, N)

    # trace weights: column 0 is the full test set, the rest are bootstrap resamples
    rng = np.random.default_rng(0)
    W = np.ones((N, N_BOOT + 1))
    W[:, 1:] = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(N_BOOT)], axis=1)
    slack_r = (slack @ W) / (usage @ W)                             # (P, R)
    with np.errstate(invalid='ignore', divide='ignore'):
        cover_r = (cov @ W) / (n_group @ W)[None, None]             # (P, D, S, G, R)

    if horizon30_dir or replication_dir:
        check(pts, slack_r[:, 0], cover_r[..., 0], horizon30_dir, replication_dir)

    # each curve read at every budget: (budget, D, S, G, R)
    D, S, G, R = cover_r.shape[1:]
    curves = {}
    for (mode, seed), g in pts[pts.kind == 'learned'].groupby(['mode', 'seed']):
        idx = g.index.values
        curves[mode, seed] = at_budgets(slack_r[idx], cover_r[idx].reshape(len(idx), -1, R)).reshape(-1, D, S, G, R)
    for fam in FAMILIES:
        out = np.full((len(BUDGETS), D, S, G, R), np.nan)
        for j, d in enumerate(DELAYS):
            idx = pts.index[(pts['mode'] == fam) & pts.readout.isin(frontiers[fam][d])].values
            out[:, j] = at_budgets(slack_r[idx], cover_r[idx, j].reshape(len(idx), -1, R)).reshape(-1, S, G, R)
        curves[fam, -1] = out
    if horizon30_dir:
        check_interpolation(curves, horizon30_dir)

    rows = []
    for (mode, seed), c in curves.items():
        for bi, b in enumerate(BUDGETS):
            for j, d in enumerate(DELAYS):
                for si, s in enumerate(names):
                    for gi, gname in enumerate(GROUPS):
                        rows.append(dict(setting=s, mode=mode, seed=seed, d=d, budget=b, group=gname,
                                         cover=c[bi, j, si, gi, 0], n_spikes=int(n_group[si, gi].sum())))
    by_type = pd.DataFrame(rows)
    by_type.to_csv(f'{out_dir}/coverage_by_type.csv', index=False, float_format='%.6g')

    pairs = LEARNED_PAIRS + [(m, fam) for m in ('cbase', 'proposed', 'fwdmax', 'fwdmax6') for fam in FAMILIES]
    rows = []
    for a, b in pairs:
        seeds = sorted({s for (m, s) in curves if m == a})
        diff = np.stack([curves[a, s] - curves[b, s if (b, s) in curves else -1] for s in seeds])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(diff, axis=0)                         # (budget, D, S, G, R)
            lo, hi = np.nanpercentile(mean[..., 1:], [2.5, 97.5], axis=-1)
        for bi, bud in enumerate(BUDGETS):
            for j, d in enumerate(DELAYS):
                for si, s in enumerate(names):
                    for gi, gname in enumerate(GROUPS):
                        if gname not in PAIRED_GROUPS:
                            continue
                        per_seed = diff[:, bi, j, si, gi, 0]
                        ok = ~np.isnan(per_seed)
                        rows.append(dict(setting=s, pair=f'{a} - {b}', d=d, budget=bud, group=gname,
                                         mean=mean[bi, j, si, gi, 0],
                                         seed_min=per_seed[ok].min() if ok.any() else np.nan,
                                         seed_max=per_seed[ok].max() if ok.any() else np.nan,
                                         n_seeds=int(ok.sum()), ci_lo=lo[bi, j, si, gi], ci_hi=hi[bi, j, si, gi],
                                         n_spikes=int(n_group[si, gi].sum())))
    paired = pd.DataFrame(rows).dropna(subset=['mean'])
    paired.to_csv(f'{out_dir}/paired_by_type.csv', index=False, float_format='%.6g')

    plot(by_type, out_dir)
    print('wrote', out_dir)


def check(pts, slack0, cover0, horizon30_dir, replication_dir):
    """Summed over types, coverage and slack must equal the published results."""
    worst = 0.0
    if horizon30_dir:
        ref = pd.read_csv(f'{horizon30_dir}/all_runs.csv')
        for i, p in pts[pts.kind == 'learned'].iterrows():
            r = ref[(ref['mode'] == p['mode']) & (ref.seed == p.seed) & (ref.readout == p.readout)].iloc[0]
            worst = max(worst, abs(slack0[i] - r.slack_ratio),
                        *[abs(cover0[i, j, 0, 0] - r[f'cover@{d}']) for j, d in enumerate(DELAYS)])
        for i, p in pts[pts['mode'] == 'seasonal-max'].iterrows():
            r = ref[(ref['mode'] == 'seasonal-max') & (ref.readout == p.readout)].iloc[0]
            worst = max(worst, abs(slack0[i] - r.slack_ratio),
                        *[abs(cover0[i, j, 0, 0] - r[f'cover@{d}']) for j, d in enumerate(DELAYS)])
    if replication_dir:
        ref = pd.read_csv(f'{replication_dir}/results.csv').set_index('name')
        for i, p in pts[pts['mode'] == 'rolling-max'].iterrows():
            r = ref.loc[f'rollmax {p.readout}']
            worst = max(worst, abs(slack0[i] - r.slack_ratio),
                        *[abs(cover0[i, j, 0, 0] - r[f'cover@{d}']) for j, d in enumerate(DELAYS)])
    assert worst < 1e-9, f'per-type totals differ from the published results by {worst}'
    print(f'totals match the published results (largest difference {worst:.1e})', flush=True)


def check_interpolation(curves, horizon30_dir):
    """Read at each budget, the learned runs' overall coverage must equal matched_slack_by_seed.csv."""
    ref = pd.read_csv(f'{horizon30_dir}/matched_slack_by_seed.csv')
    worst = 0.0
    for _, r in ref[ref.seed >= 0].iterrows():
        for j, d in enumerate(DELAYS):
            a, e = curves[r['mode'], r.seed][BUDGETS.index(r.budget), j, 0, 0, 0], r[f'cover@{d}']
            if np.isnan(a) and np.isnan(e):
                continue
            worst = max(worst, abs(a - e)) if not (np.isnan(a) or np.isnan(e)) else np.inf
    assert worst < 1e-9, f'matched-slack coverage differs from horizon30 by {worst}'
    print(f'matched-slack coverage matches horizon30 (largest difference {worst:.1e})', flush=True)


def plot(by_type, out_dir):
    base = by_type[by_type.setting == 'base']
    groups = ['all', 'seasonal', 'follow-on', 'isolated']
    for d in DELAYS:
        fig, axes = plt.subplots(1, len(groups), figsize=(16, 4.2), sharey=True)
        for ax, gname in zip(axes, groups):
            g = base[(base.d == d) & (base.group == gname)]
            for mode, m in g.groupby('mode'):
                s = m.groupby('budget').cover.agg(['mean', 'min', 'max'])
                c = COLORS.get(mode, '#333')
                ax.plot(s.index, s['mean'], '-o', ms=3, color=c, label=mode)
                ax.fill_between(s.index, s['min'], s['max'], color=c, alpha=0.15)
            ax.set_xscale('log')
            ax.set_title(f'{gname} (n={int(g.n_spikes.iloc[0]):,})')
            ax.set_xlabel('over-provisioning: slack / usage (log)')
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(f'spikes covered, cover@{d}')
        axes[0].legend(fontsize=8)
        fig.suptitle('coverage by spike type at matched over-provisioning (line = seed mean, band = seed range)')
        fig.tight_layout()
        fig.savefig(f'{out_dir}/cover_by_type_at_{d}.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:6])
