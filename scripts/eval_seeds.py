"""Horizon-vs-shaping test across seeds.

For every <mode>_s<seed>/test_probs.npz: decode argmax and quantiles, score with the
corrected metrics, then interpolate each quantile curve at fixed slack budgets so the
modes are compared at equal over-provisioning. Reports mean +- sd over seeds and
seed-paired differences against Strategy 1 ('proposed').

usage: python eval_seeds.py <data.npz> <models_seeds_dir> <out_dir>
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines as B
import metrics as M
from train import decode, LAG

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995]
BUDGETS = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
METRICS = ['recall_paper', 'cover@1', 'cover@3', 'cover@6', 'f1_fwd',
           'throttle_core_h@1', 'throttle_core_h@3', 'throttle_core_h@6']
COLORS = {'cbase': '#F58518', 'proposed': '#B279A2', 'fwdmax': '#4C78A8', 'seasonal-max': '#72B7B2'}


def score(Xte, Y, req):
    Y = np.where(np.isnan(Y), 0.0, Y)
    Y[:, :LAG] = 0.0
    return M.summarize([M.evaluate_trace(Xte[i], Y[i], req[i]) for i in range(len(Xte))])


def at_budget(curve, metric, budget):
    """Interpolate metric at a slack budget along a curve ordered by slack (log-linear)."""
    c = curve.sort_values('slack_ratio')
    xs = np.log(c.slack_ratio.values)
    if not (xs[0] <= np.log(budget) <= xs[-1]):
        return np.nan
    return float(np.interp(np.log(budget), xs, c[metric].values))


def main(data_path, models_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(data_path, allow_pickle=True)
    te = np.where(z['split'] == 'test')[0]
    Xte = z['X'][te, 0].astype(np.float64)
    req = z['cpu_request'][te] / 100.0

    rows = []
    for path in sorted(glob.glob(os.path.join(models_dir, '*_s*', 'test_probs.npz'))):
        mode, seed = re.match(r'(.+)_s(\d+)$', os.path.basename(os.path.dirname(path))).groups()
        probs = np.load(path)['probs']
        pad = np.full((probs.shape[0], LAG), np.nan)
        for q in [None] + QS:
            r = score(Xte, np.concatenate([pad, decode(probs, q)], axis=1), req)
            r.update(mode=mode, seed=int(seed), readout='argmax' if q is None else f'q={q}')
            rows.append(r)
        print(mode, seed, 'done', flush=True)
    for w in (6, 12, 24):
        for m in (0.0, 0.1):
            r = score(Xte, B.seasonal_max(Xte, w, margin=m), req)
            r.update(mode='seasonal-max', seed=-1, readout=f'w={w} m={m}')
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f'{out_dir}/all_runs.csv', index=False)

    # matched-slack table: quantile curve (readouts q=...) per mode/seed, interpolated
    interp = []
    for (mode, seed), g in res.groupby(['mode', 'seed']):
        curve = g if mode == 'seasonal-max' else g[g.readout.str.startswith('q=')]
        for b in BUDGETS:
            interp.append({'mode': mode, 'seed': seed, 'budget': b,
                           **{m: at_budget(curve, m, b) for m in METRICS}})
    interp = pd.DataFrame(interp)
    interp.to_csv(f'{out_dir}/matched_slack_by_seed.csv', index=False)

    summary = interp.groupby(['budget', 'mode'])[METRICS].agg(['mean', 'std', 'count'])
    summary.to_csv(f'{out_dir}/matched_slack_summary.csv')

    # seed-paired differences vs Strategy 1
    paired = {}
    for other in ('fwdmax', 'cbase'):
        for b in BUDGETS:
            a = interp[(interp['mode'] == other) & (interp.budget == b)].set_index('seed')
            p = interp[(interp['mode'] == 'proposed') & (interp.budget == b)].set_index('seed')
            common = a.index.intersection(p.index)
            if len(common) == 0:
                continue
            d = a.loc[common, METRICS] - p.loc[common, METRICS]
            paired[f'{other} - proposed @ slack {b}'] = {
                m: {'mean': float(d[m].mean()), 'min': float(d[m].min()), 'max': float(d[m].max()),
                    'n_seeds': int(d[m].notna().sum())} for m in METRICS}
    json.dump(paired, open(f'{out_dir}/paired_vs_proposed.json', 'w'), indent=1)

    argmax = res[res.readout == 'argmax'].groupby('mode')[['slack_ratio'] + METRICS].agg(['mean', 'std'])
    argmax.to_csv(f'{out_dir}/argmax_summary.csv')

    for metric, label in (('cover@1', 'spikes covered, no extra delay'),
                          ('cover@3', 'spikes covered, 15-min actuation delay'),
                          ('cover@6', 'spikes covered, 30-min actuation delay'),
                          ('recall_paper', 'spike recall (paper window)'),
                          ('throttle_core_h@3', 'throttled core-hours, 15-min delay (lower is better)')):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for (mode, seed), g in res.groupby(['mode', 'seed']):
            c = COLORS.get(mode, '#333')
            curve = (g if mode == 'seasonal-max' else g[g.readout.str.startswith('q=')]).sort_values('slack_ratio')
            ax.plot(curve.slack_ratio, curve[metric], '-o', ms=3, color=c, alpha=0.75,
                    label=mode if seed in (0, -1) else None)
            am = g[g.readout == 'argmax']
            ax.scatter(am.slack_ratio, am[metric], marker='*', s=140, color=c, edgecolor='black', zorder=4)
        ax.set_xscale('log')
        ax.set_xlabel('over-provisioning: slack / usage (log)')
        ax.set_ylabel(label)
        ax.set_title('quantile readouts (lines, one per seed); argmax = star')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{out_dir}/seeds_{metric.replace("@", "_at_")}.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:4])
