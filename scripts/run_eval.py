"""Evaluate baselines and trained models on the test traces; write results and Pareto plots.

usage: python run_eval.py <data.npz> <repo_root> <models_dir> <out_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import baselines as B
import metrics as M
from train import decode, LAG

QS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
N_BOOT = 500


def forecasters(Xte, models_dir):
    f = {'persistence': ('naive', B.persistence(Xte))}
    for w in (3, 6, 12, 24, 48, 96, 288):
        for m in (0.0, 0.1, 0.25):
            f[f'rollmax w={w} m={m}'] = ('rolling-max', B.rolling_max(Xte, w, m))
    for w in (6, 12, 24):
        for m in (0.0, 0.1):
            f[f'seasonal w={w} m={m}'] = ('seasonal-max', B.seasonal_max(Xte, w, margin=m))
    for q in (0.9, 0.95, 0.99):
        f[f'rollquant w=288 q={q}'] = ('rolling-quantile', B.rolling_quantile(Xte, 288, q))
    f['oracle max-future (Table 2)'] = ('oracle', B.oracle_future_max(Xte))
    f['oracle Strategy-1 target'] = ('oracle', B.proposed_target(Xte))
    for mode in ('cbase', 'proposed', 'proposed_full'):
        path = os.path.join(models_dir, mode, 'test_probs.npz')
        if not os.path.exists(path):
            continue
        probs = np.load(path)['probs']
        pad = np.full((probs.shape[0], LAG), np.nan)
        f[f'{mode} argmax'] = (f'{mode} argmax', np.concatenate([pad, decode(probs)], axis=1))
        for q in QS:
            f[f'{mode} q={q}'] = (f'{mode} quantile', np.concatenate([pad, decode(probs, q)], axis=1))
    return f


def per_trace_table(Xte, Y, req):
    return pd.DataFrame([M.evaluate_trace(Xte[i], Y[i], req[i]) for i in range(len(Xte))])


def bootstrap(df_a, df_b, rng):
    """95% CI of recall_paper and slack_ratio differences (a - b), resampling traces."""
    n = len(df_a)
    out = {'d_recall': [], 'd_slack': [], 'd_cover3': []}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a, b = df_a.iloc[idx].sum(), df_b.iloc[idx].sum()
        out['d_recall'].append(a.hit_paper / a.real_spikes - b.hit_paper / b.real_spikes)
        out['d_cover3'].append(a['cover@3'] / a.real_spikes - b['cover@3'] / b.real_spikes)
        out['d_slack'].append(a.slack / a.usage - b.slack / b.usage)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def main(data_path, repo_root, models_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    z = np.load(data_path, allow_pickle=True)
    split = z['split']
    te = np.where(split == 'test')[0]
    Xte = z['X'][te, 0].astype(np.float64)
    req = z['cpu_request'][te] / 100.0                 # cores (100 = 1 core)
    mu, sd = float(z['mu'][0]), float(z['sd'][0])

    fc = forecasters(Xte, models_dir)
    rows, tables = [], {}
    paper_set = {'persistence', 'oracle max-future (Table 2)', 'rollmax w=12 m=0.0', 'rollmax w=12 m=0.1',
                 'cbase argmax', 'proposed argmax', 'cbase q=0.9', 'cbase q=0.95'}
    for name, (family, Y) in fc.items():
        Y = np.where(np.isnan(Y), 0.0, Y)
        Y[:, :LAG] = 0.0                                # as in the paper's pipeline (zero padding)
        tab = per_trace_table(Xte, Y, req)
        tables[name] = tab
        r = M.summarize(tab.to_dict('records'))
        r.update(name=name, family=family)
        if name in paper_set:
            r.update({f'paper_{k}': v for k, v in M.evaluate_paper(Xte, Y, mu, sd, repo_root).items()})
        rows.append(r)
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)

    res = pd.DataFrame(rows).set_index('name')
    res.to_csv(f'{out_dir}/results.csv')

    rng = np.random.default_rng(0)
    cis = {}
    if 'proposed argmax' in tables:
        ref = tables['proposed argmax']
        for other in ('cbase argmax', 'rollmax w=12 m=0.0', 'rollmax w=12 m=0.1', 'cbase q=0.9', 'cbase q=0.95'):
            if other in tables:
                cis[f'{other} - proposed argmax'] = bootstrap(tables[other], ref, rng)
    json.dump(cis, open(f'{out_dir}/bootstrap_ci.json', 'w'), indent=1)

    colors = {'rolling-max': '#4C78A8', 'seasonal-max': '#72B7B2', 'rolling-quantile': '#9D755D',
              'cbase quantile': '#F58518', 'proposed quantile': '#B279A2', 'proposed_full quantile': '#FF9DA6',
              'proposed_full argmax': '#2E7D32', 'naive': '#999999',
              'oracle': '#000000', 'cbase argmax': '#E45756', 'proposed argmax': '#54A24B'}
    matched = {}
    if 'proposed argmax' in res.index:
        budget = res.loc['proposed argmax', 'slack_ratio']
        for fam, g in res.groupby('family'):
            ok = g[g.slack_ratio <= budget * 1.0001]
            if len(ok):
                matched[fam] = {c: float(ok[c].max()) for c in ('recall_paper', 'cover@1', 'cover@3', 'f1_fwd')}
                matched[fam].update({'throttle_core_h@3 (min)': float(ok['throttle_core_h@3'].min()),
                                     'configs_within_budget': int(len(ok))})
        json.dump({'slack_budget (proposed argmax)': float(budget), 'best_within_budget': matched},
                  open(f'{out_dir}/matched_slack.json', 'w'), indent=1)

    for ycol, ylabel, higher in (('recall_paper', 'spike recall (paper window, raw-unit band)', True),
                                 ('cover@1', 'spikes covered, no extra delay', True),
                                 ('cover@3', 'spikes covered with 15-min actuation delay', True),
                                 ('throttle_core_h@3', 'throttled core-hours, 15-min delay (lower is better)', False),
                                 ('f1_fwd', 'spike F1 (forward-only precision)', True)):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for fam, g in res.groupby('family'):
            c = colors.get(fam, '#333')
            star = 'argmax' in fam or fam in ('oracle', 'naive')
            ax.scatter(g.slack_ratio, g[ycol], marker='*' if star else 'o', s=160 if star else 18,
                       color=c, label=fam, zorder=3, edgecolor='white' if star else None)
            g = g.sort_values('slack_ratio')
            best, front = (-np.inf if higher else np.inf), []
            for _, r in g.iterrows():
                if (r[ycol] > best) if higher else (r[ycol] < best):
                    best = r[ycol]
                    front.append((r.slack_ratio, r[ycol]))
            if len(front) > 1 and not star:
                fx, fy = zip(*front)
                ax.plot(fx, fy, '-', color=c, alpha=0.8)
        if 'proposed argmax' in res.index:
            ax.axvline(res.loc['proposed argmax', 'slack_ratio'], color='#54A24B', ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('over-provisioning: slack / usage (log)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{out_dir}/pareto_{ycol.replace("@", "_at_")}.png', dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:5])
