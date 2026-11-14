"""Can a small trained selector pick a policy per trace better than one fixed policy?

One decision per trace, at the first scored step (START). The selector sees only the steps before
it, as FEATURES, and picks one policy from oracle_headroom.py's library for the rest of the trace.

The selector is a shallow decision tree. Its labels are the policies hindsight picks for the
training traces (oracle_headroom.hindsight). It is cross-fitted on the validation traces: the
traces are cut into folds, and each fold gets its policies from a tree trained on the other folds,
so no trace is scored by a tree that saw it. The comparison is the fixed policy that was best on
those same training folds. Both are scored as oracle_headroom.py scores any assignment.

The rule, fixed before the run: read the gain at a 15-minute delay and slack 0.8, and continue
towards a learned or LLM selector only if the lower bound of its interval is above 0.

Validation traces and seed 0 only. Intervals resample whole traces with the assignments frozen:
they cover trace sampling, not the training of the trees.

usage: python trained_selector.py <data.npz> <val_probs_dir> <chosen_containers.csv> <out_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from oracle_headroom import (BUDGETS, DELAYS, POLICIES, SHOW, best_fixed, hindsight, interval, load_parts, mix_curve,
                             trace_weights)
from metrics import GAMMA, START, find_spikes                        # noqa: E402 (path set by oracle_headroom)

DECIDE = START                     # the selector sees steps [0, DECIDE) and decides once
DAY = 288
CLOSE = 6                          # steps, the follow-on window of ADR-009
FOLDS = 5
DEPTH = 3
MIN_LEAF = 20
PRIMARY = (3, 0.8)                 # delay in steps and slack budget the rule reads
FEATURES = ('cpu_mean', 'cpu_std', 'peak_ratio', 'step_change', 'acf_1', 'acf_day', 'early_spikes',
            'close_share', 'mem_mean', 'net_mean', 'cpu_request', 'limit_ratio')


def autocorr(x, lag):
    a, b = x[:-lag] - x[:-lag].mean(), x[lag:] - x[lag:].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def features(X, request, limit_ratio):
    """X: (N, 8, T) raw features in prep.py's order. Uses steps before DECIDE only. Returns (N, len(FEATURES))."""
    rows = []
    for x, req, lim in zip(X[:, :, :DECIDE].astype(np.float64), request, limit_ratio):
        cpu = x[0]
        spikes = find_spikes(cpu, start=GAMMA)
        gaps = np.diff(spikes)
        rows.append([cpu.mean(), cpu.std(), cpu.max() / (cpu.mean() + 1e-6), np.abs(np.diff(cpu)).mean(),
                     autocorr(cpu, 1), autocorr(cpu, DAY), len(spikes),
                     float((gaps <= CLOSE).mean()) if len(gaps) else 0.0,
                     x[2].mean(), (x[5] + x[6]).mean(), req, lim])
    return np.array(rows)


def fold_ids(N, k=FOLDS, seed=0):
    """A fold number for every trace: a seeded shuffle dealt into k folds."""
    out = np.empty(N, dtype=np.int64)
    out[np.random.default_rng(seed).permutation(N)] = np.arange(N) % k
    return out


def subset(part, idx):
    """The same sums for some of the traces."""
    return {k: v[..., idx] for k, v in part.items()}


def cross_fit(part, F, fold, j, budget):
    """Every trace's policy from a tree trained on the other folds, and the fixed policy that was
    best on those folds. Returns (assignment, fixed assignment), each (N,)."""
    assign = np.empty(len(fold), dtype=np.int64)
    fixed = np.empty(len(fold), dtype=np.int64)
    for f in np.unique(fold):
        train, held = fold != f, fold == f
        tr = subset(part, train)
        base = best_fixed(tr, j, budget)
        labels = hindsight(tr, j, budget, base)
        tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=0).fit(F[train], labels)
        assign[held], fixed[held] = tree.predict(F[held]), base
    return assign, fixed


def decide(ci_lo):
    return 'continue: the selector beats the fixed policy' if ci_lo > 0 else 'stop: the selector is not shown to beat the fixed policy'


def main(data_path, probs_dir, chosen_path, out_dir):
    ids, parts = load_parts(data_path, probs_dir)
    part = parts['all']
    z = np.load(data_path, allow_pickle=True)
    chosen = pd.read_csv(chosen_path).set_index('cid').loc[z['cids'][ids]]
    x_npy = os.path.join(os.path.dirname(data_path), 'X.npy')        # uncompressed copy, as train.py
    X = np.load(x_npy, mmap_mode='r') if os.path.exists(x_npy) else z['X']
    F = features(np.asarray(X[ids][:, :, :DECIDE]), z['cpu_request'][ids] / 100.0, (chosen.lim / chosen.req).values)
    assert np.isfinite(F).all()
    N = len(F)
    fold, W, names, D = fold_ids(N), trace_weights(N), list(POLICIES), len(DELAYS)

    rows, picks = [], []
    for j, d in enumerate(DELAYS):
        for b in SHOW:
            bi = BUDGETS.index(b)
            a, fixed = cross_fit(part, F, fold, j, b)
            mix, base = mix_curve(part, a, W)[bi], mix_curve(part, fixed, W)[bi]
            rows.append(dict(d=d, budget=b, fixed_cover=base[j, 0], selector_cover=mix[j, 0], **interval(mix[j] - base[j]),
                             fixed_throttle=base[D + j, 0], selector_throttle=mix[D + j, 0],
                             traces_moved=float((a != fixed).mean()),
                             fixed_policies=' / '.join(sorted({names[p] for p in fixed}))))
            picks += [dict(d=d, budget=b, policy=names[p], traces=int((a == p).sum())) for p in range(len(names))]
            print(f'{5 * d} min, slack {b}: done', flush=True)

    # a tree on every validation trace at the primary cell, only to show what it learns
    j, b = DELAYS.index(PRIMARY[0]), PRIMARY[1]
    labels = hindsight(part, j, b, best_fixed(part, j, b))
    tree = DecisionTreeClassifier(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=0).fit(F, labels)
    shown = export_text(tree, feature_names=list(FEATURES), class_names=[names[c] for c in tree.classes_])

    table = pd.DataFrame(rows)
    primary = table[(table.d == PRIMARY[0]) & (table.budget == PRIMARY[1])].iloc[0]
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(f'{out_dir}/selector.csv', index=False, float_format='%.6g')
    pd.DataFrame(picks).to_csv(f'{out_dir}/picks.csv', index=False)
    with open(f'{out_dir}/tree.txt', 'w') as f:
        f.write(f'trained on all {N} validation traces, {5 * PRIMARY[0]} min, slack {b}; shown, not scored\n'
                f'accuracy on its own training labels: {tree.score(F, labels):.3f}\n\n{shown}')
    with open(f'{out_dir}/summary.json', 'w') as f:
        json.dump(dict(split='val', seed=0, traces=N, folds=FOLDS, depth=DEPTH, min_leaf=MIN_LEAF, features=list(FEATURES),
                       decide_step=DECIDE, primary=dict(minutes=5 * PRIMARY[0], budget=b), rule='ci_lo > 0',
                       gain=float(primary.gain), ci_lo=float(primary.ci_lo), ci_hi=float(primary.ci_hi),
                       conclusion=decide(primary.ci_lo)), f, indent=2)
    cols = ['d', 'budget', 'fixed_cover', 'selector_cover', 'gain', 'ci_lo', 'ci_hi', 'traces_moved']
    print(table[cols].to_string(index=False, float_format='%.4f'))
    print('conclusion:', decide(primary.ci_lo))
    print('wrote', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:5])
