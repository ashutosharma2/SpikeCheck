"""Strategy 2 against cbase at matched over-provisioning (docs/adr/013-strategy-2.md).

Every model is read along its quantile curve at the same overall slack budgets as score_hold.py,
with the same paired bootstrap over whole traces. Variants are paired with cbase, fwdmax and
proposed by seed. The rule in ADR-013 reads coverage at a 5-minute delay and slack 0.8.

Each model is also read the paper's way, at its most likely class. That reading lands at one
slack, so it is compared with cbase's quantile curve read at the same slack. Spike F1 on the
authors' evaluator uses the same most-likely-class forecast.

Before any variant is scored, the comparison models have to reproduce numbers already published:
data/novelty_val/dry_run on validation, results/horizon30 and results/replication on test.

usage:
  python score_s2.py val <data.npz> <models_s2_dir> <novelty_val_dir> <out_dir> <dry_run_dir>
      dry run on the validation traces (seed 0)
  python score_s2.py test <data.npz> <models_s2_dir> <models_seeds_dir> <out_dir> <results_dir> <frozen_commit>
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

from eval_seeds import BUDGETS, QS                                             # noqa: E402
from metrics import START, evaluate_paper                                      # noqa: E402
from score_hold import GROUPS, SHOW, group_members, load_probs, seed_row, summarise, trace_sums   # noqa: E402
from score_types import DELAYS, N_BOOT, at_budgets, prepare, spike_index       # noqa: E402
from train import LAG, decode                                                  # noqa: E402
from train_s2 import VARIANTS                                                  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, 'upstream', 'ResourceForecasting')
BASE, LOOK_AHEAD, STRATEGY_1 = 'cbase', 'fwdmax', 'proposed'
COMPARED = (BASE, LOOK_AHEAD, STRATEGY_1)
ALTERED = tuple(v for v in VARIANTS if v != 'single')
PAIRS = [(v, other) for v in VARIANTS for other in COMPARED] + [(v, 'single') for v in ALTERED]
LOW_QS = [0.01, 0.02, 0.05]      # variants only: binary's curve can start above the rule's slack at q = 0.1
PRIMARY = 0.8                    # the slack budget the rule reads
RULE_DELAY = 1                   # 5 minutes
AHEAD_DELAY = 3                  # 15 minutes, where the prediction compares with fwdmax
MARGIN = 0.01                    # the single variant's margin around cbase, absolute coverage
PAPER_F1 = {'cross-entropy': 0.275, 'single': 0.387}   # the paper's Table 4, Alibaba TempInc
FROZEN = ['scripts', 'strategy2', 'docs/adr/013-strategy-2.md', 'requirements.txt']

UNREAD = 'the quantile curve does not reach this slack'
SINGLE_OUTCOMES = {'match': 'single matches cbase to within 1 point',
                   'differ': 'single differs from cbase',
                   'open': 'not distinguished from cbase, and not shown to match it'}
ALTERED_OUTCOMES = {'adds': 'adds coverage that a quantile readout of cbase does not give',
                    'none': 'not shown to add anything at equal slack',
                    'worse': 'worse than a quantile readout of cbase'}


def at_slack(slack, vals, target):
    """Read a curve at a different slack under every trace weighting, as score_types.at_budgets.
    slack: (K, R), vals: (K, F, R), target: (R,). Returns (F, R), NaN outside the curve."""
    K, F, R = vals.shape
    xs = np.log(slack)
    order = np.argsort(xs, axis=0)
    xs = np.take_along_axis(xs, order, axis=0)
    vals = np.take_along_axis(vals, order[:, None, :], axis=0)
    b, r = np.log(target), np.arange(R)
    i = np.clip((xs <= b).sum(axis=0) - 1, 0, K - 2)
    lam = (b - xs[i, r]) / (xs[i + 1, r] - xs[i, r])
    out = (1 - lam)[:, None] * vals[i, :, r] + lam[:, None] * vals[i + 1, :, r]    # (R, F)
    out[~((xs[0] <= b) & (b <= xs[-1]))] = np.nan
    return out.T


def decide_single(lo, hi):
    """ADR-013's rows for the single variant, read from the top."""
    if np.isnan(lo) or np.isnan(hi):
        return UNREAD
    if lo > -MARGIN and hi < MARGIN:
        return SINGLE_OUTCOMES['match']
    return SINGLE_OUTCOMES['differ' if lo > 0 or hi < 0 else 'open']


def decide_altered(lo, hi):
    """ADR-013's rows for binary, proportional and exponential."""
    if np.isnan(lo) or np.isnan(hi):
        return UNREAD
    return ALTERED_OUTCOMES['adds' if lo > 0 else 'worse' if hi < 0 else 'none']


def predictions(outcomes, trails):
    """The three predictions in ADR-013, each True, False, or None when the result can't say."""
    single = outcomes['single']
    return {'single is no different from cbase':
            True if single == SINGLE_OUTCOMES['match'] else False if single == SINGLE_OUTCOMES['differ'] else None,
            'no altered variant beats cbase at equal slack':
            None if any(outcomes[v] == UNREAD for v in ALTERED)
            else not any(outcomes[v] == ALTERED_OUTCOMES['adds'] for v in ALTERED),
            'all four variants trail fwdmax at a 15-minute delay': all(trails.values())}


def context(X, req, member, n_boot=N_BOOT):
    """What every model is scored against. Column 0 of W is the full set, the rest are resamples."""
    N = len(X)
    tid, pos = spike_index(X)
    rng = np.random.default_rng(0)
    W = np.ones((N, n_boot + 1))
    W[:, 1:] = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(n_boot)], axis=1)
    n_group = np.stack([np.bincount(tid, weights=member[:, g], minlength=N) for g in range(member.shape[1])])
    return dict(X=X, req=req, tid=tid, pos=pos, member=member, W=W, n_group=n_group, usage=X[:, START:].sum(axis=1))


def forecast(probs, q=None):
    """Forecast for every step: the q-quantile, or the most likely class when q is None."""
    pad = np.full((probs.shape[0], LAG), np.nan)
    return prepare(np.concatenate([pad, decode(probs, q)], axis=1))


def read_model(probs, ctx, qs=QS):
    """One model's quantile curve at every budget and its most-likely-class reading, under every
    trace weighting."""
    X, W, usage, n_group = ctx['X'], ctx['W'], ctx['usage'], ctx['n_group']

    def sums(q):
        slack, cov, thr = trace_sums(X, forecast(probs, q), ctx['req'], ctx['tid'], ctx['pos'], ctx['member'], ())
        return slack[0], cov, thr[:, 0]

    def cover_of(cov):
        with np.errstate(invalid='ignore', divide='ignore'):
            return (cov @ W) / (n_group @ W)

    points = [sums(q) for q in qs]
    slack, cov, thr = [np.stack([p[i] for p in points]) for i in range(3)]         # (K, N), (K, D, G, N), (K, D, N)
    K, D, G, R = len(qs), len(DELAYS), n_group.shape[0], W.shape[1]
    slack_r, cover_r = (slack @ W) / (usage @ W), cover_of(cov).reshape(K, -1, R)
    a_slack, a_cov, a_thr = sums(None)
    return dict(slack=slack_r, curve=cover_r,
                cover=at_budgets(slack_r, cover_r).reshape(-1, D, G, R),
                thr=at_budgets(slack_r, thr @ W).reshape(-1, D, R),
                argmax_slack=(a_slack @ W) / (usage @ W), argmax_cover=cover_of(a_cov), argmax_thr=a_thr @ W)


def base_at_argmax(base, model):
    """cbase's quantile curve read at the slack of the model's most-likely-class forecast. (D, G, R)"""
    D, G, R = model['argmax_cover'].shape
    return at_slack(base['slack'], base['curve'], model['argmax_slack']).reshape(D, G, R)


def guard(out_dir, frozen, repo=REPO):
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


def check_dry_run(curves, by_method):
    """On validation, fwdmax and proposed must equal the computed-hold dry run."""
    names = {'cover: all': 'cover', 'throttle core-hours: all': 'thr'}
    rows = by_method[by_method.method.isin([LOOK_AHEAD, STRATEGY_1]) & by_method.measure.isin(names)]
    assert len(rows), 'no dry-run rows to check against'
    worst = 0.0
    for _, r in rows.iterrows():
        c, idx = curves[r.method, int(r.seed)], (BUDGETS.index(r.budget), DELAYS.index(int(r.d)))
        got = c['cover'][(*idx, 0, 0)] if names[r.measure] == 'cover' else c['thr'][(*idx, 0)]
        worst = max(worst, abs(got - r.value) / max(abs(r.value), 1e-9))
    assert worst < 1e-5, f'the comparison models differ from the dry run by {worst}'
    print(f'fwdmax and proposed match the computed-hold dry run ({len(rows)} rows, {worst:.1e})', flush=True)


def check_published(curves, seeds, matched, f1, replication):
    """On test, the comparison models must equal the published tables, and cbase its published F1."""
    worst, n = 0.0, 0
    for _, r in matched[matched['mode'].isin(COMPARED) & matched.seed.isin(seeds)].iterrows():
        c, bi = curves[r['mode'], int(r.seed)], BUDGETS.index(r.budget)
        for j, d in enumerate(DELAYS):
            for got, want in ((c['cover'][bi, j, 0, 0], r[f'cover@{d}']), (c['thr'][bi, j, 0], r[f'throttle_core_h@{d}'])):
                if np.isnan(got) and np.isnan(want):
                    continue
                worst = max(worst, abs(got - want) / max(abs(want), 1e-9)) if not (np.isnan(got) or np.isnan(want)) else np.inf
                n += 1
    assert n, 'no published rows to check against'
    assert worst < 1e-6, f'the comparison models differ from matched_slack_by_seed.csv by {worst}'
    want = replication.loc[f'{BASE} argmax', 'paper_spikes_f1']
    assert abs(f1[BASE, 0]['spikes_f1'] - want) < 1e-5, (f1[BASE, 0]['spikes_f1'], want)
    print(f'the comparison models match the published tables ({n} values, {worst:.1e}) and the published F1', flush=True)


def main(split, data_path, s2_dir, others_dir, out_dir, reference, frozen=None):
    commit = guard(out_dir, frozen) if split == 'test' else None
    seeds = (0, 1, 2) if split == 'test' else (0,)

    z = np.load(data_path, allow_pickle=True)
    ids = np.where(z['split'] == split)[0]
    X = z['X'][ids, 0].astype(np.float64)
    req = z['cpu_request'][ids] / 100.0
    mu, sd = float(z['mu'][0]), float(z['sd'][0])
    tid, _ = spike_index(X)
    member = group_members(X, tid)
    ctx = context(X, req, member)
    n_spikes = member.sum(axis=0)

    def load(name, seed):
        if name not in VARIANTS:
            return load_probs(split, others_dir, name, seed)
        return np.load(f"{s2_dir}/{name}_s{seed}/{'val' if split == 'val' else 'test'}_probs.npz")['probs']

    curves, f1 = {}, {}

    def score(name, seed):
        probs = load(name, seed)
        assert probs.shape[0] == len(X)
        curves[name, seed] = read_model(probs, ctx, LOW_QS + QS if name in VARIANTS else QS)
        f1[name, seed] = evaluate_paper(X, forecast(probs), mu, sd, UPSTREAM)
        print(name, 'seed', seed, 'scored', flush=True)

    for name in COMPARED:
        for s in seeds:
            score(name, s)
    if split == 'val':
        check_dry_run(curves, pd.read_csv(f'{reference}/by_method.csv'))
    else:
        check_published(curves, seeds, pd.read_csv(f'{reference}/horizon30/matched_slack_by_seed.csv'), f1,
                        pd.read_csv(f'{reference}/replication/results.csv', index_col=0))
    for name in VARIANTS:
        for s in seeds:
            score(name, s)

    # each model read the paper's way, against cbase's quantile curve at the same slack
    methods = VARIANTS + COMPARED
    for name in methods:
        for s in seeds:
            curves[f'{name} argmax', s] = dict(read=curves[name, s]['argmax_cover'])
            curves[f'{BASE} quantile at {name} slack', s] = dict(read=base_at_argmax(curves[BASE, s], curves[name, s]))

    by_method, paired, argmax = [], [], []
    for name in methods:
        for s in seeds:
            c = curves[name, s]
            for b in SHOW:
                bi = BUDGETS.index(b)
                for j, d in enumerate(DELAYS):
                    by_method += [dict(method=name, seed=s, d=d, budget=b, measure=f'cover: {g}', value=c['cover'][bi, j, gi, 0],
                                       n_spikes=int(n_spikes[gi])) for gi, g in enumerate(GROUPS)]
                    by_method.append(dict(method=name, seed=s, d=d, budget=b, measure='throttle core-hours', value=c['thr'][bi, j, 0], n_spikes=''))
    summaries = {}
    for v, other in PAIRS:
        cover_s, thr_s = [summarise(curves, v, other, k, seeds) for k in ('cover', 'thr')]
        summaries[v, other] = cover_s
        for b in SHOW:
            bi = BUDGETS.index(b)
            for j, d in enumerate(DELAYS):
                paired += [dict(pair=f'{v} - {other}', d=d, budget=b, measure=f'cover: {g}', **seed_row(cover_s, (bi, j, gi), seeds),
                                n_spikes=int(n_spikes[gi])) for gi, g in enumerate(GROUPS) if g != 'unknown']
                base = np.mean([curves[other, s]['thr'][bi, j, 0] for s in seeds])
                row = seed_row(thr_s, (bi, j), seeds)
                paired.append(dict(pair=f'{v} - {other}', d=d, budget=b, measure='throttle core-hours', relative=row['mean'] / base, **row))
    for name in methods:
        gap = summarise(curves, f'{name} argmax', f'{BASE} quantile at {name} slack', 'read', seeds)
        for j, d in enumerate(DELAYS):
            argmax.append(dict(method=name, d=d, slack=np.mean([curves[name, s]['argmax_slack'][0] for s in seeds]),
                               cover=np.mean([curves[name, s]['argmax_cover'][j, 0, 0] for s in seeds]),
                               cbase_cover_at_same_slack=np.mean([curves[f'{BASE} quantile at {name} slack', s]['read'][j, 0, 0] for s in seeds]),
                               **seed_row(gap, (j, 0), seeds)))
    scores = [dict(method=name, seed=s, **f1[name, s]) for name in methods for s in seeds]

    # the rule in ADR-013
    bi, rule = BUDGETS.index(PRIMARY), DELAYS.index(RULE_DELAY)
    rows = {v: seed_row(summaries[v, BASE], (bi, rule, 0), seeds) for v in VARIANTS}
    outcomes = {v: (decide_single if v == 'single' else decide_altered)(r['ci_lo'], r['ci_hi']) for v, r in rows.items()}
    ahead = {v: seed_row(summaries[v, LOOK_AHEAD], (bi, DELAYS.index(AHEAD_DELAY), 0), seeds) for v in VARIANTS}
    trails = {v: bool(r['ci_hi'] < 0) for v, r in ahead.items()}
    checks = [dict(check=f'{v} minus cbase', minutes=5 * RULE_DELAY, budget=PRIMARY, outcome=outcomes[v], **rows[v]) for v in VARIANTS]
    checks += [dict(check=f'{v} minus fwdmax (prediction)', minutes=5 * AHEAD_DELAY, budget=PRIMARY,
                    outcome='trails fwdmax' if trails[v] else 'not shown to trail fwdmax', **ahead[v]) for v in VARIANTS]
    checks += [dict(check=f'{v} minus single (reported)', minutes=5 * RULE_DELAY, budget=PRIMARY, outcome='',
                    **seed_row(summaries[v, 'single'], (bi, rule, 0), seeds)) for v in ALTERED]
    predicted = predictions(outcomes, trails)

    os.makedirs(out_dir, exist_ok=split == 'val')
    for name, table in (('checks', checks), ('paired', paired), ('by_method', by_method), ('argmax', argmax), ('spike_f1', scores)):
        pd.DataFrame(table).to_csv(f'{out_dir}/{name}.csv', index=False, float_format='%.6g')
    with open(f'{out_dir}/conclusion.json', 'w') as f:
        json.dump(dict(split=split, frozen_commit=commit, seeds=list(seeds), traces=len(X), n_boot=N_BOOT, budget=PRIMARY, margin=MARGIN,
                       n_spikes={g: int(n) for g, n in zip(GROUPS, n_spikes)}, outcomes=outcomes, predictions=predicted,
                       altered_variants_adding_coverage=sum(outcomes[v] == ALTERED_OUTCOMES['adds'] for v in ALTERED)), f, indent=2)

    print(pd.DataFrame(checks)[['check', 'minutes', 'mean', 'seed_min', 'seed_max', 'ci_lo', 'ci_hi', 'outcome']].to_string(index=False, float_format='%.4f'))
    print(pd.DataFrame(argmax)[lambda t: t.d == RULE_DELAY][['method', 'slack', 'cover', 'cbase_cover_at_same_slack', 'mean', 'ci_lo', 'ci_hi']]
          .to_string(index=False, float_format='%.4f'))
    print(pd.DataFrame(scores).groupby('method').spikes_f1.mean().reindex(methods).to_string(float_format='%.3f'))
    print(json.dumps(predicted, indent=2))
    print('wrote', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:])
