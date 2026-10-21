"""Turn the aggregated 5-min arrays into the experiment dataset.

Approximates the paper's Alibaba preprocessing (its scripts are not public; §4.1.2):
long-running containers, one per app_du (done during streaming), drop traces with many
nulls, impute, drop traces whose max CPU is close to their average, 8 days at 5 min.
Standardisation stats are fit on TRAIN traces only (the paper's scope is unclear).

usage: python prep.py <aggregated.npz> <chosen_containers.csv> <out.npz>
"""
import sys

import numpy as np
import pandas as pd

NSTEPS = 2304          # 8 days of 5-min bins
OFFSET = 288           # usage timestamps start at ~86,400 s; bins 288..2591 hold the 8 days
MAX_NULL_FRAC = 0.10   # "filter out the time series that contain many null values"
MIN_PEAK_GAP = 5.0     # drop if max - mean < 5 CPU-percent points ("max very close to average")
SEED = 0

FEATURES = ['cpu_max', 'cpu_mean', 'mem', 'cpi', 'mpki', 'net_in', 'net_out', 'disk']


def ffill_bfill(a):
    """Forward then backward fill NaNs along the last axis."""
    out = a.copy()
    for row in out.reshape(-1, out.shape[-1]):
        mask = np.isnan(row)
        if mask.all():
            row[:] = 0.0
            continue
        idx = np.where(~mask, np.arange(row.size), 0)
        np.maximum.accumulate(idx, out=idx)
        row[:] = row[idx]
        first = np.argmax(~mask)
        row[:first] = row[first]
    return out


def main(agg_path, chosen_path, out_path):
    z = np.load(agg_path, allow_pickle=True)
    cpu_max = z['cpu_max'][:, OFFSET:OFFSET + NSTEPS].astype(np.float64)
    cnt_cpu = z['cnt_cpu'][:, OFFSET:OFFSET + NSTEPS].astype(np.float64)
    sums = z['sums'][:, :, OFFSET:OFFSET + NSTEPS].astype(np.float64)
    cnt_oth = z['cnt_oth'][:, :, OFFSET:OFFSET + NSTEPS].astype(np.float64)
    cids = z['cids']
    chosen = pd.read_csv(chosen_path).set_index('cid').loc[cids]

    with np.errstate(invalid='ignore', divide='ignore'):
        cpu_max[cnt_cpu == 0] = np.nan
        cpu_mean = np.where(cnt_cpu > 0, sums[0] / cnt_cpu, np.nan)
        others = np.where(cnt_oth > 0, sums[1:] / cnt_oth, np.nan)

    null_frac = np.isnan(cpu_max).mean(axis=1)
    keep = null_frac <= MAX_NULL_FRAC
    print(f'traces: {len(cids)}; with <= {MAX_NULL_FRAC:.0%} null bins: {keep.sum()}')

    X = np.concatenate([cpu_max[None], cpu_mean[None], others], axis=0)[:, keep]  # (F, N, T)
    X = ffill_bfill(X)
    cids, chosen = cids[keep], chosen[keep]

    target = X[0]
    variable = (target.max(axis=1) - target.mean(axis=1)) >= MIN_PEAK_GAP
    print(f'with max - mean >= {MIN_PEAK_GAP} pts: {variable.sum()}')
    X, cids, chosen = X[:, variable], cids[variable], chosen[variable]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(cids))
    n_tr, n_va = int(0.7 * len(cids)), int(0.15 * len(cids))
    split = np.empty(len(cids), dtype='<U5')
    split[perm[:n_tr]], split[perm[n_tr:n_tr + n_va]], split[perm[n_tr + n_va:]] = 'train', 'val', 'test'

    X = np.transpose(X, (1, 0, 2)).astype(np.float32)  # (N, F, T)
    tr = X[split == 'train']
    mu = tr.mean(axis=(0, 2))
    sd = tr.std(axis=(0, 2)) + 1e-6
    np.savez_compressed(out_path, X=X, cids=cids, split=split, mu=mu, sd=sd,
                        cpu_request=chosen['req'].values, features=np.array(FEATURES))
    print({s: int((split == s).sum()) for s in ('train', 'val', 'test')})


if __name__ == '__main__':
    main(*sys.argv[1:4])
