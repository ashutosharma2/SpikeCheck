"""Corrected provisioning metrics, plus a wrapper around the paper's original evaluator.

All corrected metrics work in raw CPU units (percent of request, >= 0), never divide
per point by x, and score each forecast at the time it would actually be in effect.

Changes from the paper's evaluation.py (see agullo2025_critical_review.md, section D):
  D1  slack/shortfall are sums over the trace divided by total usage (well-conditioned),
      plus exceedance rate and core-hours; no shift of the minimum to 1e-10.
  D2  tolerance bands are applied to raw CPU, so they do not depend on an arbitrary origin.
  D3  the spike detector keeps a full-length trailing window and adds a refractory period
      and a minimum absolute rise, so it no longer fires on every uptick after a spike.
  D4  precision only looks forward (a predicted spike must come at or before the real
      level), so lagging copies are no longer rewarded.
  D6  counts are pooled across traces; traces with no spikes do not score 0.
Added: coverage recall and throttling cost under an actuation delay d.
"""
import importlib.util
import os
import sys

import numpy as np

GAMMA, SIGMA, ALPHA, BETA, TAU = 250, 2.0, 0.2, 0.5, 4   # paper constants (§4.1.4)
LAG = 512                                                 # paper's lag; forecasts exist from here
START = LAG + 8                                           # metrics start here (room for delay d <= 8)
MIN_RISE = 2.0                                            # CPU-percent points
REFRACTORY = TAU


def _weighted_rolling_stats(x, window):
    """Mean/std of x[k-window:k] with linearly increasing weights, as in the paper (Alg. 1)."""
    w = np.arange(1, window + 1, dtype=np.float64) / window  # oldest ... newest
    w = w / w.sum()
    m1 = np.convolve(x, w[::-1], mode='valid')        # m1[j] = weighted mean of x[j:j+window]
    m2 = np.convolve(x * x, w[::-1], mode='valid')
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    mean = np.full(x.shape, np.nan)
    sd = np.full(x.shape, np.nan)
    mean[window:] = m1[:-1]                              # stats of the window ending at k-1
    sd[window:] = std[:-1]
    return mean, sd


def find_spikes(x, start=LAG + GAMMA):
    mean, sd = _weighted_rolling_stats(x.astype(np.float64), GAMMA)
    rise = x - mean
    cand = np.where((rise > SIGMA * sd) & (rise > MIN_RISE))[0]
    cand = cand[cand >= start]
    spikes, last = [], -10 ** 9
    for k in cand:
        if k - last > REFRACTORY:
            spikes.append(k)
            last = k
    return np.asarray(spikes, dtype=np.int64)


def in_band(v, ref):
    return ((1 - ALPHA) * ref < v) & (v < (1 + BETA) * ref)


def evaluate_trace(x, y, req_cores, delays=(1, 3, 6)):
    """x, y: raw CPU (% of request) for the whole trace; y[t] is the forecast for time t."""
    T = len(x)
    xs, ys = x[START:], y[START:]
    out = {
        'usage': xs.sum(),
        'slack': np.maximum(ys - xs, 0).sum(),
        'short': np.maximum(xs - ys, 0).sum(),
        'exceed': (xs > ys).sum(),
        'n': len(xs),
    }
    real = find_spikes(x)
    pred = find_spikes(y)
    out['real_spikes'] = len(real)
    out['pred_spikes'] = len(pred)
    # paper-style recall window [k - tau, k], band on raw values
    out['hit_paper'] = sum(bool(in_band(y[max(k - TAU, LAG):k + 1], x[k]).any()) for k in real)
    # forward-only precision: predicted level must appear in x at or after the predicted spike
    out['rel_fwd'] = sum(bool(in_band(x[k:min(k + TAU + 1, T)], y[k]).any()) for k in pred)
    # paper's symmetric precision for comparison
    out['rel_sym'] = sum(bool(in_band(x[max(k - TAU, LAG):min(k + TAU + 1, T)], y[k]).any()) for k in pred)
    for d in delays:
        # allocation in effect at t is the forecast issued d steps earlier, i.e. y[t - d + 1]
        alloc = np.full(T, np.nan)
        alloc[d - 1:] = y[:T - d + 1]
        a = alloc[START:]
        out[f'cover@{d}'] = sum(bool(alloc[k] >= (1 - ALPHA) * x[k]) for k in real)
        out[f'throttle_core_h@{d}'] = np.maximum(xs - a, 0).sum() / 100 * req_cores * 5 / 60
        out[f'slack_core_h@{d}'] = np.maximum(a - xs, 0).sum() / 100 * req_cores * 5 / 60
    return out


def summarize(per_trace, delays=(1, 3, 6)):
    """Pool counts across traces (micro-averaging)."""
    s = {k: float(np.sum([p[k] for p in per_trace])) for k in per_trace[0]}
    r = {
        'slack_ratio': s['slack'] / s['usage'],
        'short_ratio': s['short'] / s['usage'],
        'exceed_rate': s['exceed'] / s['n'],
        'recall_paper': s['hit_paper'] / max(s['real_spikes'], 1),
        'prec_fwd': s['rel_fwd'] / max(s['pred_spikes'], 1),
        'prec_sym': s['rel_sym'] / max(s['pred_spikes'], 1),
        'real_spikes': s['real_spikes'],
        'pred_spikes': s['pred_spikes'],
    }
    p, rc = r['prec_fwd'], r['recall_paper']
    r['f1_fwd'] = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
    for d in delays:
        r[f'cover@{d}'] = s[f'cover@{d}'] / max(s['real_spikes'], 1)
        r[f'throttle_core_h@{d}'] = s[f'throttle_core_h@{d}']
        r[f'slack_core_h@{d}'] = s[f'slack_core_h@{d}']
    return r


def evaluate(X, Y, req, delays=(1, 3, 6)):
    """X, Y: (N, T) raw CPU; req: (N,) cpu_request in cores."""
    return summarize([evaluate_trace(X[i], Y[i], req[i], delays) for i in range(len(X))], delays)


def _load_paper_evaluation(repo_root):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    path = os.path.join(repo_root, 'app', 'evaluation', 'evaluation.py')
    spec = importlib.util.spec_from_file_location('paper_evaluation', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_paper(X, Y, mu, sd, repo_root):
    """The authors' own evaluator, run on z-scored values exactly as in their pipeline."""
    ev = _load_paper_evaluation(repo_root)
    vals = [((X[i] - mu) / sd).astype(np.float64) for i in range(len(X))]
    preds = [((Y[i] - mu) / sd).astype(np.float64) for i in range(len(X))]
    vals, preds = ev.move_over_0(vals, preds)
    rows = [ev.compute_evaluation('x', str(i), vals[i], preds[i], '/tmp', LAG) for i in range(len(vals))]
    a = np.asarray(rows)
    return dict(zip(['over_prov', 'under_prov', 'spikes_prec', 'spikes_rec', 'spikes_f1'], a.mean(axis=0)))
