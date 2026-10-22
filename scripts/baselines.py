"""Causal, training-free provisioning forecasts. Every y[t] uses only x[:t]."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DAY = 288  # 5-min steps per day


def _past_window_reduce(x, w, fn):
    """out[t] = fn(x[t-w:t]); NaN where the window does not fit."""
    out = np.full(x.shape, np.nan)
    win = sliding_window_view(x, w, axis=-1)          # win[..., j, :] = x[..., j:j+w]
    out[..., w:] = fn(win[..., :-1, :], axis=-1)
    return out


def persistence(X):
    Y = np.full(X.shape, np.nan)
    Y[:, 1:] = X[:, :-1]
    return Y


def rolling_max(X, w, margin=0.0):
    """Autopilot-style moving-window peak recommender: (1+m) * max of the last w samples."""
    return (1 + margin) * _past_window_reduce(X, w, np.max)


def rolling_quantile(X, w, q):
    """Autopilot-style percentile recommender over the last w samples (row by row to bound memory)."""
    return np.stack([_past_window_reduce(row, w, lambda a, axis: np.quantile(a, q, axis=axis)) for row in X])


def seasonal_max(X, w, half_width=6, margin=0.0):
    """max(rolling max of last w, max of x around the same time yesterday)."""
    rm = _past_window_reduce(X, w, np.max)
    seas = np.full(X.shape, np.nan)
    width = 2 * half_width + 1
    win = sliding_window_view(X, width, axis=-1)       # win[:, j] = x[j : j+width]
    # for time t use x[t-DAY-h : t-DAY+h+1]  ->  j = t - DAY - h
    t0 = DAY + half_width
    seas[:, t0:] = win[:, : X.shape[1] - t0].max(axis=-1)
    return (1 + margin) * np.fmax(rm, seas)


def oracle_future_max(X, k=3, margin=0.1):
    """Paper's 'max future filter' (Table 2): uses the future, not implementable."""
    Y = np.full(X.shape, np.nan)
    win = sliding_window_view(X, k, axis=-1)
    Y[:, : X.shape[1] - k + 1] = (1 + margin) * win.max(axis=-1)
    return Y


def proposed_target(X, past=7, future=5):
    """Paper's Strategy-1 target as implemented: max(x[t-past+1 : t+future])."""
    Y = np.full(X.shape, np.nan)
    width = past - 1 + future
    win = sliding_window_view(X, width, axis=-1)       # win[:, j] = x[j : j+width]
    Y[:, past - 1: past - 1 + win.shape[1]] = win.max(axis=-1)
    return Y
