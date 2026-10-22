"""Train the paper's Temporal Inception classifier on either target and save test softmax.

  cbase     target = class of x[t]                    (paper's CBaseline)
  proposed  target = class of max(x[t-6 : t+5])       (paper's Strategy 1 target, as in its code)
  fwdmax    target = class of max(x[t : t+5])         (Strategy 1's 5-step look-ahead without the
            6-step past hold; isolates horizon from shaping)
  proposed_full  same target, plus the paper's input ("values") transform: max over the last 2
            samples, then log (approximated on raw non-negative features before z-scoring)

cbase and proposed use the same inputs (8 z-scored features, lag 512), so the only difference
is the target. proposed_full adds the input change the paper made at the same time (review C1).

usage: python train.py <data.npz> <repo_root> <mode> <out_dir> [minutes]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

LAG = 512
NCLS = 40
BIN_W = 100.0 / NCLS     # 40 equal bins over 0-100 % (paper: 40 equal intervals over the data range)
WINDOWS_PER_TRACE = 200  # sampled per epoch per training trace
VAL_WINDOWS = 50
BATCH = 2048
LR = 0.01                # paper's best Alibaba TempInc config: Adam(lr=0.01, amsgrad)
PATIENCE = 3
SEED = 0


def load_tempinc(repo_root):
    sys.path.insert(0, repo_root)
    from app.models.networks.types.temporal_inception.main import \
        DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed as TempInc
    return TempInc


def make_target(x, mode):
    """x: (N, T) raw CPU. Returns (N, T) class targets, -1 where undefined."""
    from baselines import proposed_target
    if mode == 'cbase':
        v = x.copy()
    elif mode == 'fwdmax':
        v = proposed_target(x, past=1, future=5)
    elif mode in ('proposed', 'proposed_full'):
        v = proposed_target(x)
    else:
        raise ValueError(mode)
    cls = np.where(np.isnan(v), -1, np.clip(np.floor(v / BIN_W), 0, NCLS - 1)).astype(np.int64)
    return cls


def gather(series, idx_trace, idx_t, offsets):
    # series: (N, F, T); returns (B, F, LAG) with x[t-LAG:t]
    cols = idx_t[:, None] - LAG + offsets[None, :]
    w = series[idx_trace[:, None], :, cols]          # (B, LAG, F)
    return w.permute(0, 2, 1).float().contiguous()


def main(data_path, repo_root, mode, out_dir, minutes=90, seed=SEED):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Memory: build each split's inputs on its own (float32), hand them to torch, and drop
    # the numpy copies. Targets are kept as int16 and widened per batch. The test split is
    # only materialised after training, once the train/val tensors are freed.
    z = np.load(data_path, allow_pickle=True)
    split, mu, sd = z['split'], z['mu'], z['sd']
    x_npy = os.path.join(os.path.dirname(data_path), 'X.npy')   # uncompressed copy, memory-mapped
    X = np.load(x_npy, mmap_mode='r') if os.path.exists(x_npy) else z['X'].astype(np.float32)
    T, n_feat = X.shape[2], X.shape[1]
    tgt = make_target(X[:, 0].astype(np.float64), mode).astype(np.int16)
    if mode == 'proposed_full':
        def inputs(a):
            v = a.copy()
            v[:, :, 1:] = np.maximum(a[:, :, 1:], a[:, :, :-1])
            return np.log1p(np.maximum(v, 0, out=v), out=v)
        tr_v = inputs(X[split == 'train'])
        mu, sd = tr_v.mean(axis=(0, 2)), tr_v.std(axis=(0, 2)) + 1e-6
        del tr_v
    else:
        def inputs(a):
            return a                                   # X[ids] is already a fresh copy
    mu = mu.astype(np.float32)[None, :, None]
    sd = sd.astype(np.float32)[None, :, None]

    def subset(name):
        ids = np.where(split == name)[0]
        a = inputs(X[ids])
        a -= mu
        a /= sd
        series = torch.from_numpy(a.astype(np.float16)).to(dev)   # half precision on device
        del a
        return series, torch.from_numpy(tgt[ids]).to(dev), ids

    tr_s, tr_y, _ = subset('train')
    va_s, va_y, _ = subset('val')
    del X                                              # reloaded for the test split after training
    offsets = torch.arange(LAG, device=dev)

    def sample(n_tr, k):
        i = rng.integers(0, n_tr, size=n_tr * k)
        t = rng.integers(LAG, T - 5, size=n_tr * k)    # proposed target needs t+4 < T
        return torch.tensor(i, device=dev), torch.tensor(t, device=dev)

    vi, vt = sample(va_s.shape[0], VAL_WINDOWS)

    TempInc = load_tempinc(repo_root)
    model = TempInc(LAG, NCLS, n_feat, str(dev)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(vi), 8192):
                xb = gather(va_s, vi[s:s + 8192], vt[s:s + 8192], offsets)
                yb = va_y[vi[s:s + 8192], vt[s:s + 8192]].long()
                tot += F.cross_entropy(model(xb), yb, reduction='sum').item()
                n += len(yb)
        return tot / n

    best, bad, epoch, t0, log = float('inf'), 0, 0, time.time(), []
    ti = tt = perm = None
    while time.time() - t0 < minutes * 60 and bad < PATIENCE:
        model.train()
        ti, tt = sample(tr_s.shape[0], WINDOWS_PER_TRACE)
        perm = torch.randperm(len(ti), device=dev)
        ti, tt = ti[perm], tt[perm]
        for s in range(0, len(ti), BATCH):
            xb = gather(tr_s, ti[s:s + BATCH], tt[s:s + BATCH], offsets)
            yb = tr_y[ti[s:s + BATCH], tt[s:s + BATCH]].long()
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = val_loss()
        epoch += 1
        log.append({'epoch': epoch, 'val_loss': vl, 'minutes': (time.time() - t0) / 60})
        print(json.dumps(log[-1]), flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(model.state_dict(), f'{out_dir}/model.pt')
        else:
            bad += 1

    del tr_s, tr_y, va_s, va_y, ti, tt, perm
    if dev.type == 'mps':
        torch.mps.empty_cache()
    X = np.load(x_npy, mmap_mode='r') if os.path.exists(x_npy) else z['X'].astype(np.float32)
    te_s, _, te_ids = subset('test')
    del X, tgt
    model.load_state_dict(torch.load(f'{out_dir}/model.pt'))
    model.eval()
    probs = np.zeros((te_s.shape[0], T - LAG, NCLS), dtype=np.float16)
    with torch.no_grad():
        ts = torch.arange(LAG, T, device=dev)
        for i in range(te_s.shape[0]):
            ii = torch.full_like(ts, i)
            p = F.softmax(model(gather(te_s, ii, ts, offsets)), dim=1)
            probs[i] = p.cpu().numpy().astype(np.float16)
    np.savez_compressed(f'{out_dir}/test_probs.npz', probs=probs, test_ids=te_ids)
    json.dump({'mode': mode, 'seed': seed, 'best_val_loss': best, 'log': log}, open(f'{out_dir}/train_log.json', 'w'), indent=1)


def decode(probs, q=None):
    """argmax (q=None) or q-quantile of the predicted class distribution -> upper bin edge.

    Upper-edge decoding matches the paper's code (intervals[argmax])."""
    p = probs.astype(np.float32)
    if q is None:
        c = p.argmax(-1)
    else:
        cdf = np.cumsum(p / p.sum(-1, keepdims=True), axis=-1)
        c = (cdf < q).sum(-1).clip(0, NCLS - 1)
    return (c + 1) * BIN_W


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         float(sys.argv[5]) if len(sys.argv) > 5 else 90,
         int(sys.argv[6]) if len(sys.argv) > 6 else SEED)
