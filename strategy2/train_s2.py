"""Train the paper's Temporal Inception classifier with its Strategy 2 loss (docs/adr/013-strategy-2.md).

Strategy 2 keeps the plain target, the class of x[t], and changes the loss. Each label is spread
over the classes above the true one, and the model is trained towards it with the authors' loss
(their loss_functions.CdCE):

  alpha * T^2 * KLD(log_softmax(output / T), softmax(altered / T)) + (1 - alpha) * CE(output, true class)

  single        1 on the true class
  binary        1 on the true class and on every class above it
  proportional  1 on the true class, falling by -REDUCTION / classes per class above
  exponential   1 on the true class, divided by REDUCTION per class above

Everything else is scripts/train.py's cbase run: the same data, split, inputs, model, optimiser,
sampling and seeds. Early stopping reads the same loss on the validation windows and asks for
the same relative gain as train.py (MIN_GAIN). Seed 0 also saves its validation predictions,
for the dry run.

usage: python train_s2.py <data.npz> <repo_root> <variant> <out_dir> [minutes] [seed]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

from train import (BATCH, LAG, LR, NCLS, PATIENCE, SEED, VAL_WINDOWS, WINDOWS_PER_TRACE, gather,   # noqa: E402
                   load_tempinc, make_target)

ALPHA = 0.99
TEMPERATURE = 0.1
REDUCTION = {'proportional': -4.0, 'exponential': 1.6}   # the middle of the three values the authors searched
VARIANTS = ('single', 'binary', 'proportional', 'exponential')
# train.py counts a new best when the validation loss falls by 1e-4, which is 0.013% of cbase's
# loss of about 0.77. This loss is about 30 times smaller, so the same share is asked for here.
MIN_GAIN = 1.3e-4


def weights_above(variant, ncls=NCLS):
    """Label weight on the true class and on each class above it, as the authors build it."""
    if variant == 'single':
        return torch.ones(1)
    if variant == 'binary':
        return torch.ones(ncls)
    if variant == 'proportional':
        return torch.arange(1, 0, REDUCTION[variant] / ncls, dtype=torch.float)
    if variant == 'exponential':
        return REDUCTION[variant] ** -torch.arange(ncls, dtype=torch.float)
    raise ValueError(variant)


def altered_labels(cls, variant, ncls=NCLS):
    """cls: (B,) true classes. Returns (B, ncls): the altered label, zero below the true class."""
    w = weights_above(variant, ncls).to(cls.device)
    offset = torch.arange(ncls, device=cls.device)[None, :] - cls[:, None]
    inside = (offset >= 0) & (offset < len(w))
    return torch.where(inside, w[offset.clamp(0, len(w) - 1)], torch.zeros((), device=cls.device))


def s2_loss(output, cls, altered, reduction='mean'):
    """The authors' CdCE. reduction='mean' is their loss; 'sum' adds up the same loss per sample."""
    kld = F.kl_div(F.log_softmax(output / TEMPERATURE, dim=1), F.softmax(altered / TEMPERATURE, dim=1), reduction='sum')
    ce = F.cross_entropy(output, cls, reduction='sum')
    total = kld * (ALPHA * TEMPERATURE * TEMPERATURE) + ce * (1. - ALPHA)
    return total / len(cls) if reduction == 'mean' else total


def is_new_best(val_loss, best):
    """True when the validation loss beats the best so far by at least MIN_GAIN of it."""
    return val_loss < best * (1. - MIN_GAIN)


def main(data_path, repo_root, variant, out_dir, minutes=90, seed=SEED):
    assert variant in VARIANTS, variant
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # The data handling below is train.py's, for mode 'cbase'.
    z = np.load(data_path, allow_pickle=True)
    split, mu, sd = z['split'], z['mu'], z['sd']
    x_npy = os.path.join(os.path.dirname(data_path), 'X.npy')
    X = np.load(x_npy, mmap_mode='r') if os.path.exists(x_npy) else z['X'].astype(np.float32)
    T, n_feat = X.shape[2], X.shape[1]
    tgt = make_target(X[:, 0].astype(np.float64), 'cbase').astype(np.int16)
    mu = mu.astype(np.float32)[None, :, None]
    sd = sd.astype(np.float32)[None, :, None]

    def subset(name):
        ids = np.where(split == name)[0]
        a = X[ids]
        a -= mu
        a /= sd
        series = torch.from_numpy(a.astype(np.float16)).to(dev)
        del a
        return series, torch.from_numpy(tgt[ids]).to(dev), ids

    tr_s, tr_y, _ = subset('train')
    va_s, va_y, _ = subset('val')
    del X
    offsets = torch.arange(LAG, device=dev)

    def sample(n_tr, k):
        i = rng.integers(0, n_tr, size=n_tr * k)
        t = rng.integers(LAG, T - 5, size=n_tr * k)
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
                tot += s2_loss(model(xb), yb, altered_labels(yb, variant), reduction='sum').item()
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
            loss = s2_loss(model(xb), yb, altered_labels(yb, variant))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = val_loss()
        epoch += 1
        log.append({'epoch': epoch, 'val_loss': vl, 'minutes': (time.time() - t0) / 60})
        print(json.dumps(log[-1]), flush=True)
        if is_new_best(vl, best):
            best, bad = vl, 0
            torch.save(model.state_dict(), f'{out_dir}/model.pt')
        else:
            bad += 1

    del tr_s, tr_y, ti, tt, perm
    if dev.type == 'mps':
        torch.mps.empty_cache()
    model.load_state_dict(torch.load(f'{out_dir}/model.pt'))
    model.eval()
    ts = torch.arange(LAG, T, device=dev)

    def predict(series):
        probs = np.zeros((series.shape[0], T - LAG, NCLS), dtype=np.float16)
        with torch.no_grad():
            for i in range(series.shape[0]):
                ii = torch.full_like(ts, i)
                probs[i] = F.softmax(model(gather(series, ii, ts, offsets)), dim=1).cpu().numpy().astype(np.float16)
        return probs

    if seed == 0:
        np.savez_compressed(f'{out_dir}/val_probs.npz', probs=predict(va_s))
    del va_s, va_y
    if dev.type == 'mps':
        torch.mps.empty_cache()
    X = np.load(x_npy, mmap_mode='r') if os.path.exists(x_npy) else z['X'].astype(np.float32)
    te_s, _, te_ids = subset('test')
    del X, tgt
    np.savez_compressed(f'{out_dir}/test_probs.npz', probs=predict(te_s), test_ids=te_ids)
    json.dump({'variant': variant, 'seed': seed, 'alpha': ALPHA, 'temperature': TEMPERATURE,
               'reduction': REDUCTION.get(variant), 'best_val_loss': best, 'log': log},
              open(f'{out_dir}/train_log.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         float(sys.argv[5]) if len(sys.argv) > 5 else 90,
         int(sys.argv[6]) if len(sys.argv) > 6 else SEED)
