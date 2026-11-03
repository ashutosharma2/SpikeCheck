"""Run saved models on the validation traces, so exploration leaves the test traces untouched.

Inputs are built as in train.py. Each model is first checked against its saved test_probs on
three test traces (probabilities only, nothing is scored). Output: data/novelty_val/<run>.npy,
float16, shape (val traces, T - LAG, classes).

usage: python val_probs.py <run> [<run> ...]    e.g. fwdmax_s0 proposed_s0
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from train import LAG, NCLS, gather, load_tempinc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = REPO + '/data'
OUT = DATA + '/novelty_val'
os.makedirs(OUT, exist_ok=True)

z = np.load(DATA + '/data.npz', allow_pickle=True)
split = z['split']
mu = z['mu'].astype(np.float32)[None, :, None]
sd = z['sd'].astype(np.float32)[None, :, None]
X = np.load(DATA + '/X.npy', mmap_mode='r')
T, n_feat = X.shape[2], X.shape[1]
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
TempInc = load_tempinc(REPO + '/upstream/ResourceForecasting')
offsets = torch.arange(LAG, device=dev)
ts = torch.arange(LAG, T, device=dev)


def series(name):
    a = np.array(X[np.where(split == name)[0]])
    a -= mu
    a /= sd
    return torch.from_numpy(a.astype(np.float16)).to(dev)


def infer(model, s, i):
    ii = torch.full_like(ts, i)
    return F.softmax(model(gather(s, ii, ts, offsets)), dim=1).cpu().numpy().astype(np.float16)


val = series('val')
test = series('test')
for run in sys.argv[1:]:
    path = f'{DATA}/models_seeds/{run}'
    model = TempInc(LAG, NCLS, n_feat, str(dev)).to(dev)
    model.load_state_dict(torch.load(f'{path}/model.pt', map_location=dev))
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        saved = np.load(f'{path}/test_probs.npz')['probs']
        worst = max(float(np.abs(infer(model, test, i).astype(np.float32) - saved[i].astype(np.float32)).max())
                    for i in range(3))
        print(run, 'max difference from saved test_probs on 3 traces:', worst, flush=True)
        assert worst < 0.01, 'model does not reproduce its saved predictions'
        del saved
        probs = np.zeros((val.shape[0], T - LAG, NCLS), dtype=np.float16)
        for i in range(val.shape[0]):
            probs[i] = infer(model, val, i)
            if i == 49:
                print(run, f'{(time.time() - t0) / 50:.2f} s per trace', flush=True)
    np.save(f'{OUT}/{run}.npy', probs)
    print(run, f'done in {(time.time() - t0) / 60:.1f} min', flush=True)
    del model, probs
    if dev.type == 'mps':
        torch.mps.empty_cache()
