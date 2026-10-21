"""Stream Alibaba 2018 container_usage.tar.gz over HTTP (resumable), keep only the
selected containers, and aggregate them into 5-minute bins in memory.

Nothing but the aggregated arrays is written to disk.

usage: python stream_alibaba.py <container_meta.csv> <out_dir>
"""
import io
import sys
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

URL = 'http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/container_usage.tar.gz'
BIN = 300
NBINS = 2700  # usage timestamps run past 8 days (seen up to ~777,000 s)
SEED = 0


class ResumableHTTP(io.RawIOBase):
    """Read-only stream that reconnects with a Range header after a network error."""

    def __init__(self, url):
        self.url = url
        self.pos = 0
        self.total = None
        self.resp = None
        self.t0 = time.time()
        self.last_log = 0
        self._open()

    def _open(self):
        for attempt in range(1000):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-'})
                self.resp = urllib.request.urlopen(req, timeout=60)
                if self.total is None:
                    self.total = int(self.resp.headers['Content-Range'].split('/')[-1])
                return
            except Exception as e:  # noqa: BLE001
                print(f'[reopen] attempt {attempt} at {self.pos}: {e}', flush=True)
                time.sleep(min(60, 2 ** min(attempt, 6)))
        raise RuntimeError('could not reopen stream')

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            try:
                n = self.resp.readinto(b)
                if n == 0 and self.pos < self.total:
                    raise IOError('premature EOF')
                self.pos += n
                now = time.time()
                if now - self.last_log > 60:
                    self.last_log = now
                    rate = self.pos / (now - self.t0) / 1e6
                    eta = (self.total - self.pos) / max(rate * 1e6, 1) / 60
                    print(f'[progress] {self.pos / 1e9:.2f}/{self.total / 1e9:.2f} GB '
                          f'{rate:.2f} MB/s eta {eta:.0f} min', flush=True)
                return n
            except Exception as e:  # noqa: BLE001
                print(f'[network] {e}; resuming at byte {self.pos}', flush=True)
                try:
                    self.resp.close()
                except Exception:  # noqa: BLE001
                    pass
                self._open()


def select_containers(meta_path):
    """Long-running containers (present at t=0, never stopped), one random per app_du."""
    m = pd.read_csv(meta_path, header=None,
                    names=['cid', 'mid', 'ts', 'app', 'status', 'req', 'lim', 'mem'])
    g = m.groupby('cid').agg(mints=('ts', 'min'), app=('app', 'first'), req=('req', 'first'),
                             lim=('lim', 'first'),
                             allstarted=('status', lambda s: (s == 'started').all()))
    lr = g[(g.mints == 0) & g.allstarted].reset_index()
    chosen = lr.groupby('app').sample(n=1, random_state=SEED).sort_values('cid').reset_index(drop=True)
    return chosen


LOCAL_FILE = None  # set to a local csv path to test parsing without the network


def iter_lines():
    if LOCAL_FILE:
        with open(LOCAL_FILE, 'rb') as f:
            yield from f
        return
    raw = ResumableHTTP(URL)
    buffered = io.BufferedReader(raw, buffer_size=8 * 1024 * 1024)
    with tarfile.open(fileobj=buffered, mode='r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            print(f'member {member.name} {member.size / 1e9:.1f} GB', flush=True)
            yield from tar.extractfile(member)


def main(meta_path, out_dir):
    chosen = select_containers(meta_path)
    chosen.to_csv(f'{out_dir}/chosen_containers.csv', index=False)
    index = {cid.encode(): i for i, cid in enumerate(chosen.cid)}
    n = len(index)
    print(f'selected {n} containers', flush=True)

    # features: cpu max, cpu sum, mem, cpi, mpki, net_in, net_out, disk (sums) + counts
    cpu_max = np.full((n, NBINS), -1.0, dtype=np.float32)
    sums = np.zeros((7, n, NBINS), dtype=np.float32)
    cnt_cpu = np.zeros((n, NBINS), dtype=np.int16)
    cnt_oth = np.zeros((6, n, NBINS), dtype=np.int16)  # per-feature counts (fields can be empty)

    kept = seen = 0
    for line in iter_lines():
        seen += 1
        cid, _, rest = line.partition(b',')
        i = index.get(cid)
        if i is None:
            continue
        p = rest.rstrip(b'\n').split(b',')
        # p: machine_id, ts, cpu, mem, cpi, mem_gps, mpki, net_in, net_out, disk
        try:
            b = int(float(p[1])) // BIN
            cpu = float(p[2])
        except (ValueError, IndexError):
            continue
        if b >= NBINS or b < 0 or cpu < 0 or cpu > 100:
            continue
        kept += 1
        if cpu > cpu_max[i, b]:
            cpu_max[i, b] = cpu
        sums[0, i, b] += cpu
        cnt_cpu[i, b] += 1
        # mem, cpi, mpki, net_in, net_out, disk; empty or invalid fields are skipped
        for k, col in enumerate((3, 4, 6, 7, 8, 9)):
            field = p[col] if col < len(p) else b''
            if not field:
                continue
            try:
                v = float(field)
            except ValueError:
                continue
            if k == 5 and (v < 0 or v > 100):
                continue
            sums[k + 1, i, b] += v
            cnt_oth[k, i, b] += 1
        if seen % 50_000_000 == 0:
            print(f'[lines] seen {seen / 1e6:.0f}M kept {kept / 1e6:.1f}M', flush=True)

    np.savez_compressed(f'{out_dir}/alibaba_5min.npz', cpu_max=cpu_max, sums=sums,
                        cnt_cpu=cnt_cpu, cnt_oth=cnt_oth, cids=chosen.cid.values.astype(str))
    print(f'done: seen {seen} lines, kept {kept}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        LOCAL_FILE = sys.argv[3]
    main(sys.argv[1], sys.argv[2])
