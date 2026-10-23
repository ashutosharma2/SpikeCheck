# ADR-006: Keep the data and the authors' code out of the repo; pin their commit

- Status: Accepted
- Date: 2026-10-23

## Scenario

SpikeCheck should let someone check every number and rerun the work, without the repo becoming huge or republishing what isn't ours.

## Obstacle

- **The data is large.** The raw trace is 29 GB, the prepared inputs about 420 MB, and the trained models with their predictions about 1.2 GB. GitHub rejects files over 100 MB, and history keeps every file ever committed.
- **Redistribution is unclear.** Alibaba gates the trace behind a short survey, and its redistribution terms weren't confirmed.
- **The authors' code is theirs.** It's Apache-2.0, so copying it would be allowed, but it would mix their work with ours.

## Action

- **Tracked:** code, results (CSV, JSON and PNG, about 1.3 MB), docs, and `chosen_containers.csv` (300 KB, to pin the exact sample).
- **Not tracked:** everything in `data/` and `upstream/`, both in `.gitignore`.
- **Data.** `scripts/get_data.sh` downloads and prepares the trace from Alibaba's public bucket.
- **Authors' code.** `scripts/setup.sh` clones the authors' repository and checks out commit `98f0d51`, the version the experiments used.
- **Portable paths.** `SPIKECHECK_DATA` and `SPIKECHECK_UPSTREAM` let both live anywhere.

Alternatives considered:

- **Git LFS for the data.** Workable for the prepared inputs, but GitHub's free LFS quota is small, and it still republishes Alibaba's data.
- **Copy the authors' code in.** Self-contained, but blurs authorship and freezes their code without a reason.

## Result

- The repo stays small, and every claim in the docs points to a tracked result file.
- Rerunning everything takes about 2.5 hours of download plus about 30 minutes of training per model.

## Troubleshooting

- **The pin matters.** The evaluator, the model and the experiments all import from the authors' code, so a later change upstream could silently change results. Pinning the commit in `setup.sh` rules that out.
