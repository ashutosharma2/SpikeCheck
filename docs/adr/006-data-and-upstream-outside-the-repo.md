# ADR-006: Keep the data and the authors' code out of the repo; pin their commit

- Status: Accepted
- Date: 2026-10-23

## Scenario

SpikeCheck should let someone check every number and rerun the work, without the repo becoming huge or republishing things that aren't ours.

## Obstacle

The data is large. The raw trace is 29 GB, the prepared inputs about 420 MB, and the trained models with their predictions about 1.2 GB. GitHub rejects files over 100 MB, and a repo's history keeps every file ever committed.

Redistribution is unclear: Alibaba asks people to fill in a short survey before downloading the trace, and I couldn't confirm its redistribution terms.

And the authors' code is theirs. It's Apache-2.0, so copying it in would be allowed, but it would mix their work with ours.

## Action

The repo tracks the code, the results (CSV, JSON and PNG, about 1.3 MB), the docs, and `chosen_containers.csv` (300 KB, which pins the exact sample). Everything in `data/` and `upstream/` stays out, and both are in `.gitignore`.

`scripts/get_data.sh` downloads and prepares the trace from Alibaba's public bucket. `scripts/setup.sh` clones the authors' repository and checks out commit `98f0d51`, the version the experiments used. `SPIKECHECK_DATA` and `SPIKECHECK_UPSTREAM` let both live anywhere.

Alternatives considered:

- Git LFS for the data. It would work for the prepared inputs, but GitHub's free LFS quota is small, and it still republishes Alibaba's data.
- Copy the authors' code in. Self-contained, but it blurs who wrote what and freezes their code for no reason.

## Result

- The repo stays small, and every claim in the docs points to a tracked result file.
- Rerunning everything takes about 2.5 hours of download plus about 30 minutes of training per model.

## Troubleshooting

- The pin matters. The evaluator, the model and the experiments all import from the authors' code, so a later change there could quietly change our results. Pinning the commit in `setup.sh` rules that out.
