# ADR-008: Publish only results the committed code can regenerate

- Status: Accepted
- Date: 2026-10-28

## Scenario

SpikeCheck criticises the paper for results that rest on a single seed and are hard to reproduce. Its own results should hold up to the same test: anyone who clones the repo and runs its scripts should be able to regenerate the numbers it publishes.

## Obstacle

On 27 October macOS cleared the temporary folder that held the working data, the first replication's models and three seed-0 prediction files. Retraining showed that only one of the lost runs could be regenerated as published. fwdmax seed 0 came back identical. cbase, proposed and proposed_full had been trained with an earlier float32 `train.py` that no longer exists ([ADR-007](007-true-date-history.md)), so nobody could regenerate their published numbers, including me.

The old numbers weren't necessarily wrong. They just no longer matched any code in the repo.

## Action

Published results must come from the committed code. When a result can't be regenerated from the current scripts, it is regenerated and replaced, and:

- the old numbers stay in the Git history;
- the docs say why the numbers changed and what moved;
- every result and claim that depends on them is updated, not just the tables.

The new results aren't treated as more accurate than the old ones, only as traceable to the current implementation.

Alternatives considered:

- Keep the old results and add a note about the rebuild. Cheaper, but it leaves published numbers that no contributor can regenerate, produced by a script that's gone.

## Result

- The rule was first applied in the rebuild of 28 October. All three results folders were regenerated, and every dependent claim was checked against the new files ([replication write-up](../replication.md#rebuild-every-result-regenerated-from-the-committed-scripts-28-october-2026)).
- Reviewers can rerun the scripts and check the findings themselves.

## Troubleshooting

- Regenerating isn't the same as reproducing bit for bit. fwdmax seed 0 came back identical on the same Mac with the same script, but a different GPU or library version would give close numbers, not identical ones. That's one more reason the claims rest on three-seed ranges rather than on a single run.
