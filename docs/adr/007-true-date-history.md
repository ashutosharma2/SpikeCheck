# ADR-007: Date commits by when each file's content actually existed

- Status: Accepted
- Date: 2026-10-23

## Scenario

The work was done over two days, 22 and 23 October 2026, before this repository existed. The history should show that sequence.

## Obstacle

A commit's date is just a label: it can be set to anything. Backdating whole snapshots would imply that intermediate versions existed when they didn't. Some files had also been changed after they were first used: `train.py` gained seeds and a lower-memory data path after the first run.

## Action

Each commit is dated (author and committer date alike) to when its exact content existed, using file timestamps as evidence, and never earlier:

| Date | Commit |
|---|---|
| 22 Oct 00:41 | the review |
| 22 Oct 04:01 | first replication: streaming, preprocessing, metrics, evaluation, results |
| 23 Oct 01:33 | horizon experiment: `train.py`, baselines, multi-seed evaluation, results, write-up |
| 23 Oct 09:26 | 30-minute look-ahead target |
| today | repo setup, docs, wording changes |

Rules that follow from this:

- **Missing files stay missing.** A file whose earlier version no longer exists is left out of the earlier commit. The 22 October commit has no `train.py`, and its message says why.
- **Rewritten files go in later.** Runner scripts that held machine-specific paths are added later, in their portable form.
- **Wording changes are new commits.** Edits to older documents, such as plainer wording, are new commits dated when made.

Alternatives considered:

- **One import commit.** Accurate but tells no story.
- **Several commits dated today.** Accurate and structured, but loses when the work happened.
- **Backdate snapshots to session dates.** Tells the story but implies versions that never existed.

## Result

- The history matches what existed and when.
- The one gap (the first run's `train.py`) is stated in the commit message and in LOG.md, not hidden.

## Troubleshooting

- **The 22 October commit can't run on its own.** Without `train.py` it can evaluate but not train. That's the cost of keeping the dates true, and it's accepted.
