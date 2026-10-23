# ADR-005: Rebuild the preprocessing, keeping the authors' choices where they're known

- Status: Accepted
- Date: 2026-10-22

## Scenario

The replication needs the same kind of data the paper used: long-running Alibaba 2018 containers, CPU at 5-minute steps, split into training, validation and test traces.

## Obstacle

The authors' repository reads an already-processed dataset; the scripts that made it aren't public. The paper describes the steps only in words (long-running containers, one per workload, drop traces with many gaps or little variation), so every threshold had to be chosen here.

## Action

`scripts/stream_alibaba.py` and `scripts/prep.py` rebuild it:

- **Containers.** Long-running means present at the start of the trace and never stopped. One random container per application (seed 0) gives 9,505.
- **Window.** 8 days of 5-minute steps, using the maximum CPU in each step as the target, as the paper does.
- **Filters.** Drop traces with more than 10% empty steps (8,833 left), then traces whose maximum is less than 5 CPU points above their average (5,946 left; the paper had 5,168). Remaining gaps are filled from the previous value.
- **Split.** Random by trace, 70/15/15 (4,162 / 891 / 893), with scaling fitted on the training traces only.
- **Kept from the authors:** 40 equal CPU levels, decoding a predicted level to its upper edge, and a 512-step lag.

`results/replication/chosen_containers.csv` records exactly which containers were sampled.

Alternatives considered:

- **Ask the authors for their dataset.** Would match exactly, but depends on a reply and still wouldn't explain the choices.
- **Use a different, already-processed dataset.** Faster, but no longer a replication of this paper.

## Result

- The rebuilt data reproduces the paper's headline result on the authors' own evaluator (ADR-001), which suggests it's close enough for the comparisons made here.
- Every threshold is in one script, so a reader can change one and rerun.

## Troubleshooting

- **Zero traces on the first try.** The first run kept no traces at all: every container had over 15% empty steps. The usage timestamps start at about 86,400 seconds, not 0, so the first day of the window was always empty. Moving the window to steps 288 to 2,591 fixed it.
