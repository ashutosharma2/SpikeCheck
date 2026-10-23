# ADR-005: Rebuild the preprocessing, keeping the authors' choices where they're known

- Status: Accepted
- Date: 2026-10-22

## Scenario

The replication needs the same kind of data the paper used: long-running Alibaba 2018 containers, CPU in 5-minute steps, split into training, validation and test traces.

## Obstacle

The authors' repository reads a dataset that was already processed, and the scripts that made it aren't public. The paper only describes the steps in words (long-running containers, one per workload, drop traces with many gaps or little variation), so every threshold had to be chosen here.

## Action

`scripts/stream_alibaba.py` and `scripts/prep.py` rebuild it. A container counts as long-running if it's present at the start of the trace and never stopped, and one random container per application (seed 0) gives 9,505 of them. The window is 8 days of 5-minute steps, with the maximum CPU in each step as the target, as in the paper.

Traces with more than 10% empty steps are dropped (8,833 left), then traces whose maximum is less than 5 CPU points above their average (5,946 left; the paper had 5,168). Remaining gaps are filled from the previous value. The split is random by trace, 70/15/15 (4,162 / 891 / 893), and the scaling is fitted on the training traces only.

Three choices come from the authors: 40 equal CPU levels, decoding a predicted level to its upper edge, and a 512-step lag. `results/replication/chosen_containers.csv` records exactly which containers were sampled.

Alternatives considered:

- Ask the authors for their dataset. It would match exactly, but it depends on a reply and still wouldn't explain the choices.
- Use a different dataset that's already processed. Faster, but no longer a replication of this paper.

## Result

- The rebuilt data reproduces the paper's headline result on the authors' own evaluator (ADR-001), which suggests it's close enough for the comparisons made here.
- Every threshold sits in one script, so a reader can change one and rerun.

## Troubleshooting

- The first try kept zero traces: every container had more than 15% empty steps. The usage timestamps start at about 86,400 seconds, not at 0, so the first day of the window was always empty. Moving the window to steps 288 to 2,591 fixed it.
