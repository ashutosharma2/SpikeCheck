# ADR-001: Score with corrected metrics, and keep the authors' evaluator alongside

- Status: Accepted
- Date: 2026-10-22

## Scenario

The paper judges forecasts with four numbers: spike precision, spike recall, over-provisioning and under-provisioning. To test its claims, the replication needs a way of scoring forecasts that a reader can trust.

## Obstacle

Reading the authors' `evaluation.py` turned up three problems that change the numbers themselves, not just how they're presented:

1. Over-provisioning divides by almost nothing. Usage is z-scored and then shifted so the smallest value in the test set is 1e-10. The metric averages ((forecast − usage) / usage)², so the one point near 1e-10 dominates it. On our data it reports 4 × 10¹⁵ to 7 × 10¹⁶ for every forecaster, including one that just repeats the last value.
2. Precision rewards forecasts that lag. A predicted spike counts as correct if a real one is within 4 steps either side, before or after. Repeating the last value scores precision 1.00, and a rolling maximum scores 0.99.
3. The spike detector fires on noise after a spike. Its reference window shrinks to the steps since the last spike, so one step later the window holds a single value with zero spread, and any rise counts as a new spike.

There are two smaller issues as well. F1 is averaged per trace, and a trace with no spikes scores 0 instead of being left out.

## Action

`scripts/metrics.py` scores forecasts in raw CPU units, without the shift:

- Over-provisioning is total slack divided by total usage across all test traces, and under-provisioning is measured in throttled core-hours. Neither divides by a single value.
- Spikes are detected with a fixed 250-step reference window (the paper's value), a 4-step refractory period and a minimum rise of 2 CPU points.
- Precision only looks forward: a predicted spike counts only if the real one comes at the same time or later.
- Spike counts are pooled across traces, and coverage is measured at several actuation delays (ADR-002).

The authors' evaluator still runs, unchanged, on the same forecasts, and its numbers sit next to the corrected ones in `results/replication/results.csv`.

Alternatives considered:

- Use only the paper's metrics. That keeps the numbers comparable but reports values known to be wrong.
- Use only the corrected metrics. Cleaner, but then a reader couldn't check that the paper's result reproduces, or see where the old numbers mislead.

## Result

- The paper's headline reproduces on its own evaluator: spike F1 goes from 0.41 to 0.68 for the TempInc model (the paper reports 0.27 to 0.64).
- With forward-only precision, repeating the last value drops from 1.00 to 0.26, and the rolling maximum from 0.99 to 0.07.
- Every results table can be read both ways.

## Troubleshooting

- The artefact turned up in the code first and was then confirmed in the data. I spotted the 1e-10 shift while reading `evaluation.py`. The paper's own Table 3 already hinted at it: five of its six Alibaba models report exactly the same over-provisioning, which only makes sense if a single point decides the number. On our data the evaluator gives every forecaster, even one repeating the last value, a value between 10¹⁵ and 10¹⁷.
