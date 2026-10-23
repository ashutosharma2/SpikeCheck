# ADR-001: Score with corrected metrics, and keep the authors' evaluator alongside

- Status: Accepted
- Date: 2026-10-22

## Scenario

The paper judges forecasts with four numbers: spike precision, spike recall, over-provisioning and under-provisioning. To test its claims, the replication needs a way to score forecasts that a reader can trust.

## Obstacle

Reading the authors' `evaluation.py` showed three problems that change the numbers, not just their presentation:

1. **Over-provisioning divides by almost nothing.** Usage is z-scored and then shifted so the smallest value in the test set is 1e-10. The metric averages ((forecast − usage) / usage)², so the one point near 1e-10 dominates. On our data it reports 4 × 10¹⁵ to 7 × 10¹⁶ for every forecaster, including simply repeating the last value.
2. **Precision rewards forecasts that lag.** A predicted spike counts as correct if a real one is within ±4 steps, before or after. Repeating the last value scores precision 1.00, and a rolling maximum scores 0.99.
3. **The spike detector fires on noise after a spike.** Its reference window shrinks to the steps since the last spike, so one step later the window has a single value, its spread is zero, and any rise counts as a new spike.

Two smaller issues: F1 is averaged per trace, and a trace with no spikes scores 0 rather than being left out.

## Action

`scripts/metrics.py` scores forecasts in raw CPU units, with no shift:

- **Slack and throttling.** Over-provisioning is total slack divided by total usage across all test traces; under-provisioning is throttled core-hours. Neither divides by a single value.
- **Spike detection.** A fixed 250-step reference window (the paper's value), a 4-step refractory period, and a minimum rise of 2 CPU points.
- **Forward-only precision.** A predicted spike only counts if the real one comes at or after it.
- **Pooled counts and delays.** Spike counts are pooled across traces, and coverage is measured at actuation delays (ADR-002).

The authors' evaluator is still run, unchanged, on the same forecasts, and its numbers sit next to the corrected ones in `results/replication/results.csv`.

Alternatives considered:

- **Use only the paper's metrics.** Keeps comparability but reports numbers known to be wrong.
- **Use only the corrected metrics.** Cleaner, but a reader couldn't check that the paper's result reproduces or see where the old numbers mislead.

## Result

- The paper's headline reproduces on its own evaluator: spike F1 0.42 → 0.65 for the TempInc model (the paper reports 0.27 → 0.64).
- With forward-only precision, repeating the last value drops from 1.00 to 0.26, and the rolling maximum from 0.99 to 0.07.
- Every result table can be read both ways.

## Troubleshooting

- **The artefact was found in the code, then confirmed in the data.** The 1e-10 shift was spotted while reading `evaluation.py`. The paper's own Table 3 already hinted at it: five of its six Alibaba models report exactly the same over-provisioning, which only makes sense if a single point decides the number. On our data, the evaluator gives every forecaster, even repeating the last value, a value between 10¹⁵ and 10¹⁷.
