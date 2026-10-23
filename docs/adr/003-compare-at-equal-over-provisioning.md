# ADR-003: Compare forecasters at equal over-provisioning, with seed-paired differences

- Status: Accepted
- Date: 2026-10-22

## Scenario

Every comparison in this project asks "is forecaster A better than forecaster B at covering spikes?".

## Obstacle

Any forecaster can cover more spikes by asking for more CPU. Comparing two forecasts at their default settings mostly compares how much they over-provision. The paper's method, for example, over-provisions about three times as much as the plain classifier it is compared with (slack 0.59 against 0.19), so its higher recall says little on its own.

Training randomness is a second source of false differences: two runs of the same setup can land in slightly different places.

## Action

- **Curves instead of points.** Each model outputs a probability for each of 40 CPU levels. Reading that distribution at a range of quantiles (0.1 up to 0.995) gives a curve of over-provisioning against coverage, with no retraining.
- **Read every curve at the same budgets.** Each curve is interpolated at fixed slack levels (0.4 to 1.5 times usage), and forecasters are compared at the same level.
- **Pair seeds.** Each model is trained with seeds 0, 1 and 2, and differences are taken seed against seed. In the first single-seed run, 95% intervals came from resampling test traces instead.

Alternatives considered:

- **Compare at each method's default output.** Easy, but confounds skill with how much CPU each method asks for.
- **A single cost number with a price on throttling.** Hides the trade-off behind a price that would have to be invented.

## Result

- At equal over-provisioning, the plain classifier read at the 0.9 quantile matches the paper's recall with about 30% less over-provisioning.
- The horizon finding holds in all three seeds at every budget tested; spike coverage varies between seeds by at most 0.02.
- `results/horizon/matched_slack_summary.csv` and `paired_vs_proposed.json` hold the numbers.

## Troubleshooting

- **Curves that barely overlap.** With the first quantile grid (0.5 to 0.99), only two slack levels (0.8 and 1.0) had values on both the paper's curve and the classifier's curve. Adding quantiles from 0.1 and up to 0.995 widened the shared range to 0.5 to 1.5.
