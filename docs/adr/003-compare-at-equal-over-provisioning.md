# ADR-003: Compare forecasters at equal over-provisioning, with differences paired by seed

- Status: Accepted
- Date: 2026-10-22

## Scenario

Every comparison in this project asks whether forecaster A covers spikes better than forecaster B.

## Obstacle

Any forecaster can cover more spikes by asking for more CPU, so comparing two forecasts at their default settings mostly compares how much they over-provision. The paper's method, for example, over-provisions about three times as much as the plain classifier it's compared with (slack 0.59 against 0.19), so its higher recall says little on its own.

Training randomness causes false differences too: two runs of the same setup can land in slightly different places.

## Action

Each model outputs a probability for each of 40 CPU levels. Reading that distribution at a range of quantiles, from 0.1 up to 0.995, turns one model into a curve of over-provisioning against coverage, without retraining. Each curve is then interpolated at fixed slack levels (0.4 to 1.5 times usage), and forecasters are compared at the same level.

Each model is trained with seeds 0, 1 and 2, and differences are taken seed against seed. In the first run, which used a single seed, the 95% intervals came from resampling test traces instead.

Alternatives considered:

- Compare each method's default output. Easy, but it mixes up skill with how much CPU a method asks for.
- Use a single cost number with a price on throttling. That hides the trade-off behind a price someone would have to invent.

## Result

- At equal over-provisioning, the plain classifier read at an upper quantile matches the paper's recall with 14 to 29% less over-provisioning across three seeds.
- The horizon finding holds in all three seeds at every budget tested, and spike coverage varies between seeds by at most 0.02.
- The numbers are in `results/horizon/matched_slack_summary.csv` and `paired_vs_proposed.json`.

## Troubleshooting

- At first the curves barely overlapped. With the first quantile grid (0.5 to 0.99), only two slack levels (0.8 and 1.0) had values on both the paper's curve and the classifier's. Adding quantiles from 0.1 and up to 0.995 widened the shared range to 0.5 to 1.5.
