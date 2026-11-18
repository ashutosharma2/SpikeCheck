# ADR-013: Test the paper's Strategy 2 under the corrected metrics

- Status: Accepted
- Date: 2026-11-14

## Scenario

The paper has two ideas. Strategy 1 changes the target, and everything in this repo so far tests that one. Strategy 2 keeps the plain target, the class of `x[t]`, and changes the loss. Each training label is spread upward over the classes above the true one, so a forecast that is too high costs less than one that is too low.

The authors' loss ([`loss_functions.py`](https://github.com/FerranAgulloLopez/ResourceForecasting), class `CdCE`) is

`alpha * T^2 * KLD( log_softmax(output / T), softmax(altered / T) ) + (1 - alpha) * CE(output, true class)`

with `T = 0.1`. The altered label has four variants. With 40 classes and the settings chosen below, after the softmax they put this much weight on the true class and the next one up:

| Variant | Altered label before the softmax | Weight on the true class | On the next class |
|---|---|---|---|
| Single | 1 on the true class | 0.998 | 0.000 |
| Binary | 1 on the true class and on every class above it | equal across all of them | the same |
| Proportional | 1 on the true class, falling by 0.1 per class above | 0.632 | 0.232 |
| Exponential | 1 on the true class, divided by 1.6 per class above | 0.973 | 0.023 |

The [review](../review.md) made two predictions about this (item C4). The single variant is close to plain cross-entropy, so it should behave like `cbase`, yet the paper reports spike F1 of 0.387 for it against 0.275 for cross-entropy. And an altered label is an indirect way to ask for an upper quantile, which reading `cbase` at a higher quantile gives directly. Neither has been tested here.

## Obstacle

The authors' settings can't be copied. Their Alibaba search covers 48 configurations: three learning rates, two values of `alpha`, and three reduction values each for the proportional and exponential variants. The paper doesn't say which one produced its table.

Training all 48 isn't feasible on this laptop. A model takes 13 to 62 minutes, so three seeds of all 48 would be 144 models and about 80 hours of training, one at a time.

So each variant gets one setting, fixed here before any model is trained. If a variant does badly, a different setting might have done better, and this test can't tell.

The test set isn't fresh. Every published comparison has used it, though these models are new to it.

The inputs are still the rebuilt ones: 8 features against the paper's 18, and one model, TempInc.

## Action

The models:

- Four variants: single, binary, proportional and exponential.
- Three seeds each, 12 models, each with the 60-minute cap of the earlier runs. About 1.6 GB of disk, in `data/models_s2`.
- Same data, same split, same 8 inputs, same 512-step history, same TempInc model and same training settings as every other model here, including the learning rate of 0.01. Only the loss changes ([ADR-004](004-change-one-thing-at-a-time.md)).
- Early stopping reads the Strategy 2 loss on the validation windows, with the same patience of three epochs. The earlier runs count a new best when the loss falls by 0.0001, which is 0.013% of `cbase`'s validation loss of about 0.77. The Strategy 2 loss is about 30 times smaller, so here a new best is a fall of 0.013% of the best so far.
- `alpha = 0.99` and `T = 0.1`.
- Proportional: reduction -4, so the label falls from 1 to 0.1 over ten classes. Exponential: reduction 1.6. Both are the middle of the three values the authors searched.
- The training code for the existing targets is not changed.

The comparison is `cbase`, the same model with plain cross-entropy, matched by seed. Each model is read along its quantile curve at the same overall slack budgets as every earlier comparison ([ADR-003](003-compare-at-equal-over-provisioning.md)). The four variants are also read at quantiles 0.01, 0.02 and 0.05, because binary's curve starts at a slack of 0.77 at quantile 0.1, too close to the 0.8 the rule reads.

The rule reads spike coverage at a 5-minute delay and slack 0.8, the three-seed mean of variant minus `cbase`, and a paired 95% bootstrap over whole test traces. A 5-minute delay is used because Strategy 2 doesn't look ahead, so this is the delay where it has the best chance.

Single against `cbase`:

| Result | Conclusion |
|---|---|
| Interval wholly inside -0.01 to +0.01 | Single matches `cbase` to within 1 point, as the review predicted |
| Interval wholly above 0 or wholly below 0 | Single differs from `cbase`, and the review's prediction is wrong |
| Anything else | Not distinguished from `cbase`, and not shown to match it |

The rows are read from the top, so an interval inside the 1-point margin counts as a match even if it also excludes 0.

Binary, proportional and exponential against `cbase`, each on its own:

| Result | Conclusion |
|---|---|
| Interval wholly above 0 | The altered label adds coverage that a quantile readout of `cbase` doesn't give |
| Interval includes 0 | Not shown to add anything at equal slack |
| Interval wholly below 0 | The altered label is worse than a quantile readout of `cbase` |

Three variants are tested, so one of them passing by chance is more likely than for a single test. A single pass is reported as that and not as a finding about Strategy 2.

A pass against `cbase` is not credited to the upward spread on its own. Single has the same loss without the spread, so each altered variant's difference from single is reported next to its result.

The prediction, written before any model was trained:

- Single is no different from `cbase`.
- None of the three altered variants beats `cbase` at equal slack.
- At a 15-minute delay, all four trail `fwdmax`, because none of them looks ahead.

Reported, not judged:

- Delays of 15 and 30 minutes, and every variant against `fwdmax` and the paper's Strategy 1 target.
- Binary, proportional and exponential against single, which has the same loss without the upward spread.
- Throttled core-hours, and slack budgets 0.6 and 1.0.
- Each variant read the paper's way, at its most likely class: its slack and coverage against `cbase`'s quantile curve at the same slack.
- Spike F1 on the authors' own evaluator, for comparison with the paper's Table 4.
- Coverage per spike type.

The order of work:

1. Write the altered labels and the loss, and test them on synthetic data. The loss has to equal the authors' `CdCE` on the same inputs.
2. Train the 12 models. Training picks its stopping point on validation loss and doesn't score the test set.
3. Dry-run the scorer on the validation traces with seed 0 and add the result below.
4. Commit this ADR and the scripts, and push. That commit freezes the settings, the rule and the scripts.
5. Score the test set once, with three seeds. The run won't start if the output folder exists, if the frozen files differ from that commit, or if the commit hasn't been pushed. The result is reported whatever it is and never used for later tuning.

## Result

The test set has not been scored. The single test run will be added below without changing anything above.

### Validation dry run

Seed 0, 891 validation traces, 28,715 spikes, slack 0.8, coverage at a 5-minute delay. `fwdmax` and `proposed` reproduced the [ADR-010](010-computed-hold.md) dry run on all 36 values checked.

| Variant | Minus `cbase` | 95% interval | Reading under the rule | Minus single | 95% interval |
|---|---|---|---|---|---|
| Single | +0.065 | +0.059 to +0.072 | Differs from `cbase` | | |
| Binary | -0.116 | -0.132 to -0.101 | Worse than `cbase` | -0.181 | -0.198 to -0.166 |
| Proportional | +0.021 | +0.011 to +0.031 | Adds coverage | -0.045 | -0.055 to -0.035 |
| Exponential | +0.068 | +0.061 to +0.074 | Adds coverage | +0.002 | -0.002 to +0.006 |

On this one seed, two of the three predictions are wrong. Single differs from `cbase`, and two altered variants beat it. The third holds: at a 15-minute delay all four trail `fwdmax` by 0.115 to 0.163.

Read at its most likely class, single does match `cbase`: +0.002 coverage (-0.004 to +0.007) against `cbase`'s quantile curve at the same slack, and spike F1 of 0.406 against 0.401 on the authors' evaluator. The difference appears only when the quantile curve is read.

Exponential's gain over `cbase` at a 5-minute delay is the same size as single's, so it can't be credited to the upward spread. At longer delays proportional does pull ahead of single, by 0.036 (0.024 to 0.047) at 15 minutes and 0.076 (0.063 to 0.088) at 30 minutes.

## Troubleshooting

- The first version of the rule for the single variant said the review's prediction holds if the interval includes 0. An interval that includes 0 can also be a wide one, which shows nothing either way. The rule now asks for the whole interval to sit inside 1 point of 0, the size of the margin in [ADR-010](010-computed-hold.md).
- The 12 models were first trained with the fixed 0.0001 of the earlier runs. Against a loss of 0.024 to 0.036 that bar is about 30 times stricter than it is for `cbase`. In 6 of the 12 runs the lowest validation loss came after the saved epoch, and all three binary runs stopped at epoch 7 while the loss was still falling. Under-trained models would have made Strategy 2 look worse than it is. The rule was changed to the relative one above before anything was scored, and all 12 models are trained again from scratch. The first models are kept in `data/models_s2_first` and are not scored on the validation or the test traces.
- The comparison of each altered variant with single was added after the validation dry run, which showed exponential gaining no more than single. It is reported and not judged. The rule's rows against `cbase` did not change.
