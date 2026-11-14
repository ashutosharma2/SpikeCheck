# ADR-011: Check for headroom before building a per-trace policy selector

- Status: Accepted
- Date: 2026-11-13

## Scenario

The spike-type results ([ADR-009](009-spike-types.md)) show that spike types prefer different policies. The look-ahead leads on seasonal and isolated spikes, and a hold leads on follow-on spikes. That suggests a next step: a selector that picks one provisioning policy per workload from a fixed library. The selector could be a hand-written rule, a trained classifier or a large language model reading a summary of the workload.

A selector only helps if two things are true. Different traces have to prefer different policies, and that preference has to be visible before the decision is made. Neither had been measured. In the committed per-type table, `fwdmax6` is the best single policy at every delay and budget I report, and follow-on spikes are 1,533 of the 27,980 test spikes.

So before building any selector, I ran two cheap checks on the validation traces. Each had its rule and a prediction written down before the run.

## Obstacle

The test set can't judge this idea. All 893 test traces have been used for every published comparison, and the per-type results on them are what suggested a selector. Both checks therefore use the 891 validation traces only.

Validation predictions exist for seed 0 only. The trained `cbase` and `fwdmax` models for the other seeds weren't all kept, so neither check can show seed spread.

Matching the budget is harder for a mix. Until now each policy got one quantile for all traces. A mix of policies spends a different total, so it needs its own matching rule.

Hindsight flatters a selector. The best policy for one trace is decided by about 32 spikes, so picking winners after the fact also picks up noise.

## Action

The library has six policies, all from saved seed-0 predictions: `cbase`, `proposed`, `fwdmax`, `fwdmax6`, and the encoded computed hold of [ADR-010](010-computed-hold.md) on `fwdmax` and on `fwdmax6`.

An assignment gives every trace one policy for the whole scored period. All policies are read at the same quantile, and the mix is scored along its quantile curve and read at the same overall slack budgets as a single policy. A fixed policy is the assignment that gives every trace the same policy, so both are scored the same way. Results are reported at delays of 5, 15 and 30 minutes and slack 0.6, 0.8 and 1.0.

Check 1, headroom (`selection/oracle_headroom.py`), has two numbers per cell, each against the best fixed policy:

- Ceiling: the best assignment found with hindsight over the whole scored period. No real selector can know it.
- Follow: the assignment hindsight picks on the first half of the scored period (steps 520 to 1,532), scored on the second half (steps 1,533 to 2,303) against the fixed policy that was best on the first half.

The hindsight assignment is a search. For each trace it picks the policy with the most covered spikes minus a price on slack, at every quantile and price, then swaps one trace at a time until no swap helps.

The rule for check 1, written before the run: read follow at a 15-minute delay and slack 0.8. Continue with per-trace selection only if the lower bound of its 95% interval is above +0.01, the size of the follow-on margin in ADR-010. A lower bound above 0 but not above +0.01 means a gain too small on its own. A lower bound at or below 0 means nothing carries over.

The prediction for check 1, written before the run: the ceiling would be about +0.02 to +0.06, because hindsight also picks up noise, and follow would be below +0.01.

Check 2, a trained selector (`selection/trained_selector.py`):

- One decision per trace at step 520, the first scored step, from the steps before it only.
- Twelve features, fixed in advance: mean and standard deviation of CPU, peak over mean, mean step change, autocorrelation at 5 minutes and at 24 hours, the number of early spikes and the share that follow another within 6 steps, mean memory, mean network use, the CPU request and the limit over the request.
- A decision tree of depth 3 with at least 20 traces per leaf. Its labels are the policies hindsight picks for the training traces.
- Five folds over the validation traces. Each fold gets its policies from a tree trained on the other four, so no trace is scored by a tree that saw it. The comparison is the fixed policy that was best on those same four folds.

The rule for check 2, written before the run: read the gain at a 15-minute delay and slack 0.8, and continue towards a learned or language-model selector only if the lower bound of its 95% interval is above 0.

The prediction for check 2, written before the run: a gain between -0.005 and +0.005 with an interval that includes zero, because the hindsight labels looked mostly like noise.

Intervals are a 95% bootstrap over whole validation traces with the assignments frozen. They cover trace sampling, not the search or the training of the trees.

## Result

Both checks ran once, on 13 November 2026. Validation traces, seed 0, 891 traces and 28,715 spikes. The files are in `results/oracle_headroom/` and `results/trained_selector/`.

At a 15-minute delay and slack 0.8, against the best fixed policy (`fwdmax6`, coverage 0.770):

| | Coverage gain | Throttled core-hours |
|---|---|---|
| Ceiling, hindsight on the whole period | +0.0271 [+0.0214, +0.0296] | 911 against 827 |
| Follow, first half applied to the second | -0.0103 [-0.0158, -0.0043] | 381 against 309 |
| Trained selector, cross-fitted | -0.0181 [-0.0277, -0.0106] | 995 against 827 |

Across the nine delay and budget cells:

| | Lowest gain | Highest gain | Intervals above 0 | Intervals wholly below 0 |
|---|---|---|---|---|
| Ceiling | +0.0150 | +0.0424 | 9 | 0 |
| Follow | -0.0103 | +0.0013 | 0 | 1 |
| Trained selector | -0.0187 | +0.0013 | 0 | 5 |

`fwdmax6` was the best fixed policy in every cell and every fold.

By the rule for check 1, nothing carries over: the lower bound of follow is below 0. By the rule for check 2, the selector isn't shown to beat the fixed policy. I dropped per-trace policy selection with this library. The next task stays the trace-driven autoscaler simulation.

The prediction for check 1 held, except that the ceiling fell just under +0.02 in two of the 30-minute cells. The prediction for check 2 was wrong at the cell the rule reads: the tree did worse than predicted.

What this gives:

- A measured reason not to build a selector on this library, before any selector was built.
- A way to score a mix of policies at an equal budget, tested on synthetic traces.

What it doesn't show:

- That no selector could work. One naive rule and one small tree were tested, with one set of features.
- Anything about a wider library or a different decision unit, such as deciding again each day.
- Seed spread, a later time window or another cluster.
- The true hindsight optimum. The ceiling is a search result and could be higher.

## Troubleshooting

- The ceiling is real but doesn't last. With hindsight the best assignment moves 64% of traces off `fwdmax6`. The assignment picked on the first half moves 69% of them and then loses coverage on the second half.
- The tree fitted noise. A tree trained on all 891 traces matches 46% of its own training labels with six policies to choose from. At the cell the rule reads it sent 467 traces to `fwdmax` and 81 to `cbase`, both of which cover fewer spikes than `fwdmax6` when used alone.
- Coverage isn't the whole cost. The hindsight assignment was picked for coverage, and it throttles more than the fixed policy.
- `scripts/val_probs.py cbase_s0` wrote the one missing set of validation predictions. It scores nothing.
- Before either check scores a mix, the fixed policies have to reproduce the validation dry run of ADR-010. They do.
- `oracle_headroom.py` was changed after its run so that both scripts share one loader. A rerun gave the same files byte for byte.
