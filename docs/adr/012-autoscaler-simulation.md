# ADR-012: Test the computed hold in a simulated autoscaler

- Status: Accepted
- Date: 2026-11-14

## Scenario

Every comparison so far is a replay with one rule: the allocation in force at step `t` is the forecast issued `d` steps earlier ([ADR-002](002-score-under-actuation-delay.md)). In that replay the computed hold throttles less than the learned hold at slack 0.8, by 45.4 core-hours (5.4%) at a 5-minute delay and 55.1 (5.7%) at 15 ([ADR-010](010-computed-hold.md)).

A real autoscaler does more than copy the forecast. It waits before scaling down, and it ignores small changes. The first of those is itself a hold, computed inside the controller from its recent recommendations. So the replay leaves a question open: does the computed hold's throttling gain over the learned hold survive when the controller already holds allocations for a while?

This ADR fixes a small simulated autoscaler, the comparison and the rule before any simulator run.

## Obstacle

The test can't be blind. The same three methods have already been compared on the test set in the replay, so the direction of the result is known in advance. The simulator adds controller behaviour on top of numbers I have already seen.

The controller settings are borrowed. The scale-down window and the tolerance come from the defaults of Kubernetes' Horizontal Pod Autoscaler ([documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)): a 300-second window that uses the highest recommendation in it, no window for scaling up, and a 10% tolerance. That autoscaler changes the number of replicas. These traces are CPU per container, which is closer to vertical scaling. The settings are reference values, not a model of one product.

A replay still can't show how a workload responds. The traces record what containers used under the allocation they had. CPU never exceeds 100% of the request in the data, so demand above the request isn't visible. There's no latency signal and no customer data.

Validation predictions exist for seed 0 only, so the dry run can't show seed spread.

## Action

The simulated autoscaler works per trace, in 5-minute steps, on a forecast `y` read at one quantile:

1. Recommendation in force: `r[t] = y[t - d + 1]`, the forecast issued `d` steps earlier, for scaling up and down alike. This is the published replay.
2. Scale-down window of `w` steps: `s[t]` is the highest of `r[t - w .. t]`. Increases pass at once and decreases wait.
3. Tolerance `tol`: the allocation `a[t]` stays at `a[t - 1]` when `s[t]` is within `tol` of it, and moves to `s[t]` otherwise.

Two settings are reported:

- Replay: `w = 0` and `tol = 0`. The simulator must reproduce the published replay for all three methods before any other setting is scored. That check uses the published budget convention, which measures slack on the forecast before the delay is applied.
- Controller: `w = 1` step (5 minutes) and `tol = 0.10`. This is the setting the rule reads.

The methods, as in ADR-010:

- No hold: `fwdmax`.
- Computed hold: `fwdmax` with the encoded hold.
- Learned hold: the paper's target, `proposed`.

At 30 minutes, reported separately: `fwdmax6`, `fwdmax6` with the encoded hold, and `proposed`.

Budget: each method is read along its quantile curve at the same overall slack budgets as every earlier comparison ([ADR-003](003-compare-at-equal-over-provisioning.md)), with slack measured on the simulated allocation. No setting is tuned on results.

Measures, all on the simulated allocation: throttled core-hours, allocated core-hours, spike coverage at 80%, and the number of steps where the allocation changes.

The rule reads the controller setting, slack 0.8, the three-seed mean of throttled core-hours, computed minus learned, and a paired 95% bootstrap over whole test traces:

| Result | Conclusion |
|---|---|
| Interval wholly below 0 at 5 and at 15 minutes | The replay's throttling gain carries over to the simulated autoscaler |
| Wholly below 0 at one delay only | The conclusion is reported per delay |
| Interval includes 0 at a delay | Not shown to carry over at that delay |
| Interval wholly above 0 at a delay | The result reverses in the simulated autoscaler at that delay |

The prediction, written before any simulator run:

- The order stays the same as in the replay at both delays: no hold throttles least, then the computed hold, then the learned hold.
- The computed hold's gain over the learned hold is smaller, as a share of the learned hold's throttling, than the replay's 5.4% and 5.7%, because the controller's own window does part of what a hold does.

The prediction is reported as held or not. It isn't part of the rule.

Reported, not judged:

- No hold against each hold, on every measure.
- Spike coverage overall and on follow-on spikes.
- The number of allocation changes.
- Slack budgets 0.6 and 1.0, and the 30-minute results.
- A backlog version, where CPU that wasn't served is added to the next step's demand. It shows how long unserved work would queue under that assumption. The traces can't confirm the assumption.

The order of work:

1. Write the simulator and test it on synthetic traces, including that the replay setting equals `metrics.py`.
2. Dry-run it on the validation traces with seed 0 and add the result below.
3. Commit this ADR and the scripts, and push. That commit freezes the settings, the rule and the scripts.
4. Run the test set once, with three seeds. The run won't start if the output folder exists, if the frozen files differ from that commit, or if the commit hasn't been pushed. The result is reported whatever it is and never used for later tuning.

## Result

Dry run on the validation traces, seed 0, slack 0.8. Before it scored the controller setting, the replay setting reproduced the validation dry run of ADR-010.

Throttled core-hours, computed minus learned:

| Delay | Replay setting | Controller setting |
|---|---|---|
| 5 min | -34.1 [-53.2, -19.0], -4.1% | -40.2 [-59.8, -20.9], -4.6% |
| 15 min | -57.8 [-85.3, -35.9], -6.0% | -60.7 [-82.6, -39.5], -6.1% |

Under the rule, the gain carries over at both delays on validation. This isn't the test result, which will be added below without changing anything above.

The three methods in the controller setting:

| Delay | Method | Throttled core-hours | All spikes | Follow-on | Allocation changes |
|---|---|---|---|---|---|
| 5 min | No hold | 778 | 0.755 | 0.865 | 283,865 |
| 5 min | Computed hold | 823 | 0.720 | 0.917 | 221,959 |
| 5 min | Learned hold | 863 | 0.711 | 0.925 | 235,780 |
| 15 min | No hold | 851 | 0.755 | 0.857 | 283,420 |
| 15 min | Computed hold | 939 | 0.727 | 0.911 | 221,145 |
| 15 min | Learned hold | 1,000 | 0.719 | 0.912 | 234,844 |

On the prediction, on validation:

- The order holds at both delays: no hold throttles least, then the computed hold, then the learned hold.
- The second part doesn't hold. The computed hold's gain is not smaller as a share in the controller setting: 4.6% and 6.1%, against 4.1% and 6.0% in the replay setting.

Reported, not judged, in the controller setting:

- The window and tolerance about halve the number of allocation changes. No hold makes 283,865 at 5 minutes, against 546,562 in the replay setting.
- No hold throttles less than either hold. The computed hold adds 44.7 core-hours [26.2, 66.9] at 5 minutes and 88.2 [62.7, 121.6] at 15.
- Follow-on coverage, computed minus learned, is -0.0078 [-0.0130, -0.0029] at 5 minutes and -0.0007 [-0.0063, +0.0051] at 15.
- At slack 0.6 the throttling gain isn't shown at either delay: -17.9 [-62.7, +26.8] and -11.4 [-75.4, +37.0]. At slack 1.0 it is: -22.3 [-37.6, -6.7] and -39.7 [-54.7, -24.1].
- At 30 minutes, `fwdmax6` with the encoded hold throttles 240.8 core-hours less than the paper's target [-264.7, -223.8], 19.2%, mostly from the longer look-ahead. Against plain `fwdmax6` the hold adds 97.8 [65.7, 132.2].
- The backlog version doesn't separate the two holds. Queued core-hours, computed minus learned, are +477 [-722, +2,813] at 5 minutes and +380 [-841, +2,667] at 15.

## Troubleshooting

- The budget is measured in two ways, on purpose. The published replay matches methods on the slack of the forecast before any delay. The simulator matches them on the slack of the allocation it produces, because the window adds slack that has to count. So the replay column above differs slightly from ADR-010's dry run at 15 minutes (-57.8 against -56.9), and the reproduction check uses the published convention.

- The first plan let scale-downs take effect at once while scale-ups took `d` steps. That changes two things compared with the replay at the same time, the delay and the window. The delay is now the same in both directions, as in the published replay, and the window is the only thing that slows a decrease.

## Test result

Added 2026-11-14 after the single test run at commit `9cb6071`. Nothing above changed. Before any controller setting was scored, the replay setting matched `results/computed_hold/by_method.csv`. Results are in `results/autoscaler/`.

Test set, three seeds, slack 0.8, throttled core-hours, computed minus learned:

| Delay | Replay setting | Controller setting | Rule |
|---|---|---|---|
| 5 min | -45.4 [-58.4, -33.2], -5.4% | -49.3 [-59.9, -36.1], -5.8% | Interval wholly below 0, carries over |
| 15 min | -55.3 [-72.5, -42.0], -5.7% | -61.6 [-79.0, -45.9], -6.3% | Interval wholly below 0, carries over |

By the rule above, the replay's throttling gain carries over to the simulated autoscaler at both delays. All three seeds agree in sign at both delays.

The three methods in the controller setting, mean of three seeds:

| Delay | Method | Throttled core-hours | All spikes | Follow-on | Allocation changes |
|---|---|---|---|---|---|
| 5 min | No hold | 774 | 0.758 | 0.850 | 300,578 |
| 5 min | Computed hold | 799 | 0.725 | 0.926 | 231,226 |
| 5 min | Learned hold | 849 | 0.710 | 0.931 | 235,684 |
| 15 min | No hold | 859 | 0.756 | 0.847 | 300,089 |
| 15 min | Computed hold | 923 | 0.731 | 0.915 | 230,337 |
| 15 min | Learned hold | 985 | 0.719 | 0.920 | 234,786 |

On the prediction:

- The order holds at both delays: no hold throttles least, then the computed hold, then the learned hold.
- The second part was wrong. The computed hold's gain as a share is not smaller in the controller setting: 5.8% and 6.3%, against 5.4% and 5.7% in the replay.

Three limits on that reading:

- The test wasn't blind. The same three methods had already been compared on these traces in the replay, and the controller adds a 5-minute window and a 10% tolerance on top of it. A result close to the replay's was likely.
- This is still a replay of recorded usage. It can't show how a workload responds to throttling, it has no latency signal, and CPU above the request isn't visible in the data.
- The intervals cover trace sampling only, on one trace and one time window. They aren't significance tests.

Reported, not judged, in the controller setting:

- No hold still throttles least. The computed hold adds 25.1 core-hours [7.6, 52.6] at 5 minutes and 64.0 [40.0, 96.7] at 15. The learned hold adds 74.4 [47.3, 104.1] and 125.6 [90.9, 166.7].
- The window and tolerance about halve the number of allocation changes. No hold makes 300,578 at 5 minutes, against 557,949 in the replay setting.
- Overall coverage, computed minus learned, is +0.0144 [+0.0109, +0.0173] at 5 minutes and +0.0117 [+0.0088, +0.0145] at 15. Follow-on coverage is -0.0057 [-0.0112, -0.0005] and -0.0051 [-0.0106, +0.0003].
- At slack 0.6 the throttling gain isn't shown at either delay: -20.5 [-58.5, +32.2] and -5.1 [-62.2, +55.2]. At slack 1.0 it is: -32.9 [-40.2, -24.5] and -43.1 [-55.2, -32.1].
- The methods are matched on slack over usage, not on allocated core-hours. The computed hold allocates slightly more than the learned hold: +118 core-hours [18, 257] at 5 minutes and +130 [30, 264] at 15, about 0.1% of the total.
- At 30 minutes, `fwdmax6` with the encoded hold throttles 234.0 core-hours less than the paper's target [-254.5, -217.5], 18.7%, mostly from the longer look-ahead. Against plain `fwdmax6` the hold adds 81.0 [50.4, 110.0].
- In the backlog version, queued core-hours, computed minus learned, are -52.4 [-95.0, -20.1] at 5 minutes and -148.2 [-234.1, -84.6] at 15. On validation the same comparison didn't separate the two holds, so I don't lean on it.

Correction, added with the test result: the dry-run section above gives -4.6% for the controller setting at 5 minutes, in its table and once in the text. The value is -4.65%, which rounds to -4.7%. The frozen text is left as it was.
