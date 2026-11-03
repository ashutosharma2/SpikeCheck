# ADR-010: Compute the hold instead of learning it

- Status: Accepted
- Date: 2026-11-03

## Scenario

The paper's Strategy 1 target for step `t` is the highest CPU from `t-6` to `t+4`. Earlier results show what that bundle does at equal over-provisioning: the backward part, the hold, costs coverage overall and protects follow-on spikes. On the test set at slack 0.8, plain `fwdmax` beats the paper's target by 0.042 overall at 5 minutes and 0.032 at 15, and trails it by 0.081 and 0.084 on follow-on spikes.

A forecast for `t` sees everything up to `t-1`, so that window splits exactly at the present:

`max x[t-6..t+4] = max( max x[t-6..t-1], max x[t..t+4] )`

The first part is already in the model's input. The second part is `fwdmax`'s target. The split also holds after encoding into the model's 40 bins. For a known value `a` and an unknown `F`, the `q` quantile of `max(a, F)` is `max(a, q quantile of F)`, so a perfect model of the paper's target would output the larger of the encoded past peak and `fwdmax`'s forecast at every quantile.

That gives a question the earlier ablation couldn't ask: does the hold need to be learned? I chose this route because it separates deterministic historical information from uncertain future predictions. The historical maximum can be calculated directly, which leaves the model responsible only for forecasting the future.

## Obstacle

No hold still has the best overall coverage. I consider follow-on protection important enough to justify investigating a hold, but that's a judgement. This project has no customer-impact data showing these spikes matter more than others.

The expected effect is small. On validation the overall gap is under one percentage point, with intervals of about 0.006 either side, so results may land very close to any threshold.

The evaluation can't be fully blind. I've already seen plain `fwdmax` against the paper's target on the test set, and those results helped produce this idea. Only the computed hold itself is unseen on test.

Validation results also inform this decision. I saw them before setting the margin, so I won't describe it as chosen without comparative evidence.

## Action

The forecasters:

- Primary: `fwdmax` with the encoded hold. At each quantile the forecast is the larger of `fwdmax`'s output and the highest CPU in `x[t-6..t-1]`, passed through `train.py`'s class encoding and decoded to the upper bin edge, as a model's output would be. A value of 55 becomes 57.5. The hold is always on, uses no spike detector and has no setting tuned on results. The six steps come from the paper.
- Comparison: the paper's target (`proposed`), matched by seed at the same overall slack budget.
- Sensitivity: `fwdmax` with the exact, unrounded hold.
- 30 minutes, reported separately: `fwdmax6` with the encoded hold, against `fwdmax6` and the paper's target. At that delay the forward window of `fwdmax` and of the paper's target stops one step short of the spike, so this answers what to use at 30 minutes, not where the hold belongs.

The claim is combined: the computed hold should be better on overall coverage and non-inferior on follow-on coverage. It isn't a claim that the computed hold is better everywhere.

All four checks read slack 0.8, the three-seed mean of computed minus learned coverage, and a paired 95% bootstrap over whole test traces. All four must pass before I claim the complete hypothesis is supported:

- Overall coverage at 5 minutes: lower bound above zero.
- Overall coverage at 15 minutes: lower bound above zero.
- Follow-on coverage at 5 minutes: lower bound strictly above -0.01.
- Follow-on coverage at 15 minutes: lower bound strictly above -0.01.

The margin is 1 percentage point for operational reasons. On the test set that's about 15 of 1,533 follow-on spikes. It's also about one-sixth of the follow-on gain from introducing a hold, which on validation at 15 minutes is +0.060 with my first rounded version and +0.061 with the encoded hold. Losing more than that is too large a sacrifice relative to the benefit that justified the hold.

What each result means:

| Result | Conclusion |
|---|---|
| All four pass | The complete hypothesis is supported |
| Overall passes, follow-on fails at a delay | I won't claim the computed hold is an acceptable replacement at that delay |
| Follow-on passes, overall fails at a delay | Not shown to be meaningfully worse on follow-on, not shown to be better overall |
| Both fail at a delay | Neither claim holds at that delay |
| Results differ between delays | The conclusion is reported per delay |
| Both follow-on checks fail | I drop computed hold as a replacement for the learned hold |

My prediction, written before the test run: under matched slack budgets, the computed hold will spend a smaller proportion of slack after spikes, leaving more of the budget for other steps where it can prevent throttling. This should result in lower throttled CPU without violating the follow-on margin. It's a hypothesis the experiment must test, not an assumed outcome, and it's judged separately from the four checks:

- Throttled CPU at 5 minutes and at 15 minutes, computed minus learned. I claim a reduction only if the 95% interval is wholly below zero.
- Share of slack in the 6 steps after a spike, computed minus learned, by the same rule.

Reported, not judged:

- Slack budgets 0.6 and 1.0.
- The exact hold and the 30-minute results.
- Seasonal and isolated spikes, and follow-on spikes split by whether the previous peak covers them.
- Spike size, using the quartile cuts from the validation spikes: 10, 16 and 30 CPU % of request.
- Slack and throttled CPU in three regions: after spike, run-up (the spike and the 5 steps before it) and other.

Every paired result shows the three-seed mean, each seed and the trace interval. They measure different things. The interval covers trace sampling with the models held fixed, and the seeds show training randomness. Neither shows that a result carries over to other clusters or later days.

The commit that adds this ADR freezes the configuration, thresholds, subgroup definitions and scripts, and it's pushed before the test run. `scripts/score_hold.py` won't start that run if the output folder exists, if the frozen files differ from that commit, or if the commit hasn't been pushed. Before it scores any computed hold, it checks that plain `fwdmax` and the paper's target match `results/spike_types/paired_by_type.csv` and `results/horizon30/matched_slack_by_seed.csv`. The test runs once, gets reported regardless of outcome, and is never used for later tuning.

## Result

Dry run on the validation traces, seed 0, slack 0.8, computed minus learned:

| Delay | Overall coverage | Follow-on coverage | Throttled core-hours |
|---|---|---|---|
| 5 min | +0.0055 [+0.0002, +0.0122] | -0.0057 [-0.0107, -0.0012] | -34.1 [-53.2, -19.0], -4.1% |
| 15 min | +0.0070 [+0.0015, +0.0133] | -0.0026 [-0.0098, +0.0047] | -56.9 [-83.2, -35.2], -5.9% |

The share of slack after spikes is 27.1% for the computed hold and 29.3% for the learned hold, a difference of -0.023 [-0.025, -0.021]. Plain `fwdmax` spends 16.2%.

Under the rule, three of the four checks pass on validation. The 5-minute follow-on check fails narrowly, so the validation conclusion is delay-specific. Both parts of the prediction are supported on validation. None of this is the test result, which will be added below without changing anything above.

What this gives:

- A comparison that changes only where the hold lives, with the same look-ahead, the same bins and the same budget.
- A failure condition stated before the result.
- A hold whose length can change without retraining a model.

What it costs:

- The follow-on threshold sits close to the validation bounds (-0.0107 and -0.0098), so small differences may decide the outcome.
- A full pass wouldn't show that a hold is worthwhile. No hold still leads overall.
- Throttled CPU is a proxy built from 5-minute maxima, not measured customer impact.
- This is one trace and one time window.

## Troubleshooting

- The comparison changed on validation before the freeze. My first version, `fwdmax6` with a spike-triggered hold, beat the paper's target by 0.027 overall at 15 minutes across three seeds, but `fwdmax6` looks one step further, so it changed the look-ahead as well as the hold. With `fwdmax` and the exact hold the gain was 0.013 (seed 0). Rounding up, so the hold couldn't be cheaper than a model's output, gave 0.009. That rounding left values on a bin edge where they were. CPU values are whole numbers, so 22% of hold values sit on an edge, and a model's encoding sends them to the next bin. The encoded hold gives 0.0070. None of these steps used the test set.
- A spike trigger adds little. Holding only after a detected spike gave +0.016 overall at 15 minutes against +0.013 for always holding (exact hold, seed 0). It would add a detector and its thresholds, so I dropped it.
- I set the margin after seeing validation results at both delays with the earlier rounding: -0.0071 [-0.0126, -0.0024] at 5 minutes and -0.0038 [-0.0102, +0.0033] at 15. I kept it after seeing the encoded results above.
- Validation covers one seed. The trained `fwdmax` models for seeds 1 and 2 weren't kept, only their test predictions, so the dry run can't show seed spread. The test run uses all three.
- `scripts/val_probs.py` regenerates the validation predictions. To confirm a saved model is the right one, it compares its output with the saved test predictions on three test traces. That compares probabilities only and scores nothing.
- The first-version figures above came from exploration scripts outside the repo, and they're recorded here as history. The Result table regenerates with `scripts/score_hold.py val`.
