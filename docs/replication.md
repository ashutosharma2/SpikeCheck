# Agulló et al. (2025) on the Alibaba trace: corrected metrics, rolling-max baselines and quantile readout

Run on 2026-10-22 as a follow-up to the [critical review](review.md). It tests opportunities #2 and #3 and audit items D1 to D6 and C3 to C4.

## What was run

| Step | What | Notes |
|---|---|---|
| Data | Alibaba 2018 `container_usage` (29.2 GB, 4.0 billion rows), streamed from Alibaba's public bucket and summed into 5-minute bins without storing the raw file | `scripts/stream_alibaba.py` |
| Selection | Long-running containers (present at t=0, never stopped), one at random per `app_du`: 9,505. Then at most 10% empty bins: 8,833. Then max minus mean of at least 5 CPU points: 5,946 traces (the paper had 5,168) | The paper's preprocessing isn't public, so this is a reconstruction (`scripts/prep.py`). Usage data runs from about 86,400 s to 777,600 s, so the 8-day window is bins 288 to 2591. |
| Split | Random by trace, 70/15/15, seed 0: 4,162 / 891 / 893 test traces. Scaling fitted on training traces only | Same scheme as the paper (random by trace, same time period) |
| Models | The authors' own TempInc class (imported from their repo), lag 512, 8 features, 40 CPU bins, Adam with learning rate 0.01 and amsgrad, decoding to the upper bin edge as in their code | cbase: target x[t]. proposed: target max(x[t−6..t+4]), the paper's Strategy 1 target as implemented. proposed_full: also the paper's input transform. Inputs are otherwise identical, so only the target changes. |
| Training | Early stopping on validation cross-entropy (patience 3), 60-minute cap, 200 random windows per training trace per epoch, one seed | Best epochs: cbase 10, proposed 4, proposed_full 6 |
| Baselines | Persistence; rolling max (W from 3 to 288, margin 0, 10 or 25%), which is Autopilot's peak recommender; seasonal max (rolling max or the same time yesterday ±30 min, whichever is higher); rolling 24-hour percentile; two oracles that see the future | `scripts/baselines.py` |
| Readout | The classifier's softmax read at a quantile q from 0.5 to 0.99 instead of its most likely class. No retraining | `train.decode` |
| Metrics | Corrected metrics in raw CPU units (`scripts/metrics.py`), plus the authors' own `evaluation.py`, run unchanged | Corrected metrics: slack divided by usage; spike recall with the paper's window and band; forward-only precision; spike coverage and throttled core-hours with an actuation delay of d = 1, 3 or 6 steps (5, 15 or 30 minutes) |

## Results

### 1. The paper's headline reproduces, and so does its metric artefact
On the authors' own evaluator, TempInc's spike F1 goes from 0.42 (cbase) to 0.65 (proposed) on our data. The paper reports 0.27 to 0.64 for Alibaba TempInc (Table 3).

The same evaluator gives every forecaster an over-provisioning between 4×10¹⁵ and 7×10¹⁶, including persistence. That confirms D1: the shift to 1e-10 sets those numbers, not the forecasts.

### 2. On the paper's own recall measure, a quantile readout beats Strategy 1 without retraining
Compared at matched over-provisioning (proposed argmax has slack/usage 0.587):

| Forecaster | slack/usage | spike recall (paper window) | F1 with forward-only precision |
|---|---|---|---|
| proposed argmax (Strategy 1) | 0.587 | 0.616 | 0.244 |
| cbase, read at q = 0.95 | 0.581 | 0.696 | 0.247 |
| cbase, read at q = 0.9 | 0.424 | 0.615 | 0.347 |
| best rolling max within budget | 0.567 | 0.353 | 0.193 |

Bootstrap 95% intervals over test traces (500 resamples):
- cbase at q=0.9 minus proposed: recall [−0.009, +0.009], slack [−0.180, −0.148]. The same recall with about 30% less over-provisioning.
- cbase at q=0.95 minus proposed: recall [+0.069, +0.090], slack [−0.020, +0.006]. Higher recall at the same slack.

So on the paper's own recall window, a plain cross-entropy classifier read at an upper quantile beats the Strategy 1 target. Figure: [`results/replication/pareto_recall_paper.png`](../results/replication/pareto_recall_paper.png).

### 3. With an actuation delay, Strategy 1 does win, because it forecasts further ahead
The ranking flips when the allocation has to be decided 15 minutes in advance (`cover@3`):

| | slack/usage | spikes covered, 15-min delay | 30-min delay |
|---|---|---|---|
| proposed argmax | 0.587 | 0.566 | 0.423 |
| cbase q = 0.95 | 0.581 | 0.443 | 0.415 |
| best rolling max within budget | 0.567 | 0.324 | 0.329 |

The bootstrap interval for cbase at q=0.95 minus proposed on `cover@3` is [−0.139, −0.108]. Reading the proposed model at different quantiles also gives the best curve for `cover@3` ([`results/replication/pareto_cover_at_3.png`](../results/replication/pareto_cover_at_3.png)).

My reading of why [inference]: the Strategy 1 target includes the next five steps, which gives a 25-minute look-ahead. What helps is the horizon, not the asymmetric shaping. A perfect forecaster of the Strategy 1 target supports this: it covers 99.6% of spikes at a 5-minute delay but only 28% at 30 minutes.

### 4. Measured as throttled CPU, the methods end up close
Throttled core-hours with a 15-minute delay, summed over the 893 test traces:

| | slack/usage | throttled core-h |
|---|---|---|
| proposed argmax | 0.587 | 1,585 |
| cbase q = 0.95 | 0.581 | 1,526 |
| rolling max w=12 | 0.567 | 1,863 |
| seasonal max w=6 | 0.741 | 1,221 |
| proposed q = 0.7 | 0.716 | 1,169 |

At matched slack, all the learned variants and seasonal max land within about 5 to 10% of each other, and rolling max is about 15 to 20% worse ([`results/replication/pareto_throttle_core_h_at_3.png`](../results/replication/pareto_throttle_core_h_at_3.png)). Big differences in spike metrics turn into small differences in throttled CPU. That's the first concrete evidence for opportunity #1, whether the metrics measure the right thing.

### 5. Other findings
- Rolling max doesn't recover Strategy 1's gain at matched slack (recall 0.35 against 0.62), so my hypothesis H3, that the gains are mostly persistence, doesn't hold in this form. The learned models pick up more than recent peaks. Seasonal max comes closest (recall 0.58 at slack 0.74), but proposed still beats it at the same slack (0.69).
- The paper's precision does reward lagging. Its evaluator gives rolling max and persistence a precision of about 0.99; forward-only precision gives them 0.07 and 0.26. For proposed argmax, the paper's precision is 0.78 and forward-only precision is 0.15.
- The input transform barely matters. proposed_full and proposed score 0.61 and 0.62 recall, and 0.56 and 0.57 coverage at a 15-minute delay. The gain comes from the target, which answers review item C1.

## Follow-up: is it the horizon or the shaping? (3 seeds, run 22 to 23 October 2026)

Strategy 1's target, max(x[t−6 … t+4]), does two things at once. It looks ahead, covering the next 5 steps (25 minutes) from the decision point. And it holds the allocation high for 6 steps after a spike.

To separate the two, I trained a third target, fwdmax = max(x[t … t+4]), which keeps the look-ahead and drops the hold. cbase, proposed and fwdmax were each trained with three seeds (0, 1, 2) on the same data, inputs, model and settings as above.

Each model's softmax was read at 15 quantiles (q from 0.1 to 0.995), giving one over-provisioning curve per run. Each curve was then interpolated at fixed slack/usage budgets, so the comparisons below are at equal over-provisioning, and differences are paired by seed (seed k against seed k).

The files are in [`results/horizon/`](../results/horizon/): `matched_slack_summary.csv`, `paired_vs_proposed.json`, `all_runs.csv` and the `seeds_*.png` plots.

### Result: the horizon explains Strategy 1's advantage, and the hold makes it worse

Mean over 3 seeds (± sd), at matched over-provisioning:

| slack budget | metric | cbase | fwdmax (look-ahead only) | proposed (Strategy 1) |
|---|---|---|---|---|
| 0.6 | spikes covered, 15-min delay | 0.464 ± .009 | 0.681 ± .005 | 0.597 ± .010 |
| 0.6 | throttled core-h, 15-min delay | 1,522 ± 31 | 1,160 ± 11 | 1,430 ± 35 |
| 0.8 | spikes covered, 15-min delay | 0.570 ± .006 | 0.757 ± .005 | 0.718 ± .015 |
| 0.8 | throttled core-h, 15-min delay | 1,210 ± 30 | 860 ± 13 | 980 ± 43 |
| 1.0 | spikes covered, 15-min delay | 0.651 ± .004 | 0.807 ± .004 | 0.783 ± .013 |
| 1.0 | throttled core-h, 15-min delay | 989 ± 30 | 686 ± 13 | 755 ± 39 |
| 0.8 | spikes covered, no extra delay | 0.741 ± .011 | 0.751 ± .005 | 0.699 ± .019 |
| 0.8 | spikes covered, 30-min delay | 0.533 ± .004 | 0.590 ± .009 | 0.556 ± .007 |

Differences paired by seed, fwdmax minus proposed. All three seeds have the same sign in every row.

| slack | covered at 15 min [min, max] | throttled core-h at 15 min [min, max] |
|---|---|---|
| 0.6 | +0.084 [+0.079, +0.089] | −270 [−295, −232] (−19%) |
| 0.8 | +0.039 [+0.029, +0.050] | −120 [−154, −83] (−12%) |
| 1.0 | +0.024 [+0.017, +0.034] | −68 [−100, −36] (−9%) |
| 1.2 | +0.019 [+0.013, +0.027] | −55 [−86, −20] (−9%) |

What this shows:
1. The look-ahead is what helps. At equal over-provisioning, the look-ahead-only target covers more spikes than Strategy 1 at a 15-minute delay, at a 30-minute delay and with no delay, and it throttles 9 to 19% less CPU. That holds in all three seeds at every budget tested.
2. The hold costs more than it gives. Keeping the allocation high after a spike spends over-provisioning on time that has already passed. The gap is widest when the budget is tight: +8 points of coverage and 19% less throttling at slack 0.6.
3. Looking ahead costs nothing when there's no delay. With no actuation delay fwdmax matches cbase (0.674 against 0.671 at slack 0.6, 0.751 against 0.741 at 0.8). At a 15-minute delay it's far better, with 0.19 to 0.22 more coverage at slack 0.6 to 0.8.
4. The single-seed result for cbase against proposed holds across seeds. cbase wins or ties with no delay (0.01 to 0.11 more coverage) and loses at a 15-minute delay (0.11 to 0.15 less, in every seed).
5. The look-ahead has to match the delay. At a 30-minute delay, every model trained on a 25-minute look-ahead loses most of its edge (fwdmax beats proposed by only 0.02 to 0.05), and coverage at slack 0.8 stays below 0.6 for all of them. [Inference: a target whose look-ahead matches the delay should bring the gain back. Not tested yet.]
6. Seeds make little difference. Coverage varies by an sd of at most 0.02. Strategy 1 is the noisiest, with a throttled core-hours sd of 35 to 43 against 11 to 13 for fwdmax.
7. The paper's recall window can't see any of this. With no delay and a symmetric band, fwdmax and proposed score about the same at slack 0.8 and above (differences from −0.02 to +0.04). The difference only shows once delay or throttled CPU is measured.

This also changes result 4 above. Once the look-ahead is separated from the hold, the learned targets are no longer within 5 to 10% of each other: a target that only looks ahead throttles 24 to 31% less than cbase and 9 to 19% less than Strategy 1 at equal over-provisioning.

![coverage, 15-min delay](../results/horizon/seeds_cover_at_3.png)
![throttled core-hours, 15-min delay](../results/horizon/seeds_throttle_core_h_at_3.png)

Run notes:
- fwdmax, and seeds 1 and 2 of cbase and proposed, were trained with a lower-memory `train.py`: inputs are memory-mapped and stored as float16 on the GPU, which cut peak memory from 3.9 GB to 2.5 GB.
- The seed-0 cbase and proposed models come from the first run, which used float32 inputs. Their results fall inside the spread of the float16 seeds.
- Best epochs: cbase 13, 14 and 8; fwdmax 11, 16 and 16; proposed 7, 15 and 19. All finished within the 60-minute cap.
- Seasonal max is in `matched_slack_by_seed.csv` for reference. At slack 0.8 it covers 0.643 of spikes at a 15-minute delay, below both learned look-ahead targets.

## Limitations (read before citing)
- The first run used one seed per model (the horizon follow-up uses three), a 60-minute training cap and early stopping. The paper's learning rate of 0.01 made validation loss noisy, and proposed stopped after 7 epochs. The confidence intervals cover the sampling of test traces, not training variance.
- The preprocessing is a reconstruction. The trace count, the gap and variance filters, the target (max CPU per 5-minute bin) and the bins (40 of 2.5 CPU points each) approximate the authors' pipeline but aren't it.
- The corrected spike detector (a 4-step refractory period, a minimum rise of 2 CPU points) changes which events count as spikes, so recall here can't be compared number for number with the paper's tables. Only the rows from the authors' evaluator can.
- The throttling measure is crude. It treats the 5-minute maximum CPU as demand, ignores what happens inside each bin and any contention, and has no SLO signal.
- Training and test traces cover the same time window, as in the paper. Generalisation across time and clusters is untested.
- Not done yet: Strategy 2 (the altered loss), conformal calibration, splitting spikes by type (#3), and a look-ahead matched to a 30-minute delay.

## Reproduce
From the repo root:

1. `scripts/setup.sh` clones the authors' code at the pinned commit and installs the packages in `requirements.txt`.
2. `scripts/get_data.sh` streams the trace (about 2.3 hours at 4 MB/s) and builds `data/data.npz`.
3. `scripts/run_all.sh` runs the first replication (three targets, seed 0) and writes `results/replication/`.
4. `RUNS="fwdmax:0 fwdmax:1 fwdmax:2 cbase:1 proposed:1 cbase:2 proposed:2" scripts/overnight.sh` runs the horizon follow-up and writes `results/horizon/`.

`SPIKECHECK_DATA` and `SPIKECHECK_UPSTREAM` point the scripts at a different data folder or copy of the authors' code (see `scripts/common.sh`).
