# Agulló et al. (2025): Alibaba re-check with fixed metrics, rolling-max baselines and quantile readout

Run on 2026-10-22, following up on `../agullo2025_critical_review.md` (opportunities #2 and #3, audit items D1–D6 and C3–C4).

## What was run

| Step | What | Notes |
|---|---|---|
| Data | Alibaba 2018 `container_usage` (29.2 GB, 4.0 B rows) streamed from Alibaba's OSS bucket and aggregated to 5-min bins without storing the raw file | `scripts/stream_alibaba.py` |
| Selection | Long-running containers (present at t=0, never stopped), one random per `app_du` → 9,505. Then ≤10% empty bins → 8,833. Then max − mean ≥ 5 CPU-pts → **5,946 traces** (paper: 5,168) | The paper's preprocessing is not public, so this is a reconstruction (`scripts/prep.py`). Usage data covers t ≈ 86,400 s to 777,600 s; the 8-day window is bins 288–2591. |
| Split | Random by trace, 70/15/15, seed 0 → 4,162 / 891 / **893 test**. Scaler fit on train only | Same scheme as the paper (random by trace, same time period) |
| Models | The paper's own **TempInc** class (imported from their repo), lag 512, 8 features, 40 CPU bins, Adam lr 0.01 amsgrad, upper-bin-edge decoding as in their code | **cbase**: target x[t]. **proposed**: target max(x[t−6..t+4]), the paper's Strategy-1 target as implemented. **proposed_full**: also the paper's input transform. Inputs are otherwise identical, which isolates the target change. |
| Training | Early stopping on validation CE (patience 3), 60-min cap, 200 random windows per train trace per epoch, **one seed** | Best epochs: cbase 10, proposed 4, proposed_full 6 |
| Baselines | persistence; rolling max (W ∈ 3…288, margin 0/10/25%), which is the Autopilot peak recommender; seasonal max (rolling max ∨ same time yesterday ±30 min); rolling 24 h percentile; two non-causal oracles | `scripts/baselines.py` |
| Readout | CE softmax decoded at quantile q ∈ {0.5…0.99} instead of argmax. No retraining | `train.decode` |
| Metrics | Fixed metrics in raw CPU units (`scripts/metrics.py`), **and** the authors' original `evaluation.py` run unchanged | Fixed metrics: slack/usage; spike recall with the paper's window and band; forward-only precision; coverage and throttled core-hours with actuation delay d = 1/3/6 steps (5/15/30 min) |

## Results

### 1. The paper's headline reproduces, and so does its metric artefact
Using **the authors' own evaluator** on our data, TempInc spike-F1 is **cbase 0.42 → proposed 0.65**. The paper reports 0.27 → 0.64 for Alibaba TempInc (Table 3).

The same evaluator reports over-provisioning of **4×10¹⁵ to 7×10¹⁶ for every forecaster**, persistence included. That confirms D1: the shift-to-1e-10 artefact, not the forecasts, sets those numbers.

### 2. On the paper's own recall criterion, the quantile readout beats Strategy 1 without retraining
Comparison at matched over-provisioning (proposed argmax slack/usage = 0.587):

| Forecaster | slack/usage | spike recall (paper window) | forward-precision F1 |
|---|---|---|---|
| proposed argmax (Strategy 1) | 0.587 | 0.616 | 0.244 |
| **cbase, q = 0.95 readout** | 0.581 | **0.696** | 0.247 |
| **cbase, q = 0.9 readout** | **0.424** | 0.615 | **0.347** |
| best rolling max within budget | 0.567 | 0.353 | 0.193 |

Bootstrap 95% CIs over test traces (500 resamples):
- **cbase q=0.9 − proposed**: Δrecall [−0.009, +0.009], Δslack [−0.180, −0.148]. That is the same recall with about 30% less over-provisioning.
- **cbase q=0.95 − proposed**: Δrecall [+0.069, +0.090] with Δslack [−0.020, +0.006]. That is higher recall at the same slack.

So, **with the paper's own recall window, a plain CE model read at an upper quantile dominates the Strategy-1 target.** Figure: `results/pareto_recall_paper.png`.

### 3. With actuation delay, Strategy 1 does win, because it forecasts further ahead
The ranking flips when the allocation must be decided 15 min in advance (`cover@3`):

| | slack/usage | spikes covered, 15-min delay | 30-min delay |
|---|---|---|---|
| proposed argmax | 0.587 | **0.566** | 0.423 |
| cbase q = 0.95 | 0.581 | 0.443 | 0.415 |
| best rolling max within budget | 0.567 | 0.324 | 0.329 |

Bootstrap: cbase q=0.95 − proposed Δcover@3 is [−0.139, −0.108]. The proposed family's quantile readouts also form the best front for cover@3 (`results/pareto_cover_at_3.png`).

The likely mechanism [inference]: the Strategy-1 target includes x[t+1..t+4], which gives a 20-minute look-ahead. Its value is the **horizon**, not the asymmetric shaping. Supporting this, a *perfect* Strategy-1 forecaster covers only 28% of spikes at a 30-min delay, versus 99.6% at 5 min.

### 4. Measured as throttled CPU volume, the families nearly converge
Throttled core-hours with a 15-min delay, summed over 893 test traces:

| | slack/usage | throttled core-h |
|---|---|---|
| proposed argmax | 0.587 | 1,585 |
| cbase q = 0.95 | 0.581 | 1,526 |
| rolling max w=12 | 0.567 | 1,863 |
| seasonal max w=6 | 0.741 | 1,221 |
| proposed q = 0.7 | 0.716 | 1,169 |

All learned variants and seasonal-max lie within about 5–10% of each other at matched slack. Rolling max is about 15–20% worse (`results/pareto_throttle_core_h_at_3.png`). Large differences in spike-event metrics translate into **small differences in throttled volume**. This is the first concrete evidence for opportunity #1 (metric validity).

### 5. Other findings
- **Rolling max does *not* recover Strategy 1's gain** at matched slack (recall 0.35 vs 0.62). My hypothesis H3, "gains are mostly persistence", is **not supported** in this form. The learned models capture more than recent peaks. Seasonal max comes closest (recall 0.58 at slack 0.74), but proposed still beats it at equal slack (0.69).
- **Lag-rewarding precision is confirmed.** The paper's evaluator gives rolling max and persistence precision of about 0.99. Forward-only precision gives 0.07 and 0.26. For proposed argmax, the paper's precision is 0.78 while forward precision is 0.15.
- **The input transform barely matters.** proposed_full ≈ proposed: recall 0.61 vs 0.62, cover@3 0.56 vs 0.57. The gain comes from the target, which addresses review item C1.

## Follow-up: is it the horizon or the shaping? (3 seeds, run 2026-10-22/23)

Strategy 1's target, max(x[t−6 … t+4]), does two things at once:
- **look-ahead (horizon):** it covers the next 5 steps (25 min) from the decision point;
- **shaping:** it holds the allocation high for 6 steps after a spike.

To separate them, I trained a third target, **fwdmax** = max(x[t … t+4]). It keeps the same look-ahead and drops the past-hold. cbase, proposed and fwdmax were each trained with **3 seeds** (0, 1, 2) on the same data, inputs, model and settings as above.

Each model's softmax was decoded at 15 quantiles (q = 0.1 … 0.995), which gives one over-provisioning curve per run. Every curve was then interpolated at fixed **slack/usage budgets**, so the comparisons below are at equal over-provisioning. Differences are **seed-paired** (seed k vs seed k).

Files are in `results_horizon/`: `matched_slack_summary.csv`, `paired_vs_proposed.json`, `all_runs.csv` and the `seeds_*.png` plots.

### Result: the horizon explains Strategy 1's advantage, and the past-hold makes it worse

Mean over 3 seeds (± sd), at matched over-provisioning:

| slack budget | metric | cbase | **fwdmax** (horizon only) | proposed (Strategy 1) |
|---|---|---|---|---|
| 0.6 | spikes covered, 15-min delay | 0.464 ± .009 | **0.681 ± .005** | 0.597 ± .010 |
| 0.6 | throttled core-h, 15-min delay | 1,522 ± 31 | **1,160 ± 11** | 1,430 ± 35 |
| 0.8 | spikes covered, 15-min delay | 0.570 ± .006 | **0.757 ± .005** | 0.718 ± .015 |
| 0.8 | throttled core-h, 15-min delay | 1,210 ± 30 | **860 ± 13** | 980 ± 43 |
| 1.0 | spikes covered, 15-min delay | 0.651 ± .004 | **0.807 ± .004** | 0.783 ± .013 |
| 1.0 | throttled core-h, 15-min delay | 989 ± 30 | **686 ± 13** | 755 ± 39 |
| 0.8 | spikes covered, no extra delay | 0.741 ± .011 | **0.751 ± .005** | 0.699 ± .019 |
| 0.8 | spikes covered, 30-min delay | 0.533 ± .004 | **0.590 ± .009** | 0.556 ± .007 |

Seed-paired differences, **fwdmax − proposed**. In every case all 3 seeds have the same sign.

| slack | Δ covered @15 min [min, max] | Δ throttled core-h @15 min [min, max] |
|---|---|---|
| 0.6 | +0.084 [+0.079, +0.089] | −270 [−295, −232] (−19%) |
| 0.8 | +0.039 [+0.029, +0.050] | −120 [−154, −83] (−12%) |
| 1.0 | +0.024 [+0.017, +0.034] | −68 [−100, −36] (−9%) |
| 1.2 | +0.019 [+0.013, +0.027] | −55 [−86, −20] (−9%) |

What this shows:
1. **The look-ahead is what helps.** At equal over-provisioning, the horizon-only target beats Strategy 1 on spike coverage under a 15-min delay, under a 30-min delay, and with no delay, and it throttles 9–19% less CPU. This holds in all 3 seeds at every budget tested.
2. **The past-hold is a cost, not a benefit.** Holding the allocation high after a spike spends over-provisioning on periods that have already passed. The gap is largest at tight budgets: +8 points of coverage and −19% throttling at slack 0.6.
3. **The look-ahead costs nothing without delay.** With no actuation delay, fwdmax matches cbase (0.674 vs 0.671 at slack 0.6; 0.751 vs 0.741 at 0.8). Under a 15-min delay it is far better (+0.19–0.22 coverage at slack 0.6–0.8).
4. **The cbase vs proposed result from the single-seed run holds up across seeds.** cbase wins or ties with no delay (Δcover@1 +0.01 to +0.11) and loses under a 15-min delay (Δcover@3 −0.11 to −0.15, all seeds).
5. **The horizon has to match the actuation delay.** At a 30-min delay, every model trained on a 25-min look-ahead loses most of its edge (fwdmax − proposed only +0.02 to +0.05). Coverage at slack 0.8 stays below 0.6 for all of them. [Inference: a target whose look-ahead equals the actuation delay should restore the gain. Not tested.]
6. **Training-seed variance is small.** sd ≤ 0.02 on coverage. Strategy 1 is the noisiest (throttled-core-h sd 35–43 vs 11–13 for fwdmax).
7. **On the paper's own recall window** (no delay, symmetric band), fwdmax ≈ proposed at slack ≥ 0.8 (Δ ≈ 0, range −0.02 to +0.04). The paper's metric cannot see this difference. It only shows up once delay or throttled volume is measured.

This also refines result 4 above. Once the horizon is separated from the shaping, the learned targets are no longer "within 5–10% of each other". A horizon-matched target throttles 24–31% less than cbase, and 9–19% less than Strategy 1, at equal over-provisioning.

![coverage, 15-min delay](results_horizon/seeds_cover_at_3.png)
![throttled core-hours, 15-min delay](results_horizon/seeds_throttle_core_h_at_3.png)

**Run notes:**
- fwdmax and seeds 1–2 of cbase/proposed were trained with a lower-memory `train.py` (inputs memory-mapped and stored as float16 on the GPU; peak footprint 3.9 GB → 2.5 GB).
- The seed-0 cbase/proposed models come from the first run, which used float32 inputs. Their results fall within the seed spread of the float16 runs.
- Best epochs: cbase 13/14/8, fwdmax 11/16/16, proposed 7/15/19, all within the 60-min cap.
- The seasonal-max baseline is in `matched_slack_by_seed.csv` for reference. At slack 0.8 it covers 0.643 at a 15-min delay, which is below both learned look-ahead targets.

## Limitations (read before citing)
- **One seed per model in the first run** (the horizon follow-up uses 3), a 60-min training cap and early stopping. lr = 0.01 (the paper's) made validation loss noisy, and proposed stopped after 7 epochs. The CIs cover test-trace sampling only, not training variance.
- **The preprocessing is a reconstruction.** Trace count, null and variance filters, the target definition (max CPU per 5-min bin) and the bins (40 × 2.5 CPU-pts) approximate, but are not, the authors' pipeline.
- **The corrected spike detector** (refractory period of 4 steps, minimum rise of 2 CPU-pts) changes which events count as spikes, so recall here is not numerically comparable to the paper's tables. Only the authors' evaluator rows are.
- **The throttling model is crude.** It uses 5-min max CPU as demand, ignores intra-bin dynamics and contention, and has no SLO signal.
- **Same time window for train and test** (paper's design). Temporal and cross-cluster generalisation is untested.
- **Not done yet:**
  - Strategy 2 (altered loss)
  - conformal calibration
  - spike-type stratification (#3)
  - a look-ahead matched to a 30-min delay

## Reproduce
`scripts/run_all.sh` (prep → train ×3 → eval). For the horizon follow-up, run `scripts/run_seeds.sh` (7 training runs), then `scripts/eval_seeds.py`. `scripts/overnight.sh` chains both under `caffeinate`. It expects the aggregated `alibaba_5min.npz` from `stream_alibaba.py`, which takes about 2.3 h at 4 MB/s, and the authors' repo at `../repo/rf`.
