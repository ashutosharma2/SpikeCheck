# Critical review and research opportunities
## Agulló, Gutierrez-Torre, Torres and Berral: "Enhancing the output of time series forecasting algorithms for cloud resource provisioning", FGCS 170 (2025) 107833

Written on 2026-10-22 from two sources: the paper (15 pages) and the authors' public code at
`github.com/FerranAgulloLopez/ResourceForecasting` (commit `98f0d51`, 16 April 2025).

Each claim carries a label. [Paper] means the authors say it. [Code] means I checked it in their
repository. [Inference] is my own reading and hasn't been tested. [Lit] comes from a literature
search on 2026-10-22, which was not exhaustive.

---

## 0. Bottom line

1. **The central idea is sound.** Training a model on a provisioning target (a maximum over a window around each step) instead of the raw series is cheap, works with any model, and holds up.
2. **The over-provisioning numbers are numerical artefacts.** The code shifts the values so that the test set's smallest value is 1e-10 (`evaluation.py:27`), and over-provisioning then divides by it (`:83`). That single point dominates the metric, which is why the paper reports values from 1e12 to 1e18, and why five of its six Alibaba models get exactly the same value (Table 3: 4.8636E+12 for Proposed and 1.9181E+12 for CBaseline). The headline trade-off, better spikes at the cost of more over-provisioning, is never actually measured.
3. **The spike metrics favour two kinds of forecast:** ones that stay high after a spike, and ones that lag behind it. Section 2-D has the details.
4. **Strategy 2 (the loss change) isn't supported by its own evidence.** Its "single" variant is mathematically close to plain cross-entropy, yet it beats cross-entropy by about 40% in F1 (Table 4). That gap has to come from tuning, selection or seed noise, not from the altered targets. The authors raise this concern themselves (p. 9).
5. **The biggest open question:** nobody has shown whether spike-focused forecast metrics, or the gains reported here, lead to less throttling and lower SLO cost once actuation delay is included. It's also untested whether a calibrated quantile or conformal forecast of the forward-window maximum gives the same benefit with a single, readable setting. Section 4 ranks seven research opportunities; #1 to #3 combine into one strong paper.

---

## 1. What the paper contributes

### 1.1 Technical map

| # | Contribution | Mechanism (where) | Assumptions | Evidence offered | Unresolved |
|---|---|---|---|---|---|
| C1 | Separate input and target series | Input: past-only transform (log, then max over the last 2 steps). Target: log, max over the next 5 steps, max over the last 7, then discretised (§2.1, Fig. 6; the config matches [Code]) | Provisioning needs an allocation that rises before a spike and stays up, not the point value; transforms "must be revertible" (p. 4) | Table 3: spike F1 is about 2.2 times the best baseline in 11 of 12 model and dataset cells | A max filter isn't revertible, which contradicts the paper's own requirement. Inputs and target change together, so the gain can't be attributed (no ablation). Only one configuration was tested. |
| C2 | Loss change (altered ground truth in a distillation-style loss) | Eq. 1: α·KLD(M(x)/T, altered_y/T) + (1−α)·CE; variants single, binary, proportional, exponential (§2.2, §4.3.2) | Asymmetric soft labels push the classifier upward | Table 4 (TempInc only): F1 0.39 to 0.40 against 0.27 for CE (Alibaba) | "Single" is close to CE but scores 0.387. Eq. 1 is missing its "+". The code applies a softmax to the target (`loss_functions.py:61`) that Eq. 1 doesn't show. α was chosen by best spike F1, on a split the paper doesn't name. |
| C3 | New evaluation: spike precision and recall, over- and under-provisioning | Algorithms 1 and 2, Eqs. 2 to 5, constants γ=250, σ=2, α=0.2, β=0.5, τ=4 (§3, §4.1.4) | Statistically defined spikes stand in for harmful events; relative error is the right scale | Table 2: an oracle "max-future" filter gets precision and recall near 1 but the worst MSE | Does it track SLO cost? How sensitive is it to the constants? Is it numerically sound (see D1)? |
| C4 | Explainability | DeepLift-SHAP on TempInc; KMeans and t-SNE on the 2,000 and 7,000 largest correctly predicted spikes (§4.4, Figs. 9 to 11) | Attributions are faithful; clusters mean something | Four "reasons": last sample spiked, recent spike, recent variability, daily seasonality | Only hits are analysed. No faithfulness test. Three of the four reasons are reactions to the recent past (see E). |

### 1.2 Problem and hypotheses
- **Problem [Paper, §1.1].** Forecasters tuned for average accuracy predict spikes late or too low. In shared clusters (Borg, Alibaba), a high-priority workload that spikes may find the resources it paid for lent to low-priority work, which leads to throttling and SLO violations.
- **Hypotheses [Inference].** None are stated, but the paper implies four. H1: changing the training target can make a forecaster anticipate spikes at the right height. H2: this costs a controllable amount of over-provisioning. H3: MSE-style metrics rank forecasters wrongly for this goal. H4: model capacity matters (deep beats simple).
- **Scope [Paper, §4.1.1].** One-step-ahead CPU at 5-minute steps, a single target with several inputs (14 features for Google, 18 for Alibaba).

### 1.3 Experimental set-up (as reported, plus what the code shows)
- **Data [Paper §4.1.2, Table 1].** Google 2019: cluster *a* only, production tier, more than 15 days, one instance per workload, then a PCA cluster of "common patterns" (1,722 traces). Alibaba 2018: long-running containers with low-variability traces removed (5,168 traces). Both resampled to 5 minutes and standardised.
- **Split.** 70/15/15, random by trace [Paper p. 7], so every split covers the same calendar period.
- **Discretisation.** The paper says "40 equal intervals over dataset min-max". The Alibaba Proposed config has 45 boundaries (0.25 to 11.25), which are log-transformed [Code].
- **Decoding [Code `full_values_unbiased_intervals.py:117`].** The predicted class becomes the upper edge of its bin, so every classification variant (CBaseline and Proposed) is biased upward by up to one bin. The paper doesn't mention this.
- **Models.** Random forest, a linear SVM (LinearSVR/LinearSVC, not kernel), DLinear, SCINet, Informer and the authors' TempInc. The random forest and SVM use all features only at lag 32 because of memory limits, so the models aren't compared on equal terms [Paper App. A].
- **Tuning.** Small grids (lag and learning rate, plus a few architecture settings), with no stated selection criterion. Every config sets `manual_seed: 0` [Code], so results come from a single seed with no variance reported.
- **Unreported baselines.** The repo implements Autopilot and Theta-scan, the senior author's own vertical autoscaler (`app/factories/method_factory.py`). Neither appears in the paper or has an experiment config [Code].

### 1.4 Main results [Paper]
- Table 2: MSE ranks the oracle envelope worst; the spike metrics rank it best.
- Table 3: Proposed raises spike recall a lot and lowers under-provisioning. Precision is mixed. SVM on Google is the one failure (F1 0.13 against 0.20).
- Table 4: Strategy 2 improves F1 over MSE and CE, and "single" does about as well as the altered variants.
- Best model: TempInc (Alibaba F1 0.64, Google 0.80). DLinear comes close on Google (0.66).

---

## 2. Methodological audit

Severity: Critical (invalidates a stated conclusion), Major (materially weakens it), Minor.

### A. Problem formulation

**A1. Spike prediction is never linked to provisioning outcomes (Major).** Nothing is simulated: no autoscaler, scheduler, throttling model or SLO signal. The paper assumes that better spike F1 means fewer SLA violations (§1.2, §6). CPU is compressible, so under-provisioning causes throttling and latency, not failure. The cost of a miss depends on how deep and how long the shortfall is, which a count of spike events can't capture.

**A2. Actuation delay isn't modelled (Major).** One step is 5 minutes. The recall window accepts a hit anywhere in [k−4, k] (τ=4), including the forecast for step k itself, made at k−1. Whether 5 minutes is enough warning depends on what acts on the forecast: reclaiming CPU from best-effort neighbours takes seconds, while resizing with a restart or adding a VM takes minutes. The paper argues from lead time ("leaves no time for corrective actions", §1.3.2) but never puts a number on it.

**A3. The asymmetric cost is ad hoc (Major).** Asymmetry shows up in three places that aren't coordinated: α≠β in the tolerance band, the max-filter target, and upper-edge decoding [Code]. None comes from a cost ratio. The paper promises users a trade-off setting, but there's no mapping from any of its settings (filter width, α, T, slope) to an operating point.

**A4. New regimes and unseen magnitudes are ruled out by design (Major).** The classification head has a fixed range, and the code raises an exception for targets outside it (`full_values_unbiased_intervals.py:92`), so it can't represent a spike above the training range. Random splits by trace over one shared period also hide anything that changes over the calendar.

### B. Data and generalisation

**B1. Selection bias by construction (Major).** Google: one cluster, production tier, more than 15 days, then a PCA cluster of similar traces. Alibaba: low-variance traces dropped. The results describe a uniform, spike-rich subset. The over-provisioning cost of a whole fleet, which the many flat traces dominate, isn't represented.

**B2. No holdout in time or across clusters (Major).** With a random split by trace, test traces share the same wall-clock window as training traces. Cluster-wide events such as deployments, daily peaks and incidents affect many containers at once, so training sees "future" information from other traces. Google 2019 has 8 clusters and Alibaba covers 8 days, so tests across clusters or forward in time were easy to run, and weren't.

**B3. Possible preprocessing leakage (unverified).** Standardisation and the min-max range for binning ("of the corresponding dataset", p. 7) seem to use all traces. The preprocessing scripts aren't in the repo, which only reads `third_preprocess_compressed_standarized`. The repo also has `replace_downward_spikes_with_previous`, which uses the full trace's mean and standard deviation (`time_series_preprocessing.py:289`). I found no config that calls it, so whether it was used is unconfirmed. The severity depends on that, so I rate it Minor to Major.

**B4. Representativeness (Minor to Major, depending on the claim).** The 2018 and 2019 traces predate much of today's serverless, GPU and LLM-serving and microservice work. Alibaba's 2021 and 2022 microservice traces include response times and call rates [Lit], so labels based on service quality are available now.

### C. Forecasting method

**C1. The gains are confounded (Major).** Proposed changes the input transform (log, max over the last 2) and the target (max over the next 5 and last 7) at once, and both were picked by "an initial parameter search" (p. 8) on an unnamed split. There's no ablation (target only, input only) and no sweep over window widths, even though the widths are offered as the user's setting.

**C2. The target is built to match the metric (Major) [Inference].** The target, max(x_{t−7…t+4}), is close to the oracle envelope in Table 2, and the recall tolerance τ=4 matches its 5-step future window. So Strategy 1 largely teaches the model to imitate the oracle the metric was designed to reward. Aligning a target with a goal is legitimate, but it means part of the headline gain comes from designing the metric and target together, not from better forecasting.

**C3. Strong, cheap baselines are missing (Critical for the claim that deep models are needed, §5).** None of these is reported:
- the rolling maximum of the last W steps times (1+m), which is Autopilot's moving-window peak recommender (the code is in the repo);
- a seasonal envelope, max(x_{t−288±k});
- a quantile regressor (pinball loss) on the same target;
- the CBaseline cross-entropy classifier read at an upper quantile instead of its most likely class, which needs no retraining.

Simply repeating the last value already reaches recall 0.15 and 0.20 (Table 2), so a rolling-max envelope might recover much of the reported gain. Until that's tested, the claim that "complex models are required" (§5) is unsupported.

**C4. Strategy 2 contradicts itself (Major).** With one-hot targets and softmax(target/T) where T=0.1 [Code `loss_functions.py:61`], the KLD term is essentially cross-entropy on temperature-scaled logits. "Single" should therefore match CE, yet it scores F1 0.387 against 0.275 on Alibaba. There's also no reason for the predicted class to move up: a soft label that decays upward keeps its peak at the true class, so the predicted class only shifts through uncertainty. The altered labels are an indirect, uncalibrated way to get an upper quantile, which a direct quantile readout gives exactly.

**C5. Failure modes aren't analysed (Major).** Isolated, first-in-a-burst spikes are never separated from follow-on, ramp-preceded or seasonal ones. Three of the four "reasons" in the explainability results are reactions to the recent past, which suggests most hits are follow-on or seasonal spikes. See opportunity #3.

### D. Evaluation method

**D1. Over- and under-provisioning are numerically degenerate (Critical) [Code].**
`move_over_0` shifts all values by |min| + 1e-10, so the test set's smallest value becomes exactly 1e-10 (`evaluation.py:27`). Eq. 4 divides by x[t] (`:83`). At that one point, a prediction one bin (about 0.25) higher gives (0.25/1e-10)², about 6×10¹⁸. Averaged over about 1,800 points and 775 test traces, that's roughly 4×10¹², the same order as the Alibaba entries in Table 3. This explains:
- the absurd magnitudes (1e12 to 1e18);
- why five of six Alibaba models report identical over-provisioning (every classifier predicts the same bin at that point);
- why all four Table 4 variants share 1.9181E+12.

So the over-provisioning figures in Tables 2 to 4 say almost nothing about over-provisioning across the trace, and the trade-off claims in the abstract, §4.3.1 and §5 aren't supported by them.

Separately, the "relative" error is computed on shifted z-scores. Where zero falls depends on which traces ended up in the test set, so the error isn't relative to CPU cores, and the paper's justification ("a node can support a larger spike if the workload has greater consumption", p. 5) doesn't apply.

**D2. The tolerance band depends on an arbitrary zero point (Major) [Code].** The band x_k(1−α) < y' < x_k(1+β) is applied to shifted z-scores. Spikes in traces near the dataset's minimum get an extremely narrow band and busy traces get a wide one, so recall is mixed up with each trace's overall level. The code's default β is 0.6, not the 0.5 the paper states (`evaluation.py:202`), and the training pipeline calls the evaluation with the defaults.

**D3. The spike detector breaks right after a spike (Major) [Code].** After a detection at j, the reference window becomes [j, k) (`evaluation.py:63`). At k = j+1 that window holds a single value with zero spread, so any increase counts as a spike, and a rising ramp counts as several. The same rule rewards forecasts that stay flat and high: they produce no further predicted spikes, so they can't lose precision.

**D4. Precision rewards forecasts that lag (Major).** The precision search window is symmetric (±τ), so a forecast that copies a spike one step late still counts as matching it. Repeating the last value scores precision 0.9987 and 1.000 (Table 2). Precision can't tell anticipating from reacting.

**D5. No working metric penalises long stretches of over-provisioning (Major) [Inference].** Precision only counts rising edges of y′, and over-provisioning is broken (D1). So sustained over-allocation, the main cost of Strategy 1, is invisible in the evaluation.

**D6. Aggregation (Minor to Major) [Code].** F1 is computed per trace and then averaged (`:340`). That's why the reported F1 isn't the harmonic mean of the reported precision and recall: for Alibaba RF RBaseline, P 0.80 and R 0.066 give 0.12, but 0.089 is reported. A trace with no real spikes gets recall 0 (`:187`), and one with no predicted spikes gets precision 0 (`:143`). Correctly predicting nothing is scored as failure, which pushes models toward predicting spikes. There are no confidence intervals, no repeated seeds and no significance tests.

**D7. No sensitivity analysis on γ, σ, α, β and τ (Major).** Rankings could flip, especially with τ set against the target window (C2) and against the sampling interval.

**D8. The motivating experiment is partly circular (Minor).** Table 2 shows that MSE disagrees with an envelope objective, which is expected because they estimate different things. It doesn't show that the new metric tracks operational cost. A pinball loss at a high quantile or an explicit cost function would be the right comparison.

### E. Explainability
- **Survivorship (Major).** Only correctly predicted, large spikes are clustered, so the analysis can't explain misses.
- **No faithfulness check (Minor to Major).** No deletion or insertion curves, and no stated reference input for DeepLift. The cluster count and the t-SNE reading are qualitative.
- **What it does support [Inference].** Three patterns (the last sample spiked, a recent spike, recent variability) are cues from the immediate past. Combined with the 7-step hold in the target, they suggest many "anticipated" spikes are follow-ons within a burst. The fourth, daily seasonality, is real anticipation, but a seasonal-max heuristic captures it. Overall the analysis weakens the paper's claim that deep models are needed (§5) more than it supports it.

### F. Reporting and reproducibility (Minor)
Eq. 1 is missing its "+". The appendix labels are swapped: the text says Appendix A describes TempInc, but Appendix A lists hyperparameters. The random forest is described as a "Regressor with gini". The paper says 40 intervals; the config has 45. The models called SVM are linear. The preprocessing pipeline isn't in the repo. On the positive side, the public code made most of this audit possible.

---

## 3. What's already been done (search on 2026-10-22, not exhaustive)

| Direction | Existing work [Lit] | Open? |
|---|---|---|
| Conformal prediction for cloud workloads | RSR (Majd et al., arXiv 2607.24773): bootstrapped conformal intervals for VM right-sizing, medium and long term, with backtesting; it calls conformal prediction "underexplored" in cloud | Open: online or adaptive conformal prediction on a forward-window maximum, with a bound on under-provisioning and spike-event evaluation |
| Quantile or asymmetric-loss provisioning | Pinball-loss framing (arXiv 2008.01215); SMU 2024 robust predictive autoscaling with quantiles; an SLO-aware asymmetric LSTM loss | Partly explored. Not compared with target shaping, and not evaluated on spike events |
| Evaluation by decision outcome | Carraro et al. 2023 (model rankings change under energy cost); CLOUDADV (arXiv 2606.31470), sizing aligned with decisions; a 2026 survey calling for operational metrics | Open for spike metrics: no study tests whether rankings by spike F1, MSE, pinball loss or CRPS agree with throttling or SLO cost under actuation delay |
| Spike onset vs recurrence | GEAP (2014), event-driven bursts; HPC I/O burst classification with lead time (arXiv 2308.10311) | Apparently open: no study splits first-in-burst, follow-on and seasonal spikes in cloud CPU traces |
| Extreme value theory for peaks | Shenoy and Gorinevsky, TSC 2016; Bozóki and Pataricza, 2018 and 2022 | Explored; little novelty on its own |
| Follow-ups to this paper | None found | Few competitors (the paper is about 18 months old) |

---

## 4. Research opportunities, ranked

Ranked by novelty, how defensible the result would be, cost, and the chance of a clean answer.

### #1 Do spike metrics measure what provisioning needs? A study grounded in decisions ★★★★★
- **Question.** Across many forecasters, do rankings by MSE, pinball loss, CRPS, spike F1 (as published) and a corrected event metric agree with rankings by operational cost under an actuation delay d?
- **Why it's open.** This paper and the ones before it check their metrics only on hand-made filters (Table 2). Decision-focused work in cloud [Lit] targets sizing cost, not spike metrics.
- **Design.** Replay the traces. The CPU limit at t comes from the forecast issued at t−d, with d in {0, 1, 3, 6}. Unserved CPU, max(x − limit, 0), gives throttled core-seconds; slack, max(limit − x, 0), gives core-hours. Cost is slack + λ·throttle + μ·SLO-minutes where a response-time proxy exists. A co-location mode lets low-priority work borrow the slack, with reclamation taking r seconds.
- **Forecasters.** The paper's six models with RBaseline, CBaseline and Proposed; Autopilot (already in their repo); Theta-scan; rolling max; seasonal max; a quantile GBM; and the conformal envelope (#2).
- **Output.** Kendall-τ heatmaps over metric, λ and d, showing where spike F1 holds up and where it misleads.
- **Risk.** Spike F1 might turn out to track cost well. That would still be publishable as validation. Venues: ICPE, SoCC, CCGrid, FGCS, TPDS.

### #2 Calibrated forward-max envelopes: one setting with a guarantee ★★★★★
- **Reframe.** The paper's target is an estimate of M_t = max(x_{t−w…t+h}). Forecast its conditional q-quantile directly, with pinball loss or by reading a quantile from the existing CE softmax. Then wrap it in adaptive conformal inference (Gibbs and Candès 2021) per trace, so that the long-run rate of under-provisioning stays at or below ε even as workloads drift.
- **Hypothesis.** At equal over-provisioning in core-hours, the conformal envelope matches or beats Proposed on corrected spike recall, and beats it on throttling cost. ε replaces filter width, α, T and slope with one setting an operator can read.
- **Cheapest decisive test.** Read CBaseline's softmax at q from 0.5 to 0.99 instead of its most likely class, and plot the resulting curve against Proposed and Strategy 2. If the curve passes through or above their points, Strategy 2 adds nothing and Strategy 1 comes down to choosing a quantile.
- **Novelty.** The literature calls conformal prediction for cloud underexplored [Lit]. I found nothing that combines a forward-max target, adaptive conformal inference and event-level evaluation.

### #3 Spike types and how predictable they are from CPU history alone ★★★★☆ (cheapest; run first)
- **Question.** Which spikes does Strategy 1 actually improve on?
- **Spike types.** (i) Follow-on, with an earlier spike within W steps. (ii) Seasonal, with a spike near the same time the day before. (iii) Preceded by a ramp. (iv) Isolated onset.
- **Hypothesis H3.** The gains come mostly from (i) and (ii), rolling-max and seasonal-max heuristics match them, and recall on (iv) stays near baseline for every model. The paper's own SHAP patterns, and the 15 to 20% recall from simply repeating the last value, point this way.
- **Upper bound.** Estimate the best possible recall on (iv) from a single CPU history, using the best of all forecasters or an ensemble as a ceiling, plus conditional-entropy estimates.
- **Why it matters.** If H3 holds, progress in "spike forecasting" is mostly persistence, and effort should move to outside signals that lead CPU (#4). Either answer is publishable.
- **Cost.** Days, since their code and configs are public.

### #4 Spikes defined by service quality, and early warnings from call graphs ★★★★☆
- **Idea.** Stop defining spikes statistically (2σ above a weighted mean) and define harmful spikes as CPU rises that come with slower responses. Alibaba's 2021 and 2022 microservice traces provide CPU per container (MSResource), call rates and response times (MSRTMCR) and call graphs [Lit; check the 0.5% sampling and the time span].
- **Then.** Use the call rate of upstream services as an early warning for CPU onsets downstream: type (iv), which history alone can't reach.
- **Novelty.** Graph-based microservice autoscalers exist (for example GRAF, CoNEXT 2021; I didn't re-check it). Defining events by service quality and evaluating only onsets appear new.

### #5 A sound event metric that accounts for lead time (fold into #1) ★★★☆☆
Fixes:
- report recall as a curve over lead time ℓ, at decision time k−d;
- penalise lagging in precision;
- give the detector a minimum window and a refractory period;
- give traces without spikes a defined score;
- measure over-provisioning in absolute units (core-hours) or relative to the request or limit, never by dividing by x;
- test sensitivity to γ, σ, α, β, τ and the sampling interval.

### #6 Robustness across time and clusters ★★★☆☆
Train on Google cluster *a* and test on b to h; train on days 1 to 20 and test on 21 to 31. Report how spike F1 and conformal coverage decay, then show that adaptive conformal inference (#2) restores coverage. This supports #2 rather than standing alone.

### #7 Spikes larger than anything in training ★★☆☆☆
Pair the classifier with a peaks-over-threshold tail from extreme value theory, fitted on residuals, to handle magnitudes the fixed bins can't represent (the code raises an exception on them). Extreme value theory has been applied to peaks before [Lit], so the novelty is modest.

### The paper I'd write (#3, then #2, then #1)
Working title: *"Spike-aware provisioning forecasts are mostly persistence: a decision-grounded re-evaluation and a calibrated alternative."*

1. **Reproduce** Table 3 with their code. Fix D1 to D3 and D6. Add the rolling-max, seasonal-max and Autopilot baselines.
2. **Split** spikes by type (#3) and test H3.
3. **Introduce** the quantile envelope with adaptive conformal inference (#2), with ε as the only setting.
4. **Validate** everything in the replay simulator (#1) across d and λ.

When to change course:
- If the rolling maximum doesn't recover most of Proposed's gain, H3 is wrong. Then the question becomes what the deep models learn beyond persistence, which is interesting in its own right.
- If the quantile readout's curve doesn't beat Proposed, target shaping does something quantiles don't. Find out what.

---

## 5. What I couldn't verify
- The preprocessing (scope of standardisation, gap filling, leakage). Those scripts aren't in the repo.
- Which split was used to choose hyperparameters and Strategy 2's α. The paper says best spike F1; the repo has no selection script.
- Whether the published tables used β=0.5 (the paper) or 0.6 (the code's default).
- The full texts of the RSR (2026), SMU (2024) and CLOUDADV (2026) papers; I read only their abstracts. Check them before claiming novelty for #1 and #2.

## Sources
- Paper: Agulló et al., FGCS 170 (2025) 107833, doi:10.1016/j.future.2025.107833
- Code: https://github.com/FerranAgulloLopez/ResourceForecasting
- RSR conformal right-sizing: https://arxiv.org/abs/2607.24773
- Predictive scaling heuristic and pinball framing: https://arxiv.org/pdf/2008.01215
- SMU robust predictive autoscaling: https://ink.library.smu.edu.sg/sis_research/9264
- CLOUDADV: https://arxiv.org/pdf/2606.31470
- Predictive autoscaling survey (2026): https://arxiv.org/pdf/2606.07046
- Uncertainty-aware forecasting and energy trade-off: https://cora.ucc.ie/items/3bc19eda-3313-49d6-aa6c-e5a9c6b30e8b
- Extreme value theory for capacity: https://web.stanford.edu/%7Egorin/papers/TSC_2016.pdf ; https://pp.bme.hu/eecs/article/view/17671
- Burst prediction with lead time: https://web3.arxiv.org/abs/2308.10311 ; GEAP: https://www.springerprofessional.de/geap-a-generic-approach-to-predicting-workload-bursts-for-web-ho/2286496
- Alibaba microservice traces: https://research.spec.org/icpe_proceedings/2024/companion/p82.pdf
- Autopilot: https://research.google/pubs/autopilot-workload-autoscaling-at-google-scale/
