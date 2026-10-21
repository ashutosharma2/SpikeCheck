# Critical review and research opportunities
## Agulló, Gutierrez-Torre, Torres, Berral — "Enhancing the output of time series forecasting algorithms for cloud resource provisioning", FGCS 170 (2025) 107833

Prepared 2026-10-22. Sources: the paper PDF (15 pp.), and the authors' public code at
`github.com/FerranAgulloLopez/ResourceForecasting` (commit `98f0d51`, 16 Apr 2025).

**Labels used below.** **[Paper]** means the authors state it. **[Code]** means I checked it in their
repository. **[Inference]** means it is my interpretation and has not been verified experimentally.
**[Lit]** means it comes from a literature search done on 2026-10-22, which was not exhaustive.

---

## 0. Bottom line

1. **The central idea holds up and is useful.** Training a model on a provisioning target (a forward/backward max-envelope) instead of the raw series is sound, cheap and model-agnostic. That part is defensible.
2. **The over-provisioning numbers are numerical artefacts.** In the code, values are shifted so that the test set's global minimum equals **1e-10** (`evaluation.py:27`). Over-provisioning then divides by that value (`:83`). One point therefore dominates the metric, giving magnitudes of 1e12–1e18, and 5 of 6 Alibaba models get *identical* values (Table 3: 4.8636E+12 for "Proposed" and 1.9181E+12 for "CBaseline"). The paper's headline trade-off ("better spikes *at the cost of* more over-provisioning") is **not actually measured**.
3. **The spike metrics have structural biases** that favour hold-high envelope forecasts and lagging forecasts. Details are in §2-D.
4. **Strategy 2 (loss modification) is not supported by its own evidence.** The "single" variant is mathematically close to plain cross-entropy, yet it beats CE by about 40% in F1 (Table 4). That gap must come from tuning, selection or seed noise, not from the target alteration. The authors flag this concern themselves (p. 9).
5. **The strongest open research gap:** nobody has shown whether spike-oriented forecast metrics, or the gains reported here, translate into lower throttling and SLO cost once actuation delay is included. It is also untested whether a calibrated quantile/conformal forecast of the forward-window max gives the same benefit with one interpretable knob. Section 4 ranks seven opportunities; #1–#3 can be combined into one strong paper.

---

## 1. Phase 1 — What the paper actually contributes

### 1.1 Technical map

| # | Contribution | Mechanism (where) | Key assumptions | Evidence offered | Unresolved |
|---|---|---|---|---|---|
| C1 | **Distinction between series** (values vs target) | Input = past-only transform (log + max-past w=2); target = log + max-future w=5 + max-past w=7, then discretised (§2.1, Fig. 6; config confirms [Code]) | Provisioning needs an anticipatory, sustained envelope rather than the point value; transforms "must be revertible" (p. 4) | Table 3: spike-F1 is roughly 2.2× the best baseline in 11 of 12 model×dataset cells | The max-filter is *not* revertible, which contradicts the paper's own requirement. Values and target are changed together, so gains cannot be attributed (no ablation). Only one configuration was tested. |
| C2 | **Loss-function modification** (altered ground truth in KD-style loss) | Eq. 1: α·KLD(M(x)/T, altered_y/T) + (1−α)·CE; variants single/binary/proportional/exponential (§2.2, §4.3.2) | Asymmetric soft labels make the classifier lean upward | Table 4 (TempInc only): F1 0.39–0.40 vs CE 0.27 (Alibaba) | "Single" ≈ CE but scores 0.387. Eq. 1 is missing its "+". The code applies softmax to the target (`loss_functions.py:61`), which Eq. 1 does not show. α was chosen by best spike-F1, and the split used for that choice is unstated. |
| C3 | **New evaluation**: spikes-precision/recall, over/under-provision | Alg. 1–2, Eqs. 2–5, constants γ=250, σ=2, α=0.2, β=0.5, τ=4 (§3, §4.1.4) | Statistically defined spikes approximate operationally harmful events; relative error is the right scale | Table 2: oracle "max-future" filter gets prec/rec ≈1 but the worst MSE | Construct validity (does it track SLO cost?). Sensitivity to constants. Numerical conditioning (see D1). |
| C4 | **Explainability** | DeepLift-SHAP on TempInc; KMeans + t-SNE on the heaviest 2000/7000 *correctly predicted* spikes (§4.4, Figs. 9–11) | Attributions are faithful; clusters are meaningful | Four "reasons": last sample spiked, recent spike, recent variability, daily seasonality | Survivorship (only hits are analysed). No faithfulness test. Three of the four reasons are reactive (see E). |

### 1.2 Problem, questions and hypotheses
- **Problem [Paper, §1.1]:** mean-oriented forecasters predict spikes late or too low. In co-located clusters (Borg, Alibaba), a high-priority spike may find its paid-for resources lent to low-priority work, which causes throttling and SLO violations.
- **Implicit hypotheses [Inference]:** H1: changing the training target can make a forecaster *anticipate* spikes at the right height. H2: this costs a controllable amount of over-provisioning. H3: MSE-style metrics rank forecasters wrongly for this goal. H4: model capacity matters (deep > simple).
- **Scope [Paper, §4.1.1]:** one-step-ahead CPU at 5-min granularity; univariate target with multivariate inputs (14 Google / 18 Alibaba features).

### 1.3 Experimental set-up (as reported plus what the code shows)
- **Data [Paper §4.1.2, Table 1]:** Google 2019, cluster *a* only, production tier, >15 days, one instance per workload, then **a PCA cluster of "common patterns"** (1,722 traces). Alibaba 2018 long-running containers, with **low-variability traces removed** (5,168 traces). Both are resampled to 5 min and standardised.
- **Split:** 70/15/15 **random by trace** [Paper p. 7]. Every split covers the same calendar period.
- **Discretisation:** the paper says "40 equal intervals over dataset min–max". The Alibaba Proposed config has **45 boundaries** (0.25…11.25), which are log-transformed [Code].
- **Decoding [Code `full_values_unbiased_intervals.py:117`]:** argmax class → **upper edge of the bin**. Every classification variant (CBaseline and Proposed) is therefore biased upward by up to one bin. The paper does not disclose this.
- **Models:** RF, *linear* SVM (LinearSVR/LinearSVC, not kernel), DLinear, SCINet, Informer, TempInc (own model). RF/SVM use full features only at lag 32 (memory limits), so the cross-model comparison is uneven [Paper App. A].
- **Tuning:** small grids (lag × lr, plus a few architecture knobs). The selection criterion is not stated. Each config uses `manual_seed: 0` [Code], so results are **single-seed with no variance reported**.
- **Unreported baselines:** the repo implements **Autopilot** and **Theta-scan**, the senior author's own vertical autoscaler (`app/factories/method_factory.py`). Neither appears in the paper and neither has experiment configs [Code].

### 1.4 Principal results [Paper]
- Table 2: MSE ranks the oracle envelope worst, while the spike metrics rank it best.
- Table 3: Proposed raises spike-recall a lot and lowers under-provisioning. Precision is mixed. SVM-Google is the one failure (F1 0.13 vs 0.20).
- Table 4: Strategy 2 improves F1 over MSE/CE. "Single" is about as good as the altered variants.
- Best model: TempInc (Alibaba F1 0.64, Google 0.80). DLinear is close to the deep models on Google (0.66).

---

## 2. Phase 2 — Adversarial methodological audit

Severity: **Critical** (invalidates a stated conclusion), **Major** (materially weakens it), **Minor**.

### A. Problem formulation

**A1. Spike prediction is never connected to provisioning outcomes (Major).** No autoscaler, scheduler, throttling model or SLO signal is simulated. "Better spike-F1 ⇒ fewer SLA violations" is assumed (§1.2, §6). CPU is *compressible*: under-provisioning causes throttling (latency), not failure. The cost of a miss therefore depends on how deep and how long the shortfall lasts, which an event-count metric cannot capture.

**A2. Actuation delay is not modelled (Major).** One step is 5 min. The recall window allows a hit anywhere in [k−4, k] (τ=4), including **the forecast for time k itself, issued at k−1**. Whether 5 minutes is enough lead time depends on the actuator. Reclaiming CPU shares from best-effort co-tenants takes seconds. Vertical resizing with restart, or VM scale-out, takes minutes. The paper argues from lead time ("leaves no time for corrective actions", §1.3.2) but never parameterises it.

**A3. The asymmetric cost is ad hoc (Major).** Asymmetry appears in three uncoordinated places: α≠β in the tolerance band, the max-filter target, and bin-upper-edge decoding [Code]. None is derived from a cost ratio. The paper promises users a "trade-off knob" but gives no mapping from knob (filter width, α, T, slope) to an operating point.

**A4. Regime change and unseen magnitudes are structurally excluded (Major).** The classification head has a fixed range. The code **raises an exception** for out-of-range targets (`full_values_unbiased_intervals.py:92`), so a spike above the training range cannot be represented. Random trace-level splits over one shared time period also hide calendar-level novelty.

### B. Data and generalisation

**B1. Selection bias by construction (Major).** Google: one cluster, prod tier, >15 days, plus a PCA cluster of similar traces. Alibaba: low-variance traces dropped. Results therefore describe a homogeneous, spike-rich subpopulation. Fleet-level over-provisioning cost, which is dominated by the many flat traces, is not represented.

**B2. No temporal or cross-cluster holdout (Major).** The trace-level random split means test traces share **the same wall-clock window** as training traces. Cluster-wide events such as deployments, diurnal peaks and incidents correlate across containers, so training sees "future" cross-sectional information. Google 2019 has **8 clusters** and Alibaba is 8 days, so cross-cluster and time-forward tests were easy to run and were not done.

**B3. Possible preprocessing leakage (unverified).** Standardisation and the min–max range for binning ("of the corresponding dataset", p. 7) appear to use all traces. The offline preprocessing scripts are not in the repo, which only reads `third_preprocess_compressed_standarized`. The repo also contains `replace_downward_spikes_with_previous`, which uses full-trace mean/std (`time_series_preprocessing.py:289`). I found no config that calls it, so its use is unconfirmed. **Severity depends on this, so I rate it Minor to Major.**

**B4. Representativeness (Minor–Major, depending on claim).** 2018/2019 traces predate the large share of serverless, GPU/LLM-serving and microservice workloads. Alibaba microservice traces (2021/2022) include **response time and call rates** [Lit], so QoS-grounded labels are available today.

### C. Forecasting methodology

**C1. Gains are confounded (Major).** "Proposed" changes the input transform (log + max-past-2) and the target (max-future-5 + max-past-7) together, and the transforms were picked by "an initial parameter search" (p. 8) on an unstated split. There is no ablation (target-only, input-only) and no sensitivity sweep over window widths, even though widths are claimed to be the user's knob.

**C2. The target is aligned with the metric (Major) [Inference].** The target max(x_{t−7…t+4}) is close to the oracle envelope in Table 2, and the recall tolerance τ=4 matches the 5-step future window. Strategy 1 therefore largely trains the model to imitate the oracle the metric was designed to reward. That is legitimate as goal-alignment, but it means the headline gain partly reflects **metric–target co-design**, not general forecasting skill.

**C3. Missing strong, cheap baselines (Critical for the "deep models are needed" claim, §5).** None of these is reported:
- rolling max of the past W samples × (1+m), which is Autopilot's moving-window peak recommender; the code exists in the repo;
- the seasonal-max envelope max(x_{t−288±k});
- a quantile regressor (pinball loss) on the *same* target;
- **the CE classifier from CBaseline, decoded at an upper quantile instead of argmax**, which needs no retraining.

The persistence filter already reaches recall 0.15/0.20 (Table 2), so a rolling-max envelope may recover much of the reported gain. Until it is tested, the claim that "complex models are required" (§5) is unsupported.

**C4. Strategy 2 is internally inconsistent (Major).** With one-hot targets and softmax(target/T) where T=0.1 [Code `loss_functions.py:61`], the KLD term is essentially CE on temperature-scaled logits. "Single" ≈ CE, yet F1 is 0.387 vs 0.275 on Alibaba. There is also no expected shift under argmax: an upward-decaying soft label keeps its mode at i*, so the predicted mode moves only through uncertainty mixing. **The alteration is an indirect, uncalibrated way to obtain an upper quantile.** A direct quantile readout does this exactly.

**C5. Failure modes not analysed (Major).** Isolated, first-in-burst spikes are never separated from follow-on, ramp-preceded or seasonal spikes. The explainability results, three of whose four "reasons" are reactive, suggest most hits are follow-on or seasonal. See Opportunity #3.

### D. Evaluation methodology

**D1. Over/under-provisioning is numerically degenerate (Critical) [Code].**
`move_over_0` shifts all values by |min|+1e-10, so the test set's global minimum becomes exactly 1e-10 (`evaluation.py:27`). Eq. 4 divides by x[t] (`:83`). At that one point, a prediction one bin (≈0.25) above gives (0.25/1e-10)² ≈ 6×10¹⁸. Averaged over ≈1,800 points and ≈775 test traces, that is **≈4×10¹²**, the same order as Table 3's Alibaba entries. This explains:
- the absurd magnitudes (1e12–1e18);
- why 5 of 6 Alibaba models report *identical* over-prov (all classifiers emit the same bin at that point);
- why all four Table 4 variants share 1.9181E+12.

As a result, **over-prov in Tables 2–4 carries almost no information about over-provisioning across the trace**. The trade-off claims in the Abstract, §4.3.1 and §5 are unsubstantiated as measured.

Separately, "relative" error is computed in **shifted z-space**. Its origin depends on which traces landed in the test set, so it is not relative to CPU cores, and the paper's justification ("a node can support a larger spike if the workload has greater consumption", p. 5) does not apply.

**D2. The tolerance band depends on an arbitrary origin (Major) [Code].** The band x_k(1−α) < y' < x_k(1+β) is applied to shifted z-values. Spikes in traces near the dataset minimum get a vanishingly narrow band, while high-level traces get a wide one. Recall is therefore confounded with each trace's absolute level. The code default β is **0.6**, not the 0.5 in the paper (`evaluation.py:202`), and the training pipeline calls evaluation with defaults.

**D3. The spike detector is degenerate right after a spike (Major) [Code].** After a detection at j, the lag window becomes [j, k) (`evaluation.py:63`). At k=j+1 the window has one element and std=0, so **any increase counts as a spike**. Rising ramps therefore count as several spikes in x. The same rule rewards flat, held-high forecasts, which produce no further "predicted spikes" and so cannot lose precision.

**D4. Precision rewards lagging forecasts (Major).** The precision search window is symmetric (±τ). A forecast that simply copies a spike one step late is "related" to it. The persistence filter scores precision 0.9987/1.000 (Table 2). Precision cannot tell anticipation from reaction.

**D5. Over-provisioning plateaus are not penalised by any working metric (Major) [Inference].** Precision only counts *rising edges* of y′, and over-prov is broken (D1). Sustained over-allocation, the main cost of Strategy 1, is therefore invisible in the evaluation.

**D6. Aggregation (Minor–Major) [Code].** F1 is computed per trace and then averaged (`:340`). This is why the reported F1 ≠ the harmonic mean of the reported P and R (e.g., Alibaba RF RBaseline: P 0.80, R 0.066 gives 0.12, but 0.089 is reported). A trace with **no real spikes gets recall 0** (`:187`), and one with no predicted spikes gets precision 0 (`:143`). Correct silence is scored as failure, which pushes models towards emitting spikes. No CIs, no seeds, no significance tests.

**D7. No sensitivity analysis on (γ, σ, α, β, τ) (Major).** Rankings could flip, especially τ relative to the target window (C2) and the sampling interval.

**D8. Motivation experiment is partly circular (Minor).** Table 2 shows MSE disagrees with an envelope objective, which is expected because they estimate different functionals. It does not show the new metric tracks *operational* cost. A pinball loss at a high quantile or an explicit cost function would be the right comparator.

### E. Explainability
- **Survivorship (Major):** only *correctly predicted*, largest spikes are clustered, so the method cannot explain misses.
- **No faithfulness check (Minor–Major):** no deletion/insertion curves and no reference-baseline choice for DeepLift. Cluster count and t-SNE readings are qualitative.
- **What it actually supports [Inference]:** patterns (a) last sample spiked, (b) recent spike and (c) recent variability are **reactive or persistence cues**. Together with the max-past-7 target, this suggests a large share of "anticipated" spikes are follow-ons within bursts. Pattern (d), daily seasonality, is real anticipation but is captured by a seasonal-max heuristic. The analysis **undercuts** the paper's claim that deep models are needed (§5) more than it supports it.

### F. Reporting and reproducibility (Minor)
Eq. 1 is missing its "+". The appendix labels are swapped (text says App. A = TempInc, but App. A is hyperparameters). RF is described as a "Regressor with gini". The paper says 40 intervals while the config has 45. The paper calls the models SVM when they are linear. The preprocessing pipeline is not in the repo. On the positive side, the public code made most of this audit possible.

---

## 3. Phase 3 — Novelty landscape (search on 2026-10-22, not exhaustive)

| Direction | What exists [Lit] | Gap status |
|---|---|---|
| Conformal prediction for cloud workloads | RSR (Majd et al., arXiv 2607.24773): bootstrapped conformal PIs for VM right-sizing, medium/long-term, backtesting; it calls CP "underexplored" in cloud | **Open:** online/adaptive conformal on a *forward-window max* with an under-provisioning guarantee and spike-event evaluation |
| Quantile / asymmetric-loss provisioning | Pinball-loss framing (arXiv 2008.01215); SMU 2024 robust predictive autoscaling with quantiles; SLO-aware asymmetric LSTM loss | Partially explored. **Not** compared against target-shaping, and not evaluated on spike events |
| Decision-grounded evaluation | Carraro et al. 2023 (model rankings shift under energy cost); CLOUDADV (arXiv 2606.31470), decision-aligned sizing; 2026 survey calls for operational metrics | **Open for spike metrics:** no metric-validity study testing whether spike-F1, MSE, pinball or CRPS rankings agree with throttling/SLO cost under actuation delay |
| Spike/burst onset vs recurrence | GEAP (2014) event-driven bursts; HPC I/O burst classification with lead time (arXiv 2308.10311) | **Apparently open:** no stratified study of first-in-burst vs follow-on vs seasonal spike predictability in cloud CPU traces |
| EVT for peaks | Shenoy/Gorinevsky TSC 2016; Bozóki & Pataricza 2018/2022 | Explored; low novelty alone |
| Follow-ups to this paper | None found | Low-competition space (paper is ~18 months old) |

---

## 4. Phase 4 — Ranked research opportunities

Ranking criteria: novelty × defensibility × cost × likelihood of a clean result.

### #1 — "Do spike metrics measure what provisioning needs?" A decision-grounded validity study ★★★★★
- **Question:** across a population of forecasters, do rankings by MSE, pinball@q, CRPS, spike-F1 (as published) and a corrected event metric agree with rankings by **operational cost** under actuation delay d?
- **Why it is unexplored:** this paper and its predecessors validate metrics only on hand-made filters (Table 2). Decision-focused work in cloud [Lit] targets sizing cost, not spike metrics.
- **Design:** trace-replay simulator. The limit at t is set from the forecast issued at t−d, with d∈{0,1,3,6}. Unserved CPU = max(x−limit, 0) gives throttled core-seconds; slack = max(limit−x, 0) in core-hours. Cost = slack + λ·throttle + μ·SLO-minutes (where an RT proxy exists). Add a co-location mode: low-priority work borrows slack, and reclamation takes r seconds.
- **Forecaster zoo:** the paper's six models × {RBaseline, CBaseline, Proposed}, Autopilot (already in their repo), Theta-scan, rolling-max, seasonal-max, quantile GBM, and conformal envelope (#2).
- **Output:** Kendall-τ heatmaps (metric × λ × d); show where spike-F1 is valid or misleading.
- **Risk:** it may turn out spike-F1 correlates well. That is still publishable as validation. **Venue:** ICPE, SoCC, CCGrid, FGCS, TPDS.

### #2 — Calibrated forward-max envelopes: one knob with a guarantee ★★★★★
- **Reframe:** the paper's target series is an estimate of the functional M_t = max(x_{t−w…t+h}). Forecast its **conditional q-quantile** directly, using pinball loss or a quantile readout from the existing CE softmax. Then wrap it in **adaptive conformal inference** (Gibbs & Candès 2021) per trace, so the long-run under-provisioning (exceedance) rate is ≤ ε even under drift.
- **Hypothesis:** at matched over-provisioning in core-hours, the conformal envelope matches or beats Proposed on corrected spike-recall and dominates it on throttle cost. ε replaces filter width, α, T and slope with a knob operators can read.
- **Cheapest decisive experiment:** decode CBaseline's softmax at q∈{0.5,…,0.99} instead of argmax and plot the Pareto front against Proposed and Strategy 2. If the curve contains their points, **Strategy 2 is subsumed and Strategy 1 reduces to choosing a quantile**.
- **Novelty:** conformal for cloud is explicitly underexplored [Lit]. Forward-max + ACI + event evaluation was not found.

### #3 — Spike taxonomy and the predictability ceiling of CPU-only history ★★★★☆ (cheapest, run first)
- **Question:** which spikes does Strategy 1 actually gain on?
- **Strata:** (i) follow-on, with a prior spike within W; (ii) seasonal, with a spike near t−24h; (iii) ramp-preceded; (iv) isolated onset.
- **Hypothesis H3:** gains concentrate in (i)–(ii) and are matched by rolling-max and seasonal-max heuristics, while recall on (iv) stays close to baseline for every model. The paper's own SHAP patterns and the persistence filter's 15–20% recall support this.
- **Upper bound:** estimate the best achievable recall on (iv) from univariate history, using a best-of-zoo or ensemble ceiling and conditional-entropy estimates.
- **Why it matters:** if H3 holds, the field's "spike forecasting" progress is mostly persistence. That would redirect effort to exogenous leading signals (#4). Either outcome is publishable.
- **Cost:** days, since their code and configs are public.

### #4 — QoS-grounded spike definitions and leading indicators from call graphs ★★★★☆
- **Idea:** stop defining spikes statistically (2σ above a weighted mean) and define **harmful spikes** as CPU excursions associated with response-time degradation. Alibaba microservices v2021/v2022 provide MSResource (CPU), MSRTMCR (call rate, RT) and call graphs [Lit; check 0.5% sampling and the time span].
- **Then:** use **upstream call rate** as a leading indicator for downstream CPU onsets, the stratum (iv) that history alone cannot reach.
- **Novelty:** graph-based microservice autoscalers exist (e.g., GRAF, CoNEXT 2021; not re-verified here). The QoS-labelled event definition and onset-only evaluation appear new.

### #5 — A well-conditioned, lead-time-aware event metric (fold into #1) ★★★☆☆
Proposed fixes:
- report recall@lead-time ℓ curves at decision time k−d;
- penalise one-sided lag in precision;
- give the detector a minimum lag window and refractory period;
- give spike-free traces a defined score;
- measure over-provisioning in absolute units (core-hours) or normalised by request/limit, never ÷x;
- run a sensitivity analysis over (γ, σ, α, β, τ) and the sampling interval.

### #6 — Temporal and cross-cluster robustness ★★★☆☆
Train on Google cluster *a*, test on b–h; train on days 1–20, test on 21–31. Report spike-F1 and conformal coverage decay, then show ACI (#2) restores coverage. This strengthens #2 rather than standing alone.

### #7 — Tail extrapolation beyond the training range ★★☆☆☆
Combine a classification body with a peaks-over-threshold EVT tail on residuals, to handle magnitudes the fixed-bin head cannot represent (the code raises on them). Prior EVT work exists [Lit], so novelty is modest.

### Recommended paper (combine #3 → #2 → #1)
**Working title:** *"Spike-aware provisioning forecasts are mostly persistence: a decision-grounded re-evaluation and a calibrated alternative."*

1. **Reproduce** Table 3 with their code. Fix D1–D3 and D6. Add the rolling-max, seasonal-max and Autopilot baselines.
2. **Stratify** spikes (#3) and test H3.
3. **Introduce** the quantile/ACI envelope (#2) with ε as the sole knob.
4. **Validate** everything in the closed-loop simulator (#1) across d and λ.

**Kill criteria:**
- If rolling-max does *not* recover most of Proposed's gain, H3 fails. Then pivot to explaining what the deep models learn beyond persistence, which is itself interesting.
- If the quantile readout's Pareto front does not dominate Proposed, then target-shaping offers something quantiles do not. Characterise what that is.

---

## 5. What I could not verify
- The offline preprocessing (standardisation scope, imputation, leakage). Those scripts are not in the repo.
- Which split was used to choose hyperparameters and Strategy 2's α. The paper says best spike-F1; the repo has no selection script.
- Whether the published tables were produced with β=0.5 (paper) or 0.6 (code default).
- The full texts of the RSR (2026), SMU (2024) and CLOUDADV (2026) papers; I read abstracts only. Check them before claiming novelty for #1 and #2.

## Sources
- Paper: Agulló et al., FGCS 170 (2025) 107833, doi:10.1016/j.future.2025.107833
- Code: https://github.com/FerranAgulloLopez/ResourceForecasting
- RSR conformal right-sizing: https://arxiv.org/abs/2607.24773
- Predictive scaling heuristic / pinball framing: https://arxiv.org/pdf/2008.01215
- SMU robust predictive autoscaling: https://ink.library.smu.edu.sg/sis_research/9264
- CLOUDADV: https://arxiv.org/pdf/2606.31470
- Predictive autoscaling survey (2026): https://arxiv.org/pdf/2606.07046
- Uncertainty-aware forecasting and energy trade-off: https://cora.ucc.ie/items/3bc19eda-3313-49d6-aa6c-e5a9c6b30e8b
- EVT for capacity: https://web.stanford.edu/%7Egorin/papers/TSC_2016.pdf ; https://pp.bme.hu/eecs/article/view/17671
- Burst prediction with lead time: https://web3.arxiv.org/abs/2308.10311 ; GEAP: https://www.springerprofessional.de/geap-a-generic-approach-to-predicting-workload-bursts-for-web-ho/2286496
- Alibaba microservice traces: https://research.spec.org/icpe_proceedings/2024/companion/p82.pdf
- Autopilot: https://research.google/pubs/autopilot-workload-autoscaling-at-google-scale/
