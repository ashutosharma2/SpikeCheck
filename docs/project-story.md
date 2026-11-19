# SpikeCheck: the project story (SOART)

Scenario, Obstacle, Action, Result, Troubleshooting. This is how SpikeCheck was planned and done, the decisions behind it, and what went wrong along the way. The reasoning behind each decision is in the [ADRs](adr/).

**Status:** in progress. The review, the replication, both three-seed horizon experiments, the spike-type split, the computed-hold test, the per-trace selection checks, the simulated autoscaler and the Strategy 2 test are done, and the earlier results have been regenerated from the committed scripts. The next steps are in [NEXT.md](../NEXT.md).

---

## S: Scenario

**A published method claims to forecast CPU spikes twice as well. Does it, and does that help anyone provision cloud resources?**

Cloud platforms forecast how much CPU each workload will need, so they can lend spare capacity to other work without starving anyone. Most forecasters are trained to be right on average, which makes them late or too low exactly when a workload spikes. Agulló et al. (2025) propose changing what the forecaster is trained to predict so that it sees spikes coming, and report that this roughly doubles spike-prediction scores.

I wanted to find a question the paper left open, and to check its claims on real data before building on them. Two constraints shaped the work. I had one laptop: an 8 GB M1 MacBook with 18 GB of free disk and a 4 MB/s connection, against a 29 GB trace. And I'm new to Python, PyTorch and this kind of evaluation, so every step had to be something I could explain. The replication covers Strategy 1 with TempInc on Alibaba and a later test of Strategy 2 at one setting per variant, not the paper's Google data or its other five models ([scope and why](replication.md#scope-one-part-of-the-paper-and-why)).

## O: Obstacles

1. **The paper's numbers couldn't be taken at face value.** Its over-provisioning figures run from 10¹² to 10¹⁸, and five of its six Alibaba models report exactly the same value. Nobody had checked its spike metrics against what provisioning needs.
2. **The data was bigger than the disk.** The Alibaba container trace is a single 29 GB archive (176 GB uncompressed), and the laptop had 18 GB free.
3. **The preprocessing wasn't public.** The authors' code reads a dataset that was already prepared, and the steps that made it are only described in words.
4. **Fair comparisons are hard.** Any forecaster covers more spikes if it asks for more CPU, and the paper scores as if resources can be added instantly.
5. **There was only 8 GB of memory to train in.** The first version of the training script peaked at 3.9 GB and pushed the laptop into swap.

## A: Actions

### Audit (22 October)
I read the paper and the authors' evaluation code side by side, and found that over-provisioning divides by about 1e-10, that spike precision counts forecasts that lag, and that the spike detector fires on noise right after a spike. The [review](review.md) has the details.

### Replication (22 October)
- I streamed the trace over HTTP straight into a parser that kept only the selected containers and summed them into 5-minute steps, so the 29 GB file never touched the disk: 4.0 billion rows read, 588 million kept.
- I rebuilt the preprocessing from the paper's description and wrote down every threshold ([ADR-005](adr/005-rebuilt-preprocessing.md)).
- I trained the authors' own model on the paper's target and on the plain next value, changing nothing else ([ADR-004](adr/004-change-one-thing-at-a-time.md)).
- I scored everything with corrected metrics and with the authors' evaluator side by side ([ADR-001](adr/001-corrected-metrics.md)), at three actuation delays ([ADR-002](adr/002-score-under-actuation-delay.md)) and at equal over-provisioning ([ADR-003](adr/003-compare-at-equal-over-provisioning.md)).

### Horizon experiment (22 to 23 October)
The paper's target does two things at once: it looks 25 minutes ahead, and it keeps the allocation high for 30 minutes after a spike. A third target that only looks ahead separates the two. I trained each target with three seeds, overnight.

### 30-minute look-ahead (23 October)
When the delay is 30 minutes, the 25-minute look-ahead's window stops one step short of the spike, though nearby values can still reach 80% of it. A target that looks 30 minutes ahead tests whether matching the look-ahead to the delay brings coverage back. Three seeds again.

### This repository (23 October)
Code, results and docs, with scripts that fetch the data and the authors' code ([ADR-006](adr/006-data-and-upstream-outside-the-repo.md)), and a history dated by when each file actually existed ([ADR-007](adr/007-true-date-history.md)).

### Rebuild (27 to 28 October)
macOS cleared the temporary folder that held the working data, along with several trained models and their predictions. I re-streamed the trace with the repo's own scripts, checked that the rebuilt inputs were byte for byte the same as a saved copy, retrained the lost models with the committed `train.py` and regenerated every result folder. Three of the old models had been trained with an earlier version of the script, so I replaced their results instead of keeping numbers nobody could reproduce ([ADR-008](adr/008-publish-reproducible-results.md)). What changed is in the [replication write-up](replication.md#rebuild-every-result-regenerated-from-the-committed-scripts-28-october-2026).

### Spike types (28 October)
An overall score can't say where a method gains. I sorted every test spike into unknown, seasonal, follow-on or isolated, and fixed the definitions before scoring any method per type ([ADR-009](adr/009-spike-types.md)). I matched over-provisioning overall, split coverage by type at six label settings, and added a 95% bootstrap over whole test traces on top of the seed ranges. Then I split follow-on spikes by whether the previous peak alone would cover them. The [write-up](replication.md#follow-up-which-spikes-does-each-method-gain-on-3-seeds-run-28-october-2026) has the tables.

### Computed hold (3 November)
The hold costs coverage overall but leads on follow-on spikes, and the paper's target splits exactly into a part already in the model's input and fwdmax's target. So I tested computing the hold instead of learning it. I chose this route because it separates deterministic historical information from uncertain future predictions. Before opening the test set I wrote down the rule: better overall, no more than 1 point worse on follow-on spikes, at 5 and 15 minutes, with all four checks required and every mixed outcome defined. I pushed the rule and the scorer, dry-ran it on validation, and ran the test once ([ADR-010](adr/010-computed-hold.md)).

### Per-trace policy selection (13 November)
Spike types prefer different policies, so the next idea was a selector that picks one policy per workload from a fixed library: a rule, a trained classifier or a large language model. Before building one I ran two checks on the validation traces, each with its rule and a prediction written down first ([ADR-011](adr/011-per-trace-policy-selection.md)). The first measured how much a per-trace choice could gain with hindsight, and how much of that carries over from the first half of a trace to the second. The second trained a small decision tree to make the choice from the first 43 hours of each trace.

### Simulated autoscaler (14 November)
Every comparison so far set the allocation to the forecast from a few steps earlier. A real autoscaler also waits before scaling down and ignores small changes, and that wait is itself a hold. So I tested whether the computed hold still throttles less than the learned hold in a simulated autoscaler with a 5-minute scale-down window and a 10% tolerance. The rule and a prediction were written down and pushed first, the simulator had to reproduce the published replay with both switched off, and the test ran once ([ADR-012](adr/012-autoscaler-simulation.md)).

### Strategy 2 (14 to 18 November)
Everything so far tested the paper's first idea, changing the target. The second keeps the plain target and changes the loss, spreading each training label over the classes above the true one. I trained its four variants with three seeds each, one setting per variant fixed in advance, and compared them with the plain classifier read along its quantile curve at equal slack. The rule and a prediction were written down and pushed first, the comparison models had to reproduce the published tables before any variant was scored, and the test ran once ([ADR-013](adr/013-strategy-2.md)).

## R: Results

- **The paper's result reproduces.** On the authors' evaluator, spike F1 goes from 0.41 to 0.68 (the paper reports 0.27 to 0.64). The over-provisioning artefact reproduces too: every forecaster scores between 10¹⁵ and 10¹⁷.
- **Which method wins depends on the delay.** With no actuation delay, a plain classifier read at an upper quantile matches the paper's recall with 14 to 29% less over-provisioning across three seeds. With a 15-minute delay the paper's method wins, with coverage of 0.57 against 0.43.
- **Overall, looking ahead helps and holding hurts.** A target that only looks ahead beats the paper's at every level of over-provisioning, in all three seeds: 8 more points of spike coverage and 18% less throttled CPU at slack 0.6. Holding the allocation after a spike spends budget on spikes that are already over. The exception is follow-on spikes, below.
- **The paper's metric can't see this.** On its recall window the two targets look the same. The difference only shows once delay or throttled CPU is measured.
- **The look-ahead has to match the delay.** Even a perfect forecaster of the paper's 25-minute target covers only 28% of spikes when the delay is 30 minutes. A 30-minute look-ahead brings it back: 0.74 coverage at slack 0.8 against 0.59 for the 25-minute target, in all three seeds, with no measurable cost at 5 or 15-minute delays.
- **One overall score hides where each method gains.** At a 15-minute delay and slack 0.8, the look-ahead beats the paper's target on seasonal (+0.030) and isolated (+0.045) spikes and trails it on follow-on spikes (−0.084). That follow-on result belongs to the 6-step definition: at 12 steps the 5-minute sign flips, and what that means is still open.
- **Computing the hold passes three of four pre-set checks.** Compared with the learned hold at slack 0.8, it covers more spikes overall (+0.011 and +0.008) with less throttled CPU (−5.4% and −5.7%) at 5 and 15 minutes. Follow-on coverage passes at 5 minutes and fails the 1-point margin at 15, so the complete hypothesis isn't supported and the conclusion is delay-specific. Validation split the other way, so this isn't a stable difference between the delays. No hold still has the best overall coverage.
- **Choosing a policy per trace showed no usable signal.** On validation at a 15-minute delay and slack 0.8, a per-trace choice made with hindsight gains 0.027 of coverage over the best fixed policy. The choice made on the first half of a trace loses 0.010 on the second half, and a small trained decision tree loses 0.018. Both pre-set rules said stop, so I dropped the idea for this library. One naive rule and one small tree were tested, so this doesn't show that no selector could work.
- **The computed hold's throttling gain carries over to a simulated autoscaler.** With a 5-minute scale-down window and a 10% tolerance, at slack 0.8, the computed hold throttles 49.3 core-hours less than the learned hold at a 5-minute delay (5.8%) and 61.6 less at 15 (6.3%), with both intervals wholly below zero and all three seeds agreeing. The test wasn't blind, it is still a replay of recorded usage, and no hold still throttles least.
- **One of three altered losses adds coverage at equal slack.** At a 5-minute delay and slack 0.8, exponential gains 0.030 over the plain classifier [+0.024, +0.034] and is ahead on all three seeds. Proportional isn't shown to add anything and binary is worse. Single, the same loss with no upward spread, gains 0.023, so most of exponential's gain doesn't need the spread. Single's seeds run from −0.002 to +0.062, and the plain classifier's own seeds differ by 0.045, so I can't say its gain would return on a new training run. On the authors' evaluator single scores 0.407 and the plain classifier 0.417, so the paper's gap of 0.387 against 0.275 didn't reproduce. Each variant was tried at one setting.

## T: Troubleshooting

| Problem | Cause | Fix | Lesson |
|---|---|---|---|
| A 29 GB download onto 18 GB of free disk | The trace is a single archive | Stream it and aggregate as it arrives, keeping only 5-minute sums for the chosen containers | Process large data as it arrives instead of storing it |
| Parallel downloads were no faster | Total throughput stayed at about 4 MB/s however the download was split | One resumable stream that reconnects from the last byte | Measure throughput before designing around it |
| Preprocessing kept zero traces | Usage timestamps start at about 86,400 s, so day 0 of the window was always empty | Move the 8-day window to steps 288 to 2,591 | Check where the data starts, not where the schema says it does |
| Evaluation crashed with "File is not a zip file" | The pipeline was rewriting `data.npz` while evaluation read it | Wait for preprocessing to finish before starting anything that reads its output | Two jobs that share a file need an order |
| Training pushed the laptop into swap | Several full copies of the 440 MB dataset were in memory at once | Build each split once, memory-map the inputs and store them at half precision: peak memory went from 3.9 GB to 2.5 GB | Measure peak memory, then remove copies before shrinking anything else |
| Overnight runs stopped early | The run was tied to the terminal that started it, and idle sleep paused it | Start it as a detached process under `caffeinate -i` | A long job has to outlive whatever launched it |
| A stop command missed the training process | The match pattern used the script's full path, but the process had been started with a relative one | Stop it by process ID | Check that a stop command stopped something |
| The quantile curves barely overlapped | The first quantile grid (0.5 to 0.99) was too narrow | Widen it to 0.1 to 0.995 | Make sure a comparison has data on both sides before reading it |
| The first run's `train.py` no longer exists | It was changed for seeds and memory before the repo existed | Leave it out of the dated commit and say so ([ADR-007](adr/007-true-date-history.md)) | Commit early; a lost history can't be rebuilt |
| A draft ADR quoted a number that wasn't in our data | It came from memory of the paper, not from our results | Check every figure in the docs against the result files before committing | A number in a doc needs a source file |
| The working data and several trained models disappeared | macOS deletes files in temporary folders that haven't been opened for a few days, and the data lived in one | Keep data in the repo's git-ignored `data/` folder, re-stream the trace and retrain the lost runs | Keep anything expensive to rebuild out of temporary folders |
| Three retrained models didn't match the published ones | cbase, proposed and proposed_full had first been trained with the earlier float32 `train.py`, which no longer exists. fwdmax seed 0, first trained with the current script, came back identical | Retrain with the committed script and regenerate every result ([ADR-008](adr/008-publish-reproducible-results.md)) | Publish only results the committed code can reproduce |
| A run note said the old seed-0 runs fell inside the spread of the other seeds | The claim wasn't checked at every delay: with no delay, the old proposed seed 0 was below both other seeds | Correct the note and the conclusions that leaned on it | Check a claim about the data against every slice it covers |
| My first follow-on window of 4 steps would have been empty | The detector's 4-step refractory period means no two spikes start closer than 5 steps | Use 6 steps, the length of the paper's hold | Check a definition against the detector that produces the data |
| Spikes with no match a day earlier looked isolated | The detector starts at step 762, so 5,916 spikes have no complete day of history | Label them unknown and leave them out of type comparisons | Missing history isn't evidence |
| The first ±2 and ±3 seasonal match rates were slightly off (37.3% and 42.3%) | They came from a check outside the repo, on a larger set of spikes than the ±1 rate | Regenerate every ADR-009 count with `spike_types.py`, which gives 37.6% and 42.6% | Counts belong in committed code |
| I first said the previous peak "explains much" of the hold's lead | 87% of follow-on spikes sit in the group the peak covers, so most of any lead lands there by size | Compare the lead per spike, and say the split can't give a share | Compare rates before counts |
| I didn't write predictions before scoring per type | I fixed the labels first and stopped there | Report these results as description, and write the prediction first in the next test | Fixing a definition isn't fixing a prediction |
| The docs said the 25-minute look-ahead "misses every spike" at a 30-minute delay | I reasoned from the window instead of checking the score: it still covers 0.59 of spikes. An outside review caught it | Say the window stops one step short and nearby values can still pass | Check a claim against the number it implies |
| The docs said the hold hurts, without saying overall | The first experiments only had overall scores, and the per-type split later showed the hold leading on follow-on spikes | Add "overall" and point to the follow-on result | A conclusion holds for the slice it was measured on |
| The first computed-hold comparison showed +0.027 overall at 15 minutes | It used fwdmax6, which looks one step further, so it changed the look-ahead as well as the hold | Compare fwdmax with and without the hold, which gave +0.013 | Change one thing, even in a quick check |
| The exact hold wasn't a fair comparison with a model's output | Models output bin edges. Rounding up left 22% of hold values sitting on an edge, where a model's encoding sends them to the next bin | Encode the hold exactly as the target is encoded, which took the gain to +0.0070 on validation | Compare like with like, down to the output's resolution |
| I first picked the follow-on margin having seen only the 15-minute validation row | The 5-minute row, which fails narrowly, was shown afterwards | Look at both rows, keep the margin for its stated reason, and record the timing in ADR-010 | See every slice before setting a threshold, and write down what you had seen |
| Absolute core-hours in the exploration scripts were 100 times too high | The CPU request wasn't divided by 100 | `score_hold.py` uses the repo's unit, and the percentages weren't affected | Reuse the committed metric instead of rewriting it |
| A pattern by spike size on validation didn't replicate on test | It was an exploratory reading with no prior reason behind it | Report it as not replicated and don't use it | A pattern found while looking needs a second look |
| The validation dry run could only use seed 0 | The trained fwdmax models for seeds 1 and 2 weren't kept, only their test predictions | Say so in ADR-010, and use all three seeds on test | Keep model files while a question is still open |
| With hindsight, choosing a policy per trace gained up to 4 points of coverage | Most of it didn't last: the policy that won on the first half of a trace wasn't the winner on the second half | Score the first-half choice on the second half before trusting the gain | A gain found with hindsight isn't a signal until it carries over |
| The prediction for the trained selector was wrong | It said the gain would be near zero. The tree fitted labels that were mostly noise and moved traces to weaker policies, so it lost 0.018 | Report the miss in ADR-011 and keep the rule's verdict | A selector trained on noise can do worse than no selector |
| The prediction for the simulated autoscaler was half wrong | It said the computed hold's gain would shrink once the controller had its own window. The gain was slightly larger as a share: 5.8% and 6.3% against 5.4% and 5.7% | Report both parts in ADR-012 as they came out | A prediction written down first shows where the reasoning was off |
| The simulator's replay numbers differed slightly from the published ones at 15 minutes | The published replay matches budgets on the forecast before the delay. The simulator matches them on the allocation it produces, because the window adds slack | Check the reproduction under the published convention, report results under the simulator's, and say so | State which budget a comparison is matched on |
| The first 12 Strategy 2 models stopped training too early | I copied the earlier runs' fixed minimum improvement of 0.0001, but the new loss is about 30 times smaller, so the bar was about 30 times stricter. In 6 of 12 runs the lowest validation loss came after the saved epoch | Make the bar a share of the best loss so far, and retrain all 12 before scoring anything | A threshold copied from another loss has to scale with that loss |
| All 12 retraining runs failed the moment they started | The runner was given a relative Python path and changes folder before using it | Make the path absolute in `train_all.sh` and start again | Watch a long job start before leaving it |
| Binary's quantile curve started at a slack of 0.77, too close to the 0.8 the rule reads | Its label spreads weight over every class above the true one, so even a low quantile asks for a lot of CPU | Also read the four variants at quantiles 0.01, 0.02 and 0.05 | Check that a curve reaches the point the rule reads |
| My first explanation for single's gain was wrong | I suspected it came from reading probabilities at a different temperature from training. A check outside the repo didn't support that | Report the cause as unknown | A plausible cause needs its own test |
| The dry run overstated single's gain: +0.065 on validation against +0.023 on test | The dry run used seed 0 alone. On test, seed 0 gains +0.062 and the other two +0.009 and −0.002 | Report every seed next to the mean in ADR-013 | A one-seed dry run checks the pipeline, not the size of a result |
| Single's interval was narrow while its seeds ran from −0.002 to +0.062 | The bootstrap resamples traces with the trained models held fixed, so it can't see differences between training runs | Keep the rule's verdict, and state the limit next to it | An interval only covers the variation it resamples |
| Two of the three Strategy 2 predictions were wrong | I expected single to match the plain classifier and no altered variant to beat it | Report both in ADR-013 as they came out | A prediction written down first shows where the reasoning was off |
