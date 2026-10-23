# SpikeCheck: the project story (SOART)

Scenario, Obstacle, Action, Result, Troubleshooting. This is how SpikeCheck was planned and done, the decisions behind it, and what went wrong along the way. The reasoning behind each decision is in the [ADRs](adr/).

**Status:** in progress. The review, the replication and the three-seed horizon experiment are done. A 30-minute look-ahead experiment is running, and the next steps are in [NEXT.md](../NEXT.md).

---

## S: Scenario

**A published method claims to forecast CPU spikes twice as well. Does it, and does that help anyone provision cloud resources?**

Cloud platforms forecast how much CPU each workload will need, so they can lend spare capacity to other work without starving anyone. Most forecasters are trained to be right on average, which makes them late or too low exactly when a workload spikes. Agulló et al. (2025) propose changing what the forecaster is trained to predict so that it sees spikes coming, and report that this roughly doubles spike-prediction scores.

I wanted to find a question the paper left open, and to check its claims on real data before building on them. Two constraints shaped the work. I had one laptop: an 8 GB M1 MacBook with 18 GB of free disk and a 4 MB/s connection, against a 29 GB trace. And I'm new to Python, PyTorch and this kind of evaluation, so every step had to be something I could explain.

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

### 30-minute look-ahead (23 October, running)
A target whose look-ahead matches a 30-minute delay, to test whether that brings coverage back at that delay.

### This repository (23 October)
Code, results and docs, with scripts that fetch the data and the authors' code ([ADR-006](adr/006-data-and-upstream-outside-the-repo.md)), and a history dated by when each file actually existed ([ADR-007](adr/007-true-date-history.md)).

## R: Results

- **The paper's result reproduces.** On the authors' evaluator, spike F1 goes from 0.42 to 0.65 (the paper reports 0.27 to 0.64). The over-provisioning artefact reproduces too: every forecaster scores between 10¹⁵ and 10¹⁷.
- **Which method wins depends on the delay.** With no actuation delay, a plain classifier read at an upper quantile matches the paper's recall with about 30% less over-provisioning. With a 15-minute delay the paper's method wins, with coverage of 0.57 against 0.44.
- **Looking ahead helps and holding hurts.** A target that only looks ahead beats the paper's at every level of over-provisioning, in all three seeds: 8 more points of spike coverage and 19% less throttled CPU at slack 0.6. Holding the allocation after a spike spends budget on spikes that are already over.
- **The paper's metric can't see this.** On its recall window the two targets look the same. The difference only shows once delay or throttled CPU is measured.
- **The look-ahead has to match the delay.** Even a perfect forecaster of the paper's 25-minute target covers only 28% of spikes when the delay is 30 minutes.

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
