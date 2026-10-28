# ADR-009: Define mutually exclusive spike types

- Status: Accepted
- Date: 2026-10-28

## Scenario

SpikeCheck needs a reproducible way to sort real CPU spikes into types by timing, so we can see whether provisioning strategies behave differently on different workload patterns ([NEXT.md](../../NEXT.md)). The definitions have to be fixed before anyone sees a method score per type. Otherwise the thresholds could end up chosen because they favour one strategy.

## Obstacle

The test set has 27,980 real spikes across 893 traces. The median start-to-start gap between consecutive spikes is 30 steps (150 minutes). No gap is shorter than 5 steps, because the detector's 4-step refractory period never counts two spikes closer than that. Only 9.4% of consecutive gaps are 6 steps or less.

The gaps show no clear boundary between bursts and isolated spikes. They do peak at 12, 24 and 36 steps, which suggests some workloads run on an hourly schedule. Consecutive gaps also can't show daily repetition, because other spikes often come between a spike and the one at the same time the day before.

The detector starts counting at step 762: the model needs 512 steps of input, and the detector needs a 250-step rolling average. A seasonal check at ±1 step looks back as far as 289 steps, so a spike before step 1,051 has no complete day of detected history. That applies to 5,916 spikes, 21% of the test set. Finding no daily predecessor in that stretch is not evidence that a spike is isolated.

Seasonal and follow-on also overlap. Of the 6,749 spikes with a daily predecessor, 543 (8%) came within 6 steps of the spike before them. This matters when reading results for the paper's 6-step hold.

## Action

Every spike gets exactly one of four labels. The rules are checked in this order, and the first one that fits sets the label:

1. Unknown: some part of the ±1 window around 288 steps back falls before step 762. That means any spike before step 1,051, where there isn't enough history to tell whether it is seasonal.
2. Seasonal: another spike in the same trace starts within ±1 step of exactly 288 steps (24 hours) earlier. Any earlier spike counts, not only the one immediately before.
3. Follow-on: the previous spike in the same trace started 5 or 6 steps earlier.
4. Isolated: none of the above.

Because history is checked first, a first-day spike that comes right after another one is unknown, not follow-on. Follow-on means "not seasonal", and that can't be shown without a day of history. A spike with a daily predecessor is seasonal even if another spike came just before it.

How each threshold was chosen:

- The seasonal tolerance of ±1 step allows five minutes either side of the daily offset. The paper doesn't set a tolerance; this is an operational choice and is tested below.
- The follow-on limit of 6 steps matches the paper's existing 6-step hold. With 5-minute steps that's a start-to-start interval of 25 to 30 minutes. It's an operational definition, not a claim that 6 steps is a natural boundary between bursts and isolated spikes.
- A gap of exactly 12 steps (one hour) doesn't make a spike follow-on. Spikes an hour apart may come from a schedule, and the timing alone doesn't show they belong to the same burst.
- All gaps are measured from the previous spike's start to the current spike's start, not from when CPU falls back. This matches the gap analysis above and gives one consistent definition.

How per-type scores are reported:

- Unknown spikes stay in the overall scores but are left out of comparisons that need a known seasonal label. Their count is always reported, so readers can see how much data the per-type comparisons leave out.
- Because the labels are exclusive, the follow-on score covers follow-on spikes that aren't on yesterday's schedule, not every spike within 6 steps of an earlier one.
- The 543 seasonal spikes that are also within 6 steps of an earlier spike are reported next to the per-type results. A strong seasonal score for the hold may partly come from these spikes rather than from daily repetition alone. The overlap only describes the groups; it doesn't show how much of the hold's performance comes from either pattern.
- The scores describe performance on the groups as defined here. They don't prove the groups match distinct workload mechanisms.

Sensitivity tests, before drawing any conclusion from per-type scores. Each setting changes one thing from the baseline (±1 step, 6 steps, detection from step 762), giving six settings in all:

- seasonal tolerance of ±2 and ±3 steps;
- a follow-on limit of 8 and 12 steps;
- reference spikes for the seasonal check taken from a second detection run starting at step 250.

For each setting I'll report how many spikes change label. I won't pick thresholds by which setting gives the best method scores. The aim is to find out whether the conclusions survive reasonable changes to the definitions.

The step-250 run is a sensitivity test, not an assumed improvement. It uses the same `find_spikes` function and thresholds, with only the start step changed, so the spike definition stays the same. It is used only for looking back. Used for counting, it would change 38 of the published spikes after step 762.

Ramps are out of scope. A slow climb can come before a seasonal, follow-on or isolated spike, so it may fit better as a separate feature than as an exclusive type. Ramp detection waits until there's a defensible definition for how many rising steps and how big a rise count. Any later definition needs its own justification and sensitivity test before it's used to interpret method performance ([BACKLOG.md](../../BACKLOG.md)).

Checks before any per-type result is published:

- Test the rules on synthetic traces, including spikes that fit both seasonal and follow-on, missing history and the ends of a trace.
- Reconcile the 543 overlap and every count in this ADR with the labels the committed code produces.
- Run the sensitivity tests and report the label changes.
- Confirm unknown spikes still count in the overall evaluation.
- Report group sizes and the 6-step overlap next to every per-type score.

## Result

With the baseline settings, the labels split the test spikes into seasonal 6,749 (24.1%), follow-on 1,533 (5.5%), isolated 13,782 (49.3%) and unknown 5,916 (21.1%). Of the 2,536 spikes that came within 6 steps of the one before, 543 are seasonal and 460 are unknown, so 1,533 (60%) end up as follow-on.

What this gives:

- Missing history doesn't inflate the isolated group.
- Daily repetition is kept apart from short-term follow-on spikes.
- The rules are explicit and can be rerun.
- The overlap that could muddy results for the paper's hold is out in the open.
- The sensitivity tests show how much the conclusions depend on the thresholds.

What it costs:

- The thresholds are operational choices, not boundaries taken from the paper.
- Unknown labels shrink the data available for some per-type comparisons.
- A 6-step window misses longer bursts, and a wider one may group unrelated spikes.
- The step-250 run may recover the missing history, but it still has to be checked.
- Seasonal scores aren't pure daily-repetition scores, because some seasonal spikes also fall inside the hold window.

This ADR defines the labels and how to read them. It doesn't show that the labels improve provisioning, change the detector or validate any method.

## Troubleshooting

- Some seasonal labels are chance matches. Of the 21,680 spikes with enough history to check every offset, 30.9% have another spike within ±1 step of 288 steps back, but 6.0 to 9.8% have one at other offsets between 240 and 336 steps, including other whole hours such as 276 and 300. So the daily pattern is real, but between a fifth and a third of seasonal labels could be coincidence. Widening the tolerance raises the match rate to 37.3% at ±2 and 42.3% at ±3, much of it what chance alone would add.
- Some follow-on spikes are one long spike counted twice. In 332 of the 2,536 close pairs (13%), CPU never dropped below the detection threshold between the two counts. The refractory period ends after 4 steps, so a spike that stays high is counted again.
- The counts in this ADR come from a first check on the test traces, run outside the repo. Under [ADR-008](008-publish-reproducible-results.md) they have to come from committed code, so the labelling script prints them, and this ADR is corrected if any differ.
