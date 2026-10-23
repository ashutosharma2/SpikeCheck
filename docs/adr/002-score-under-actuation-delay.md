# ADR-002: Score forecasts under actuation delay

- Status: Accepted
- Date: 2026-10-22

## Scenario

A forecast is only useful if there's time to act on it. Adding CPU to a container, reclaiming it from a low-priority neighbour or starting a new instance all take time, so the allocation in force at any moment was decided earlier.

## Obstacle

The paper's recall counts a spike as predicted if a matching forecast appears in the 4 steps up to and including the spike. That includes the forecast for the spike's own step, made one step (5 minutes) before it. In effect the paper scores as if resources can be changed instantly, so it can't tell a forecaster that anticipates from one that reacts at the last moment.

## Action

Every forecaster is also scored with an actuation delay d: the allocation in force at step t is the forecast that was issued d steps earlier. Three delays are reported:

- d = 1 (5 minutes): the forecast for the next step is used straight away.
- d = 3 (15 minutes)
- d = 6 (30 minutes)

At each delay, `metrics.py` reports spike coverage (the allocation in force at a spike is at least 80% of it) and throttled core-hours.

Alternatives considered:

- **One delay only.** Simpler, but the right delay depends on the platform, and results change with it.
- **A full autoscaler simulation.** More realistic, but it adds many choices (scaling rules, cooldowns) that would blur what is being compared. It's in BACKLOG.md.

## Result

The delay flips the ranking:

- With no extra delay, a plain classifier read at the 0.95 quantile beats the paper's method at the same over-provisioning (recall 0.70 against 0.62).
- At a 15-minute delay, the paper's method wins (coverage 0.57 against 0.44), because its target looks ahead.
- A perfect forecaster of the paper's target covers 99.6% of spikes at a 5-minute delay but only 28% at 30 minutes, because its look-ahead (25 minutes) is shorter than the delay.

This is what led to the horizon experiments: a target that only looks ahead, and one whose look-ahead matches a 30-minute delay.

## Troubleshooting

- **Index off by one.** The allocation in force at step t is the forecast *for* step t − d + 1, which was issued at step t − d. Writing this down once, in a diagram (docs/architecture.md, section 2), avoided mixing up "issued at" and "made for" in later scripts.
