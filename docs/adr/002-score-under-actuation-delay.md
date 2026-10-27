# ADR-002: Score forecasts under actuation delay

- Status: Accepted
- Date: 2026-10-22

## Scenario

A forecast only helps if there's time to act on it. Adding CPU to a container, taking it back from a low-priority neighbour or starting a new instance all take time, so the allocation in force at any moment was decided earlier.

## Obstacle

The paper counts a spike as predicted if a matching forecast appears in the 4 steps up to and including the spike. That includes the forecast for the spike's own step, made one step (5 minutes) before it. In effect the paper scores as if resources can be changed instantly, so it can't tell a forecaster that sees a spike coming from one that reacts at the last moment.

## Action

Every forecaster is also scored with an actuation delay d: the allocation in force at step t is the forecast issued d steps earlier. Three delays are reported. With d = 1 (5 minutes) the forecast for the next step is used straight away; d = 3 is 15 minutes, and d = 6 is 30 minutes.

At each delay, `metrics.py` reports spike coverage (whether the allocation in force during a spike is at least 80% of it) and throttled core-hours.

Alternatives considered:

- A single delay. Simpler, but the right delay depends on the platform, and the results change with it.
- A full autoscaler simulation. More realistic, but it adds many choices (scaling rules, cooldowns) that would blur what is being compared. It's in BACKLOG.md.

## Result

The delay flips the ranking:

- With no extra delay, a plain classifier read at the 0.95 quantile beats the paper's method at the same over-provisioning (recall 0.70 against 0.62).
- At a 15-minute delay the paper's method wins (coverage 0.57 against 0.43), because its target looks ahead.
- A perfect forecaster of the paper's target covers 99.6% of spikes at a 5-minute delay but only 28% at 30 minutes, because its 25-minute look-ahead is shorter than the delay.

This led to the horizon experiments: one target that only looks ahead, and one whose look-ahead matches a 30-minute delay. The second confirmed the rule. A look-ahead at least as long as the delay brings coverage back (0.74 against 0.59 at a 30-minute delay and slack 0.8), and costs nothing at shorter delays.

## Troubleshooting

- It's easy to get the index off by one. The allocation in force at step t is the forecast made *for* step t − d + 1, which was issued at step t − d. Writing this down once, in a diagram (docs/architecture.md, section 2), kept "issued at" and "made for" from getting mixed up in later scripts.
