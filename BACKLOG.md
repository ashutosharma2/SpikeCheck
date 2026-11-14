# Backlog (ideas parked so they don't hijack a session)

## Docs
- [x] README with a summary, status table, findings and how to reproduce
- [x] docs/architecture.md: six diagrams (data pipeline, forecast to score, paper vs corrected metric, target windows, experiment grid, TempInc model)
- [x] docs/project-story.md (SOART) and ADRs for the main decisions
- [x] Fix links in docs/replication.md for the new layout

## Experiments
- [ ] Strategy 2 from the paper (the altered loss) under the corrected metrics
- [ ] Calibrate the quantile readout (conformal prediction) so the over-provisioning setting maps to a target miss rate
- [x] Split spikes by type (isolated, follow-on, seasonal) and see where each method gains
- [ ] Flag spikes that follow a slow ramp: first define how many rising steps and how big a rise count, then sensitivity-test it (deferred in [ADR-009](docs/adr/009-spike-types.md))
- [ ] Hold out time and clusters (later days, other Google 2019 clusters), not just traces
- [ ] Sweep look-ahead length against delay instead of testing two points
- [ ] Replace the throttling proxy with a closed-loop autoscaler simulation and a latency signal. The simulated autoscaler of ADR-012 is still a replay of recorded usage
- [ ] Score spikes by size, with the cuts and a prediction fixed in advance
- [ ] Work out why the follow-on result changes with the window: at 12 steps the 5-minute sign flips
- [ ] Test the renewed-activity reading of the hold's lead on follow-on spikes where the previous peak is too low
- [x] Check whether choosing a policy per trace could help before building a selector: two pre-set checks on validation found no usable signal ([ADR-011](docs/adr/011-per-trace-policy-selection.md))
- [x] Test the computed hold in a simulated autoscaler with a scale-down window and a tolerance ([ADR-012](docs/adr/012-autoscaler-simulation.md))
- [ ] Per-trace selection left untested: features beyond the twelve tried, a library whose policies differ more, and deciding again each day

## Repo
- [ ] Reread docs/review.md before making the repo public
- [x] Pick a licence: MIT for the code, CC BY 4.0 for the docs and results
