# Backlog (ideas parked so they don't hijack a session)

## Docs
- [ ] README with a summary, status table, key findings and how to reproduce
- [ ] docs/architecture.md: six diagrams (data pipeline, forecast-to-score timeline, paper vs corrected metric, target windows, experiment grid, TempInc model)
- [ ] docs/project-story.md (SOART) and ADRs for the main decisions
- [ ] Fix links in docs/replication.md for the new layout

## Experiments
- [ ] Strategy 2 from the paper (the altered loss) under the corrected metrics
- [ ] Calibrate the quantile readout (conformal prediction) so the over-provisioning knob maps to a target miss rate
- [ ] Split spikes by type (isolated, follow-on, seasonal) and see where each method gains
- [ ] Hold out time and clusters (later days, other Google 2019 clusters), not just traces
- [ ] Sweep look-ahead length against delay instead of testing two points
- [ ] Replace the throttling proxy with a closed-loop autoscaler simulation and a latency signal

## Repo
- [ ] Reread docs/review.md before making the repo public
- [ ] Pick a licence before making the repo public
