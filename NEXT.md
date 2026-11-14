# Next session: one task

**Choose the next experiment and write its rule first**
The simulated autoscaler of [ADR-012](docs/adr/012-autoscaler-simulation.md) is done, so the next task isn't set yet. Pick one item from the [backlog](BACKLOG.md), then write its rule and a prediction in an ADR before running anything.

The open items closest to the current results:

- Strategy 2 from the paper (the altered loss) under the corrected metrics.
- Hold out time and clusters, not just traces.
- A closed-loop simulation with a latency signal. The simulated autoscaler is still a replay of recorded usage.
- Work out why the follow-on result changes with the window.

Work on one branch and one PR per task, and merge with a merge commit, never squash.

Done when: this file names one task, with the question it answers and what done looks like.

## End of every milestone
- Update README.md (status, findings) and docs/architecture.md (all diagrams; solid = done, dashed = planned).
- Update docs/project-story.md: new actions, results and every real error hit (problem, cause, fix, lesson).
- Commit, add one line to LOG.md, set the next single task here.
