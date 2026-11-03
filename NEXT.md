# Next session: one task

**Build a trace-driven autoscaler simulation**
Build a small autoscaler simulation using the existing Alibaba traces, with explicit resource-allocation and actuation delays. Compare no hold, computed hold and learned hold using allocated CPU, throttling and workload demand. This would help establish whether the improvements observed in offline coverage and throttling proxies translate into better provisioning decisions, addressing the biggest limitation of the current investigation.

Work on a branch, `autoscaler-sim` (one branch and PR per task; merge with a merge commit, never squash).

1. Decide the simulator's rules and the comparison before running anything, and record them in an ADR with a written prediction. Any tuning uses the validation traces, because the test set has already been used for these comparisons.
2. Test the simulator on synthetic traces.
3. Run no hold, computed hold and learned hold from the saved predictions, so nothing needs retraining.
4. Write the result up in `docs/replication.md`, update the README findings and the diagrams, add a LOG line, and set the next task here.
5. Open the PR with a SOART description.

A limit to keep in view: the traces record what containers used under the allocation they had. Replaying them can't show how a workload would respond to throttling, and it adds no latency signal, so this improves on the proxy without measuring customer impact.

Done when: the three methods have been compared in the simulator, and the write-up says what that does and doesn't show about provisioning.

## End of every milestone
- Update README.md (status, findings) and docs/architecture.md (all diagrams; solid = done, dashed = planned).
- Update docs/project-story.md: new actions, results and every real error hit (problem, cause, fix, lesson).
- Commit, add one line to LOG.md, set the next single task here.
