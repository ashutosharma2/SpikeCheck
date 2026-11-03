# Next session: one task

**Test whether the hold needs to be learned**
The spike types show the hold leading on follow-on spikes at 5 and 15 minutes while costing coverage on the other types. The paper's target splits exactly into a part already in the model's input and the look-ahead target, so the hold could be computed instead of learned, with no retraining.

Work on a branch, `computed-hold` (one branch and PR per task; merge with a merge commit, never squash).

1. Fix the rule before opening the test set: what counts as better overall, how much follow-on loss is acceptable, and what each mixed outcome means. Record it in an ADR and push it before the run.
2. Dry-run the scorer on the validation traces.
3. Run the test once, and report it whichever way it goes.
4. Write the result up in `docs/replication.md`, update the README findings and the diagrams, add a LOG line, and set the next task here.
5. Open the PR with a SOART description.

Done when: the test has run once at the frozen commit, and the write-up says which claims held at which delay.

## End of every milestone
- Update README.md (status, findings) and docs/architecture.md (all diagrams; solid = done, dashed = planned).
- Update docs/project-story.md: new actions, results and every real error hit (problem, cause, fix, lesson).
- Commit, add one line to LOG.md, set the next single task here.
