# Next session: one task

**Split spikes by type and see where each method gains**
The look-ahead beats the paper's target overall, but nobody knows which spikes it gains on. Spikes come in at least three kinds: isolated ones that arrive out of nowhere, follow-on ones that come soon after an earlier spike, and seasonal ones that repeat at about the same time each day. If the gains sit mostly in follow-on and seasonal spikes, a simple heuristic might match them, and isolated spikes may be close to unpredictable from CPU history alone (opportunity #3 in [the review](docs/review.md)).

Work on a branch, `spike-types` (one branch and PR per task; merge with a merge commit, never squash).

1. Decide the definitions before looking at any results:
   - how close an earlier spike has to be for a spike to count as follow-on;
   - how close to the same time yesterday counts as seasonal;
   - what happens when a spike fits both, and whether a spike preceded by a slow ramp is its own type.

   Write the choices down in an ADR.
2. Label every real spike on the test traces with its type, using `find_spikes` in `scripts/metrics.py`. Count how many fall into each type.
3. Score spike coverage per type for cbase, proposed, fwdmax and fwdmax6 (three seeds each), plus the rolling-max and seasonal-max baselines, at matched over-provisioning and at 5, 15 and 30-minute delays. The saved test predictions are enough, so nothing needs retraining. If `data/` was wiped, rerun `scripts/get_data.sh` and the runners first.
4. Write the result up in `docs/replication.md`, update the README findings and the diagrams if anything changes, add a LOG line, and set the next task here.
5. Open the PR with a SOART description.

Done when: every test spike has a type, coverage is reported per type for every method across three seeds, and the write-up says which types each method gains on.

## End of every milestone
- Update README.md (status, findings) and docs/architecture.md (all diagrams; solid = done, dashed = planned).
- Update docs/project-story.md: new actions, results and every real error hit (problem, cause, fix, lesson).
- Commit, add one line to LOG.md, set the next single task here.
