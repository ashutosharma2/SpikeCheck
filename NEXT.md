# Next session: one task

**Write up the 30-minute look-ahead results**
The 30-minute target (fwdmax6) tests whether a look-ahead that matches the actuation delay restores spike coverage at a 30-minute delay. At slack 0.8, the 25-minute target covers 0.59 at that delay.

Work on a branch, `horizon30-results` (one branch and PR per task; merge with a merge commit, never squash).

1. Check the run finished: three fwdmax6 seeds trained, then `eval_seeds.py` over all twelve models.
2. Copy the outputs into `results/horizon30/`.
3. Compare fwdmax6 against fwdmax and the paper's target at matched slack, at 15- and 30-minute delays. Note what the longer look-ahead costs at shorter delays.
4. Add a section to `docs/replication.md`; one line to this LOG; set the next task here.
5. Open the PR (SOART description).

Done when: the 30-minute question has an answer backed by three seeds, and it's written up.

## End of every milestone
- Update README.md (status, key findings) and docs/architecture.md (all diagrams; solid = done, dashed = planned).
- Update docs/project-story.md: new actions, results and every real error hit (problem, cause, fix, lesson).
- Commit, add one line to LOG.md, set the next single task here.
