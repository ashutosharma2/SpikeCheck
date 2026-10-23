# ADR-004: Change one thing at a time: the authors' model and inputs stay, only the target varies

- Status: Accepted
- Date: 2026-10-22

## Scenario

The question is whether the paper's ideas work and why, not whether a better forecaster can be built.

## Obstacle

The paper's "Proposed" method changes two things at once: the model's inputs (a short running maximum, then a log) and its target (the maximum over a window around each step). A gain could come from either. Any change of model or training settings on our side would add a third possible explanation.

## Action

Every run uses the authors' TempInc classifier, imported from their code at a pinned commit, with their training settings (Adam, learning rate 0.01) and their 512-step lag. Every run also uses the same 8 inputs and the same split.

Only the target changes: the next value (cbase), the paper's target (proposed), the look-ahead alone (fwdmax) and a 30-minute look-ahead (fwdmax6). The paper's input change gets its own test: proposed_full adds the paper's input transform on top of the paper's target.

Alternatives considered:

- Use the paper's full method as published. Faithful, but the two changes can't be pulled apart.
- Use a stronger, modern forecaster. It might score better, but it would answer a different question.

## Result

- The input transform adds almost nothing: proposed_full and proposed score 0.61 and 0.62 recall, and 0.56 and 0.57 coverage at a 15-minute delay.
- Every difference between targets can be put down to the target.

## Troubleshooting

- Training was noisy. The paper's learning rate of 0.01 makes validation loss jump between epochs, and the seed-0 run of the paper's target stopped early, after 7 epochs. Instead of tuning it, which would be another change, each target was trained with three seeds and the comparison uses all of them (ADR-003).
