# NOTES: net0_haiku edit

## What was built
Final network (`work/weights.npz`) = start network with only the output layer W3 changed (256 of 7428 params,
3.4 %). W1, b1, W2, b2, b3 are untouched.

Each row k of W3 (logit of phase k) gets `alpha * (m_k - m)` added, where
- `m` = mean of the 64 second-layer activations h2 over recorded states with `ok = 1`;
- `m_k` = mean of h2 over those states where phase k has the strictly largest stop-line queue on its served
  lanes (phase 0: N0,N1,S0,S1; phase 1: N2,S2; phase 2: E0,E1,W0,W1; phase 3: E2,W2), and that queue is non-zero;
- alpha = 1.0.

Purpose: nudge the readout toward the hidden directions that already track "this phase has the most queue". This
is a construction from class-mean statistics of recorded activations (no regression, no gradient, no search over
weights). It is the readout-level analogue of a max-queue rule, and it leaves the rest of the network as it was.

## Simulator results (dev, mean TTS veh-h)
| call | candidate | low | nominal | high | mean |
|---|---|---|---|---|---|
| 1 | queue readout, alpha 1.0 (FINAL) | 34.4 | 44.6 | 54.6 | 44.6 |
| 2 | stickiness readout, beta 0.2 | 354 | 662 | 1004 | 673 (gridlock) |
| 3 | queue readout, alpha 4.0 | 34.4 | 44.6 | 54.7 | 44.6 |

Start network (from TASK.md, same 9 episodes): 44.7 (34.7 / 44.9 / 54.4).

So the queue readout is neutral: -0.1 veh-h overall, well inside what I would call noise for this kind of change.
I did not measure a genuine improvement.

## Checks run
- `tools.edit_size` = 256 changed / 7428 (limit 371).
- `tools.save` shape and finiteness check passed.
- Decision changes on recorded states (ok = 1): about 13 % for alpha 0.5-6, so the edit is moderate, not zero.
- Trainability of start network: median max prob 0.98, 43 % of states above 0.99; the policy is confident.

## What I learned
- The start network switches almost every time the 10 s minimum green allows (about 92 % of ok decisions
  choose a different phase). Biasing the readout toward the current phase ("stay") is catastrophic: call 2 gridlocked
  (vehicles left at the drain cap, up to 1284). Stay-bias readouts should be avoided; even beta 0.05 changes only
  3 % of decisions, and beta 0.2 changes 31 %.
- Queue-readout changes are too weak to move TTS at this scale. Going from alpha 1 to alpha 4 changed nothing
  measurable.

## What I expect to go wrong
- The predicted improvement is essentially zero. If the test set matches dev, the final network is roughly tied
  with the start network (within noise).
- The edit is a readout-only change with no PPO-trained justification, so it may not generalise beyond dev demand.
- Stickiness-type edits are dangerous: they can gridlock the junction.
- Only 3 of 10 simulator calls were used; the rest were not spent because further readout variants looked neutral.

## Files in work/
- `build.py`: construction of both candidate edits (`build`, `build_stay`) with statistics only.
- `cand_a1.npz` (final), `cand_a4.npz`, `cand_stay02.npz`: candidates that were simulated.
- `weights.npz`: final weights (alpha 1.0). `predictions.json`: dev-based predictions.
