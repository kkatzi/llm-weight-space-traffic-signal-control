# NOTES: a "waiting cars" ranking term added to the PPO actor

## What I built and why

Analysis of the recorded states (`analyze*.py`) showed that the trained network:

- almost always switches at the first honoured decision (12 s of green; 98% of honoured decisions are switches),
  and by then its stop-line lanes are empty (about 0.05 cars per lane) and it leaves no upstream halting queue
  behind. So green extension is not where the loss is.
- chooses the phase with the most waiting vehicles at only 58-63% of its switches (the tuned actuated controller:
  81%); in 18-25% of switches the chosen phase has at least 2 fewer waiting cars than the best alternative.

So the edit adds a simple, interpretable term to the logits:

    logit_q += c * (u_q - mean_q u_q),   c = 1.0 per car

where `u_q` is the approximate number of vehicles waiting for phase q:
stop-line density of q's lanes x 2.99 cars, plus 0.5 x halting upstream cars and 0.25 x all upstream cars on the
upstream lanes feeding q (upstream lane 1 is shared: 65% to the through phase, 35% to the left phase). The
original network keeps deciding *when* to switch; the new term mostly re-ranks *which* phase to switch to. Values
were set by construction (lane capacities, turning shares, a slope and a gain), not fitted to anything.

## Which weights do what (371 changed parameters, the cap)

- `W1` rows 50, 42, 38 and their `b1` (3 x 47 = 141): the three Hadamard contrasts of u,
  A = u0+u1-u2-u3, B = u0-u1+u2-u3, C = u0-u1-u2+u3, as `tanh(0.1 * contrast)` (slope 0.1 per car). These three
  first-layer units were the ones whose removal flipped the fewest original decisions (0.2-0.4% each).
- `W2` rows 1, 27, 14 and their `b2` (3 x 65 = 195): pass-through units `tanh(1.0 * h1_i)`, one per contrast.
  Also the three least influential second-layer units (0.04-0.5% flips each).
- `W3` columns 1, 27, 14 (12): +-2.5 with the Hadamard signs, which in the linear regime gives
  c * (u_q - mean u) with c = 1.0. `b3` (4): absorbs the mean contribution of the three removed h2 units.
- `b2` of the 19 other h2 units most affected by the rewritten h1 units (19): compensates the mean shift of their
  input (reduces the collateral flips of the scaffolding alone from 1.6% to 1.2%).

`build.py 1.0 work/weights.npz --upq=0.5 --upd=0.25` rebuilds the file exactly.

## Checks

Offline (recorded states, honoured decisions): 9.5% of decisions change; switch rate 98.1% -> 99.1%; share of
switches going to the phase with most waiting cars 63% -> 71%; cars forgone per switch 0.87 -> 0.52; no dead
or saturated units introduced.

Simulator (8 of 10 calls used; same 9 dev episodes; original network = 44.8 veh-h, low/nominal/high
34.5/44.9/55.0):

| variant (c, upstream weights halting/all) | low | nominal | high | all | leftovers |
|---|---|---|---|---|---|
| 0.7, 1.0/0.5 | 34.4 | 44.8 | 54.6 | 44.6 | 1 |
| 1.5, 1.0/0.5 | 34.3 | 44.9 | 54.8 | 44.6 | 0 |
| 1.0, 1.0/0.5 | 34.3 | 44.8 | 54.7 | 44.6 | 0 |
| 2.5, 1.0/0.5 | 34.9 | 44.9 | 66.4 | 48.7 | 2 |
| **1.0, 0.5/0.25 (delivered)** | **34.3** | **44.4** | **54.1** | **44.3** | **0** |
| 1.0, 0/0 | 34.5 | 44.8 | 53.8 | 44.4 | 1 |
| 1.5, 0.5/0.25 | 34.2 | 44.8 | 54.8 | 44.6 | 0 |
| 0.7, 0.5/0.25 | 34.4 | 44.6 | 54.8 | 44.6 | 2 |

Lessons: a strong term (c = 2.5) makes the controller hold a phase whose upstream lanes are busy and collapses
at high demand; weighting the upstream lanes less removes that failure mode and gave the best result. The gains
between the moderate variants (44.3-44.6) are of the order of the episode-to-episode noise, so the delivered
network is better than the original on these 9 episodes by about 0.5 veh-h, but the true gain is probably
0.2-0.5 veh-h.

## What I expect to go wrong

- The improvement is small (about 1%) and the variants differ by amounts comparable to noise, so on the 30 test
  seeds the edited network may come out only marginally ahead, or even behind on one scenario.
- One vehicle was left at the drain cap in one dev episode for two of the sibling variants (not the delivered
  one): the scaffolding changes about 1% of decisions, and a rare all-but-one-lane-empty state could deadlock
  where the original network served the lone vehicle. The demand term should favour the lone car's phase, but
  this is not proven for every state.
- Under heavier-than-tested demand the stop-line features saturate at 3 cars per lane and the term becomes less
  informative; it then mostly defers to the original network (by design), but the high-demand collapse of c = 2.5
  shows the mechanism has a failure mode when it is too strong.
- Predictions (`predictions.json`): low 34.4, nominal 44.6, high 54.6 veh-h, i.e. the dev gains shrunk toward
  the original network's reference to account for seed noise.
