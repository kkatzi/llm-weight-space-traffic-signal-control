# Repair of the network for the failed W2 detector

## What I built and why

The network was trained with the W2 pocket visible, and it relies on it heavily: with `q_W2 = d_W2` swept from 0
to 1 on the recorded states, its share of phase-3 (EW_left) choices goes from 17% to 65%. With the detector
stuck at 0, it thinks the west left pocket is always empty. It under-serves phase 3, so the 3-car pocket spills
back onto upstream lane W1u (which also carries W straight traffic) and blocks it. Signs of this in the data:
under the start network, W1u has far more halted cars than its mirror E1u (90th percentile 6 cars vs 2).

I did not retrain anything. Instead I give the network a **stand-in for q_W2 / d_W2**, built from inputs that
still work, and feed it through the network's own learned W2 pathways:

1. **Copy the opposite pocket (exact, at layer 1).** E2 and W2 have the same demand and are served by the same
   phase (3), so `q_W2 := q_E2` and `d_W2 := d_E2`. Implemented as `W1[:, q_E2] += W1[:, q_W2]` and
   `W1[:, d_E2] += W1[:, d_W2]`. (The W2 columns themselves are left alone, since their inputs are always 0.)
2. **Spillback override: "W2 is full".** I checked this on the mirror lanes, where the detectors work. If at
   least one car is halted on the upstream lane 1 while the pocket's own left phase is red, the pocket is almost
   always full: mean q = 0.85–0.92 for E, 0.92–0.99 for S, against about 0.3 otherwise (`analyze5.py`). This is
   a threshold rule, so it needs a tanh unit. Layer-1 unit **57** was the least important unit: freezing it at
   its mean changes only 0.18% of honoured decisions. I repurposed it as an indicator:
   `h57 = tanh(170*q_W1u - 180*phase3 - 2.5)`. That is -0.99 with 0 halted cars, +0.99 from 1 halted car upward,
   and always off while phase 3 is green, when the pocket is draining. Its outgoing weights `W2[:,57] = v/2`, plus
   `+v/2` in `b2`, move the layer-2 pre-activations by `v` when it is on and by 0 when it is off. `v` is the
   **average** (over the 2,240 recorded states where the rule fires) of [layer-2 pre-activation of the start
   network with `q_W2 = d_W2 = 1`] minus [that of the copy network]. That is a mean of forward-pass activations,
   not a fit. The old mean contribution of unit 57 (`W2_old[:,57] * 0.497`) is folded into `b2`.

So the network now acts as if W2 = "full if W1u shows spillback and phase 3 is not green, else equal to E2".
On the working N/S pockets, this "ind|copy" imputation gave the highest agreement with the true-input
decisions of the candidates I tried: zero, mean, copy, indicator alone, and indicator + mean (`analyze3.py`,
`analyze6.py`).

## Which units and weights do what

| changed | count | role |
|---|---|---|
| `W1[:, 5]` (q_E2 column) | 64 | += `W1[:, 11]`: q_E2 also drives the q_W2 pathway |
| `W1[:, 17]` (d_E2 column) | 64 | += `W1[:, 23]`: d_E2 also drives the d_W2 pathway |
| `W1[57, :]`, `b1[57]` | 44 + 1 (2 entries of the row are already in the columns above) | unit 57 becomes the W2-spillback indicator: weight 170 on q_W1u, -180 on phase3, 0 elsewhere, bias -2.5 |
| `W2[:, 57]` | 64 | carries v/2 (the "pocket full" correction) into layer 2 |
| `b2` | 64 | +v/2 (on/off offset) + old mean contribution of unit 57 |
| **total** | **301 / 7428** | `tools.edit_size` = 301 (limit 371). W3 and b3 are untouched. |

## Checks I ran (forward passes on recorded or synthetic states only; no simulator, no fitting)

- `tools.edit_size` = 301. The saved file reloads (float32) with identical behaviour.
- The indicator unit matches the rule exactly on all 14,340 recorded states: min h = 0.986 where it should be on,
  max h = -0.987 where it should be off.
- Agreement of honoured decisions with the "ind|copy" target (the start network fed the imputed W2):
  start network 0.755 → copy only 0.923 → final **0.986** (0.941 on indicator states, 0.996 elsewhere).
  On non-indicator states the final network matches the copy network in 99.6% of decisions.
- On indicator states in phase 2 (W through green, W1u blocked), the start network keeps phase 2 in 369/776
  cases. The full-information target picks phase 3 in 743/776, and the final network in 721/776.
- Stress sweep (`stress.py`): 0–34 halted cars on W1u in phase 0, 2 and 3 states. The response is monotone.
  Phase 3 states are unaffected (gated off). Max |logit| stays at 9.66, the same as the start network.
- `tools.trainability`: no dead units. Layer-1 saturation rises from about 0 to 0.5% (unit 57 is a hard
  switch by design). Layer-2 saturation rises from 0.068 to 0.122. Median max prob is 0.984 vs 0.978.
- Analogue tests: I pretended a *working* pocket (N2, S2, E2) had failed and scored each imputation against the
  network's true-input decisions. Only the best formula was used (no tuning beyond picking from 6 fixed formulas).

## What I expect to go wrong

- Information that no other lane carries is still missing. When W2 fills without spillback and E2 is empty,
  the network does not see it, and when E2 is full but W2 is empty it serves phase 3 too early. On the N/S
  analogue, about 8–12% of decisions in non-indicator states still differ from full information.
- The override is a linear correction averaged over states (94% agreement on indicator states, not 100%). It
  also pushes a little too hard when E2 is already full.
- The indicator fires on any halted car on W1u. In heavy W through traffic, W1u can be queued with straight
  vehicles, which would cut phase 2 short after 10 s and switch to phase 3. The mirror data says the pocket
  is then nearly always full anyway (0.93 in phase 2), and the full-information network makes the same choice,
  but a phase 2 ↔ phase 3 back-and-forth with 3 s yellows could cost some capacity at high demand.
- Expected effect: most of the gain at **high** demand, where spillback hurts most, and little at low demand.
  Predictions: low 35.1, nominal 45.8, high 57.2 veh-h. That is between the failed network (35.5 / 46.8 / 61.8)
  and the full-information network (34.7 / 44.9 / 54.4). The uncertainty is about ±2 veh-h at high demand.

## Files

`build.py` (constructs `weights.npz` and `weights_copy_only.npz`), `check.py` (+ `check_output.txt`),
`stress.py`, `analyze1.py`–`analyze6.py` (statistics and analogue tests), `weights.npz`, `predictions.json`.
