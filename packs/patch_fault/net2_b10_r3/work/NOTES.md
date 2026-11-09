# Repair of the network for the failed W2 detector

Final file: `work/weights.npz` (built as candidate `C3` by `work/build.py`, same file as `cand_C3.npz`).
332 of 7428 parameters changed (limit 371). Dev TTS: **44.7 veh-h** (low 34.6, nominal 45.2, high 54.4) and no
vehicles left at the drain cap. For comparison, the start network with the failed detector scored 47.9 and the
network with every detector working scored 44.8.

## What I built and why

With q_W2 = d_W2 = 0, the network thinks the west left-turn pocket is always empty. On the recorded states,
setting W2 to a plausible value changes about 17% of the honoured decisions, mostly towards phase 3
(EW_left), so the start network under-serves that pocket. In the recorded start-network states, upstream lane
W1u has a queue about twice as often as E1u (left-turners spill back and block the through lane).

I put back an estimate of the missing reading in three parts:

1. **Shrunk symmetric proxy (folded into existing inputs).** E2 is the east pocket: same demand, same geometry,
   and phase 3 empties both pockets. The two pockets are only partly correlated: on the N2/S2 pair, where both
   detectors work, the correlation is 0.3 to 0.5, the slope is about 0.3 to 0.5, and the intercept is
   0.15 to 0.25 when not served and 0.02 to 0.1 when served. So I use a conditional-mean-style proxy:
   `W2_hat = 0.45*d_E2 + c`, with c = (0.22 q, 0.25 d) when phase 3 is not green and (0.03 q, 0.10 d) when it is.
   These numbers come from those statistics and from the mean E2 occupancy per phase, not from any fit.
   They are folded as:
   - `W1[:,17] += 0.45*(W1[:,11] + W1[:,23])`: the d_E2 column also carries the slope for both W2 inputs.
     I merged q and d to save 64 parameters.
   - `b1 += 0.22*W1[:,11] + 0.25*W1[:,23]`: the intercept outside phase 3.
   - `W1[:,43] += (0.03-0.22)*W1[:,11] + (0.10-0.25)*W1[:,23]`: the lower intercept while phase 3 is green.
2. **Spillback unit (layer-1 unit 42).** On the E-side analogue, a halted car on the upstream lane outside the
   pocket's own phase means the pocket is full: P(q_E2 = 1 | q_E1u > 0) = 0.96 to 0.99 under the start network.
   Unit 42 was the least important unit: replacing it by its mean flips under 0.5% of decisions. I rewired it to
   `s = tanh(2400*q_W1u + 80*t_green - 2600*phase3 - 66 + small rest)`. That is +1 when at least one car halts on
   W1u, or when any green other than phase 3 has lasted about 50 s or more; it is -1 otherwise and always -1
   during phase 3. Its old contribution went into b2 at its mean. Its new outgoing column `W2[:,42]` adds to
   layer 2 the mean change in layer-2 pre-activation between "W2 detector reads full (q = d = 1)" with the
   original layer 1 and "proxy" with the patched layer 1, over the recorded spill states. That is an average of
   forward passes, not a fit. On the recorded spill states the patched network now agrees with the original
   network fed W2 = full in 93.5% of decisions (start network: much lower; it picked phase 3 in 30% of them
   instead of 74%).
   The long-green part of the trigger is an anti-starvation guard. With W2 invisible, the network could rest in
   phase 2 for ever while a lone W2 car (or a lone N0/S1 car, a quirk the original network also has) waits.
3. **Phase-3 max-green (layer-1 unit 50 and layer-2 unit 1, both the least important of their layers).**
   Unit 50 becomes `g = tanh(100*phase3 + 80*t_green - 166 + small rest)`. That is on only when phase 3 has been
   green about 50 s or more, which in the recorded data never happens during the demand hour. Layer-2 unit 1
   becomes `u = tanh(20*g + centred rest)`; its old contribution was folded into b3. `W3[:,1] = (-3, 0, +3, -3)`
   with a matching b3 offset, so it votes for phase 2 only when on. This removes a lock-up in which the network
   holds phase 3 indefinitely while a lone W0 or E1 car waits. That lock-up left one vehicle in the junction
   until the drain cap in dev episode high_1101 in calls 1 and 2.

## Parameter count (332)
W1: 131 (column 17: 64, column 43: 64, W1[42,31], W1[42,45], W1[50,45]); b1: 64; W2: 65 (column 42 and W2[1,50]);
b2: 64; W3: 4 (column 1); b3: 4. Checked with `tools.edit_size` on the saved float32 file.

## Checks
- `work/analyse.py`: decision distributions on the recorded states, agreement with the original network fed
  W2 = full on spill states, and the trainability numbers. In the patched network no unit is dead and the
  saturation share is small.
- Unit behaviour on the recorded states: the spill unit is +1.000 on every q_W1u > 0 state outside phase 3,
  -1.000 in phase 3, and -1.000 to -0.99 otherwise except during long drain-time greens. The max-green units
  are -1 everywhere except phase-3 greens of 50 s or more.
- Compared with C (call 1), C3 changes only 10 honoured in-hour decisions out of about 6000; they are near-ties
  caused by keeping the rest of the rewired rows.
- `work/table.py` and `work/probe.py`: forward passes on synthetic lone-car states. `probe.py` follows the
  network's own choices through the green, minimum-green and yellow rules in TASK.md for one fixed halted car;
  it does not simulate traffic. The start network starves lone cars in 42 (lane, start) cases, C in 51 and
  C3 in none.
- The float32 file gives the same decisions as the float64 build on all 14337 recorded states.

Simulator calls (5 of 10 used, 9 dev episodes each, mean TTS):

| call | candidate | mean | low / nominal / high | left at cap |
|---|---|---|---|---|
| 1 | C: proxy + spill unit (zeroed row) | 44.79 | 34.8 / 45.0 / 54.6 | 1 (high_1101) |
| 2 | C2: C + long-green trigger in phases 0-2 | 44.78 | 34.8 / 45.0 / 54.6 | 1 (high_1101) |
| 3 | **C3: C2 + phase-3 max-green (final)** | **44.74** | 34.6 / 45.2 / 54.4 | 0 |
| 4 | V70: C3 with rho = 0.7 | 45.14 | 35.0 / 45.2 / 55.3 | 0 |
| 5 | V25: C3 with rho = 0.25 | 44.97 | 34.9 / 45.2 / 54.8 | 0 |

Single episodes move by up to ±1.5 veh-h between near-identical variants, so differences below about 0.3 in the
mean are noise. The proxy slope is not sharply tuned; the middle value, which was also my prior, scored best.

## What I expect to go wrong
- The main gain comes from the proxy and the spill unit, and it lands at the level of the network with every
  detector working. It cannot do better than that network's own quirks allow.
- The spill unit fires on any halted car on W1u outside phase 3. If heavy through traffic alone backs up into
  W1u, as in high demand when phase 2 is short, it calls phase 3 when W2 may not be full. The E-side analogue
  under fixed-time control shows P(pocket full | spill) of only 0.3 to 0.7.
- Lone-car behaviour at the end of a run has improved, but the probe is only a rough check. A rare drain-time
  lock-up of some other shape may remain, costing about 0.5 veh-h in an affected episode.
- The guards act on greens of 50 s or more. If a held-out episode needs a green that long during the demand
  hour (none appear in the recorded data), the guard cuts it short.
- Predictions (`predictions.json`): low 34.7, nominal 45.2, high 54.7. These are the dev scores shrunk slightly
  towards the average of the five variants. Test seeds differ, so expect about ±0.5 per scenario.

## Files
`build.py` (construction; `python build.py C3` rebuilds the final weights), `analyse.py`, `table.py`,
`probe.py`, `cand_*.npz` (evaluated candidates), `weights.npz`, `predictions.json`.
