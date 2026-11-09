# Repair of the network for the failed W2 detector

## What went wrong with the start network

With q_W2 = d_W2 = 0, the network never sees the west left-turn pocket fill. In the recorded start-network
episodes (states.npz, controller 0) 4 of 6 episodes reach the drain cap. The network locks into phase 2
(EW_through) for good (e.g. episode 3 from t = 495 s to 5400 s). In that state both left pockets are full and their
left-turners block the upstream lanes E1u and W1u (q_E1u = q_W1u = 1, q_E1 = q_W1 = 0). The network reads the
upstream queues as through demand, so it keeps EW_through, which cannot discharge them. Forward passes on that
state show the fix: with q_W2 = d_W2 = 1 (what a working detector would report) the network switches to phase 3
(logit 1.57 vs 1.29); with 0 it stays in phase 2 (0.33 vs 1.26).

## What I built (all edits are in W1; 320 of 7428 parameters changed)

The network keeps reading its two W2 channels through the original weights, but the lost input is replaced by a
stand-in built from working inputs:

    W2_hat = E2  +  0.75 * (t_green - 0.3*ok - 0.7*phase3)        (used for both q_W2 and d_W2)

1. **Twin substitution (128 params):** `W1[:, q_E2] += W1[:, q_W2]`, `W1[:, d_E2] += W1[:, d_W2]`.
   E2 (east left pocket) is W2's symmetric twin: the same geometry, the same demand (E and W share the level, the
   profile and the directional bias e^b; 30 % left), and the same serving phase (3, EW_left), so both fill and drain
   together. Every first-layer unit now gets, from q_E2 / d_E2, the contribution it used to get from q_W2 / d_W2.
   No fitted coefficient: the scale 1 comes from symmetry.
2. **Time term (192 params):** `W1[:, t_green] += 0.75*u`, `W1[:, ok] -= 0.225*u`, `W1[:, phase3] -= 0.525*u`
   with `u = W1[:, q_W2] + W1[:, d_W2]` (the original W2 weights).
   Why: the twin alone never shows W left-turners once E2 is empty. On an empty junction the network holds phase 1
   or 2 forever (forward passes on synthetic states), so hidden W2 cars stay until the drain cap. Eval 1 (twin
   only) left 2 and 5 vehicles at the cap. While any phase other than 3 is green, W2 can only fill, so a long green
   is real evidence of W2 demand. Design of the constants:
   - `-0.3*ok`: the term is about 0 at the first honoured decision (t_green = 0.2, ok = 1). Decisions with ok = 0 are
     never honoured, so the term's value there doesn't matter.
   - `-0.7*phase3`: A + B = 1 keeps the term <= 0 throughout phase 3, so it never stretches EW_left (W2 is draining
     then). It also stops the original network holding phase 3 forever with one car waiting on E0, W1 or S1.
   - K = 0.75: about the smallest value at which an empty junction in phase 2 hands over to phase 3 (at 60 s).
     K = 0.6 hardly changes behaviour. K = 1.0 brings odd decisions (lone N2/N0 cars trigger needless phase-3
     visits) and longer waits for hidden W2 cars, so I didn't evaluate it.

Unchanged: W1[:, q_W2] and W1[:, d_W2] (they multiply 0 anyway), every other W1 column, b1, W2, b2, W3, b3.

## Checks run

- `tools.edit_size`: 320 changed (limit 371), all in W1 columns q_E2, d_E2, phase3, ok, t_green. Saved file
  reloaded and identical to the evaluated v2 file.
- Forward passes on recorded states: on the start network's gridlock states, v1/v2 pick phase 3 (breaks the lock).
  On the fixed/actuated states with t_green <= 0.55, v2 changes 1.7-11 % of honoured decisions relative to the
  twin-only network. Most changes are leaving phase 3 when E2 is already empty.
- Synthetic probes (`work/probe.py`): with E2 still occupied, phase 3 is held at every t_green; with N/S through
  lanes full it switches to phase 0, the same as the original network.
- Single-car service check (`work/strand_check.py`, forward passes only): one waiting car on each lane, from each
  starting phase, stepping the decision rules. v2 serves every lane, including a hidden W2 car (35-50 s). The one
  exception is N1 (below).
- Simulator (dev, 9 episodes; 2 of 10 calls used):

| version | low | nominal | high | all | vehicles left at cap |
|---|---|---|---|---|---|
| start network | 37.7 | 48.7 | 556.5 | 214.3 | (gridlock) |
| v1 twin only (128 params) | 34.8 | 45.6 | 57.5 | 46.0 | 0,0,0,0,2,0,0,5,0 |
| **v2 twin + time term (final, 320 params)** | **34.9** | **45.1** | **56.1** | **45.4** | all 0 |
| full-detector network (reference) | 34.5 | 44.6 | 54.4 | 44.5 | |
| fixed-time (tuned) | 37.7 | 48.4 | 58.7 | 48.3 | |

## What I expect to go wrong

- **Lone N1 car (inherited):** the original network answers a queue on N1 (N straight) alone with phase 2, even
  with working detectors. If the last vehicle in the drain is an N1 car and phase 0 isn't green, it can wait until
  the drain cap (+~0.5 veh-h per car). It's rare: 2 of 9,221 realistic recorded states have an N1-only queue, and
  it didn't happen on dev. Fixing it would mean re-mapping the q_N1/d_N1 columns (128 params, over budget).
- **W2 fuller than E2:** random arrivals make the twin wrong at times. W left-turners then wait an extra cycle or
  spill into W1u and briefly block W straight traffic. This is the likely cause of most of the remaining gap to the
  full-detector network, which is largest in the high scenario (+1.7 veh-h).
- In quiet periods the junction now cycles 1 <-> 3 instead of holding one phase. This costs a few yellows and should
  have little effect.
- Test seeds differ: dev per-episode spread is about +-1 (low) to +-2 (high) veh-h, so the test means can move by
  about 1 veh-h.

## Files

`build_v1.py` (twin only), `build_v2.py` (final construction), `make_final.py` (writes `weights.npz` and checks),
`probe.py`, `strand_check.py` (forward-pass probes), `v1_twin.npz`, `v2_twin_time.npz`, `weights.npz` (= v2),
`predictions.json`.
