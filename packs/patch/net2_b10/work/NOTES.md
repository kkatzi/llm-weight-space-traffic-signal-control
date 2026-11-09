# Notes: spillback patch for the PPO signal controller (net2, budget 10 evals)

## What I built and why

**Diagnosis (recorded states only, no simulator).**
- The net almost always switches at the first allowed moment (12 s of green). When it decides, the current
  phase's stop-line queue is always 0. So it never really chooses a green length; it only chooses which
  phase comes next.
- The stop-line lanes are 22.4 m long, so each holds at most 3 vehicles and the queue/density inputs saturate at 1.
  A longer queue shows only in the upstream inputs `q_X0u` / `q_X1u` (halted vehicles on the 256 m lanes).
- The net almost ignores those upstream-queue inputs: about 0.05 logit per halted vehicle, against 1–2 logits
  per stop-line vehicle. They are non-zero in only ~4 % of training states, so PPO barely learned them.
- Consequence (seen in a traced episode): the S left pocket stayed full for 100 s while halted vehicles on S1u
  grew from 2 to 8. S1u is the upstream lane that feeds both straight lane 1 and the left pocket, so a
  left-turner waiting for the full pocket blocks the straight traffic behind it. Meanwhile the net kept serving
  NS-through and EW phases, which cannot clear that blockage. In spillback states (pocket full, ≥1 halted on
  X1u) the net served the matching left phase in only 593 of 1060 honoured decisions.

**Fix ("virtual vehicles").** Each halted upstream vehicle is shown to layer 1 as extra vehicles on the
stop-line lane it is waiting for. I add scaled copies of existing W1 columns into the upstream-queue columns:

    W1[:, q_X1u] += K * ( 2.0 * (W1[:, q_X2] + W1[:, d_X2])  +  1.0 * (W1[:, q_X1] + W1[:, d_X1]) )
    W1[:, q_X0u] += K * 3.0 * 0.5 * (W1[:, q_X0] + W1[:, d_X0] + W1[:, q_X'0] + W1[:, d_X'0])
    K = (256/7.5)/(22.4/7.5) = 11.43   (one upstream vehicle -> one stop-line vehicle)

- `X'` is the opposite approach, which is served by the same phase. For through lanes I average the column
  with its mirror, because the net's own response to N0 is oddly weak (it barely reacts to q_N0).
- Upstream lane 1 carries straight and left traffic, but it is blocked mostly by a full pocket
  (391 of 471 cases in the net's own episodes). So it counts mainly toward the left pocket, plus some toward
  lane 1.
- `q_up` is exactly 0 in ~96 % of states. There the edited net gives bit-identical logits to the original.
- No fitting. Every number is a construction constant: the capacity ratio K, and multipliers ρ = 1, 2, 3
  chosen as hypotheses and tested in the simulator. To fit the budget I keep, per column, the 48 (lane-1u)
  or 40 (lane-0u) largest-magnitude entries of the added vector. These carry about 95 % of its L1 mass and
  change 13 recorded decisions compared with the dense version.

## Which weights do what
- Changed: W1 columns `q_N1u, q_E1u, q_S1u, q_W1u` (48 entries each) are the left-spillback channel. When
  a left pocket is full and halted vehicles back up into upstream lane 1, the matching left phase gets priority.
- Changed: W1 columns `q_N0u, q_E0u, q_S0u, q_W0u` (40 entries each) are the through-spillback channel. Halted
  vehicles on upstream lane 0 push the through phase. They also act as a proxy gate: when lane 1 is full of
  straight vehicles (a through block on X1u), lane 0 usually spills too, so the through push stops the
  lane-1u term from wrongly calling the left phase.
- **Total: 352 of 7428 parameters changed** (all in W1; b1, W2, b2, W3, b3 untouched).

## Checks run
- Offline, on the recorded states:
  - Honoured decisions changed: 2.2 % (56 of 1503 in the net's own episodes), all in states with an upstream queue.
  - Left spillback served: 779/1060 (original 593).
  - Through spillback served: 556/727 (original 537).
  - Through-blocked states wrongly given the left phase: 12/326 (original 0).
- Synthetic sweeps: left-phase choice rises monotonically with 1 → 34 halted vehicles, and |logit| ≤ 12.4.
- `tools.trainability`: no dead units. Layer-1 saturation rose from 0.00 % to 1.1 % (spillback states only).

Simulator (dev, 9 episodes; original 34.5 / 44.9 / 55.0, mean 44.80):

| call | candidate | low | nom | high | mean |
|---|---|---|---|---|---|
| 1 | A: spillback ρ=1 (left 1, lane1 0.5, thr 1) | 34.5 | 44.7 | 54.3 | 44.49 |
| 2 | B: anticipation (moving upstream vehicles ×0.25) | 34.9 | 45.1 | 54.7 | 44.91 |
| 3 | C: spillback ρ=2 (2 / 1 / 2) | 34.5 | 44.6 | 53.7 | 44.25 |
| 4 | D: spillback ρ=3 (3 / 1.5 / 3) | 34.5 | 44.5 | 54.5 | 44.51 |
| 5 | **E: spillback 2 / 1 / thr 3 (delivered)** | 34.5 | 44.6 | 53.9 | 44.34 |
| 6 | F: E + phase-3 "stay" fix (16 W1[:,phase3] entries) | 34.5 | 44.8 | 54.2 | 44.49 |

- All spillback variants beat the original on nominal and high. Low is unchanged, as expected, because low
  demand has no spillback. B (anticipating moving upstream vehicles) hurt at low demand, so I dropped it.
- F removed the net's habit of re-picking EW-left at 12 s and did not help. Some of those stays are useful
  extensions for arriving left-turners, so I dropped it too.
- C vs E is a tie within noise; the difference sits only in the 3 high episodes, ±0.8 each. I deliver E
  because, unlike C, it does not reduce service to through-lane spillbacks below the original net.
- 6 of 10 simulator calls used.

## What I expect to go wrong
- The gain is small, about 0.5 veh-h on the mean, concentrated in high demand. The dev noise between
  near-identical variants is about ±0.3 on a scenario mean, so the test gain could be as small as ~0.2 or
  as large as ~1.
- There is no gating. If lane 1 is full of straight vehicles while the pocket is not, and lane 0 is not
  spilling, the lane-1u term can still call the (empty) left phase. That is rare under this net (12/326
  recorded), but it may matter if test demand has more EW-through congestion.
- Two simultaneous left spillbacks (NS and EW) are resolved by the net's own bias, which favours EW-left.
- Choosing the best of several variants on 9 dev episodes is optimistic, so my predictions sit between E's
  dev numbers and the mean over all spillback variants: low 34.5, nominal 44.7, high 54.2.

## Files
- `build.py`: the constructions (`build` = spillback, `build_h2` = anticipation, `build_stayfix`).
- `make_A.py` … `make_F.py`: the candidates sent to the simulator.
- `make_final.py`: writes `weights.npz` (= cand_E) and `predictions.json`.
- `analyze*.py`, `compare*.py`, `sanity.py`: the diagnostics above.
