# Patch of the PPO signal-control network: notes

Final file: `work/weights.npz` (built by `work/build2.py`, identical to `work/r4.npz`).
**271 of 7428 parameters changed** (W1 126, b1 5, W2 68, b2 64, W3 4, b3 4). No optimiser, regression or search
over weights was used. Every value is set by construction from the rule definitions below. Recorded-data
statistics were used only to pick the hidden units to reuse (the least important ones) and to centre biases (means
and ranges of activations).

## What I found in the original network

* It is already a strong controller: it switches at the minimum green (12 s) about 97% of the time, serves the left
  pockets just as they fill (mean red of EW-left 43 s, pocket fill time about 42 s), and its phase shares are close
  to a square-root allocation. I found no evidence-based direction for changing its normal phase choice. Its
  near-ties (for example phase 0 vs phase 3 after phase 2) are split 50/50 and don't track the queues, so I left them alone.
* **Bug 1, idle EW-left hold.** With phase 3 green, ok=1 and both EW left pockets empty, it sometimes keeps phase 3
  (5 s at a time, up to 4 times in a row) while other lanes have queues (36 of 842 such recorded states).
* **Bug 2, lone-vehicle deadlock (more important).** When only a few vehicles are left (the drain after 3600 s),
  the observation stops changing: t_green clips at 60 s and nothing else moves. The network then either holds a
  green with nobody on it forever (for example a lone vehicle at S1 or N0 while phase 2 is green) or cycles through
  empty phases (for example a lone W0 vehicle: 0→1→3→0…, never 2). If I iterate the network's decisions on
  frozen 1–3-lane drain observations, **the original never serves the waiting lane in 790 of 6948 configurations
  (11%)**. In the simulator this leaves a vehicle in the network until the 5400 s cap (seen in dev episode high #1:
  1 vehicle left, 358 switches instead of about 245).

## What I built

Five rules, each a repurposed layer-1 "detector" unit plus a layer-2 "gate" unit that lowers one logit by M = 30
when it fires. The original network still chooses among the phases that are not suppressed.

| rule | condition (all must hold) | effect | layer-1 unit | gate (layer-2 unit) |
|---|---|---|---|---|
| R1 | phase 3 green, ok=1, d_E2 = d_W2 = 0, at least 1 halting vehicle on another stop-line lane | logit 3 −30 | 50 (full new row) | 1 |
| E0 | ok=1, no *moving* vehicle on any upstream lane (Σ(up_density − up_queue) = 0), d of N0,N1,S0,S1 = 0 | logit 0 −30 | 42 | 63 |
| E1 | same, d of N2,S2 = 0 | logit 1 −30 | 38 | 20 |
| E2 | same, d of E0,E1,W0,W1 = 0 | logit 2 −30 | 5 | 27 |
| E3 | same, d of E2,W2 = 0 | logit 3 −30 | 48 | 1 (OR-shared with R1) |

The E rules mean that in the drain, only phases with vehicles at their stop line get green. When every phase is
empty, all four logits drop by the same amount and the original choice stands. "No moving upstream vehicle" is
only true at t<300 or t≥3600 in the recorded data, so the E rules never act in normal traffic. Using
"moving" instead of "any" upstream vehicle also covers a full stop-line lane with halted vehicles behind it.

Weights (all inputs in [0,1]; one stop-line vehicle = 0.335, one upstream vehicle = 0.0293):
* R1 row: +6 on the 10 other q inputs, −200 on d_E2 and d_W2, +80 on phase3 and ok, bias −161. The other 31 inputs
  are set to 0.
* E rows (sparse; the other old weights are kept and are outweighed by the margins): +30 on ok, −60 on each
  density of the phase's lanes, −600 on the 8 upstream densities, +600 on the 8 upstream queues, bias −22.
* Detectors sit at −1 and go to +1 when firing (min |pre-activation| ≥ 6.6 for the E detectors; R1 is 1.0 in the
  "nobody waiting" corner, still far from the gate threshold).
* Unit 50's outgoing column is zeroed except for its gate. For the other repurposed units the column is kept, and
  each layer-2 bias absorbs the mean shift W2[k,j]·(old mean − (−1)) (one b2 entry per layer-2 unit, shared by all
  detectors).
* Gates: W2[gate, detector] = 12. The gate bias centres the gate's remaining (old) input range, so a gate sits at
  −1 normally and +1 when any of its detectors fires (min margin 7.2). W3[target, gate] = −15. b3 cancels the
  +15 offset and the old mean contribution of the gate unit.

## Checks

* Truth tables: every detector and gate fires exactly when its rule holds. There are 0 mismatches on the 13,306
  recorded states and on 20,000 random synthetic busy and drain states (`verify2.py`). float32 and float64 forward
  passes agree on 100% of argmax decisions (`check_f32.py`).
* Decision changes on recorded ok-states: 83 of 5077. 52 are where a rule fires (the intended changes). 31 elsewhere
  (0.7%) are side effects of freezing 9 hidden units, all near-ties with an original logit gap < 1 (median 0.19). I
  chose the units to minimise these. On the network's own trajectories that is about 4 rule changes and 1.5
  near-tie changes per episode.
* Frozen drain sweeps (`loop_built.py`, `loop_up.py`). With 1–3 waiting lanes × 1–3 vehicles × every start phase,
  the waiting lane is never served in 790/6948 configurations for the original and 0 for the final network (worst
  case 8 decisions). With a full lane plus halted upstream vehicles: 14 failures for the original, 0 for the final
  network.
* Simulator (dev, 9 episodes; I used 3 of the 5 calls):

| file | low | nominal | high | mean | left at cap |
|---|---|---|---|---|---|
| original (reference) | 34.5 | 44.9 | 55.0 | 44.8 | ? |
| r1 (R1 only) | 34.8 | 44.9 | 54.7 | 44.8 | 1 (high #1) |
| **r4 = final (R1 + E0–E3)** | 34.7 | 44.9 | **54.2** | **44.6** | 0 |
| r5 (E0–E3 only) | 34.6 | 45.0 | 54.5 | 44.7 | 0 |

  Most of the gain comes from the high scenario and the drain. R1 adds a little (r4 vs r5, −0.1 on the mean).

## What I expect to go wrong / caveats

* The gain is modest (about 0.2 veh-h on dev) and of the same order as the trajectory-divergence noise between
  near-identical controllers. Any decision change makes the stochastic run diverge, so per-episode differences of
  ±0.3 veh-h are noise.
* Low demand came out 0.1–0.3 above the original in all three variants. Those runs share their trajectories until
  the drain, so this is one noisy sample, not three. It could be a real small cost of the 31 near-tie flips.
* The drain fix only pays off in episodes where the original would deadlock or wander. If the test seeds rarely
  produce such endings, the test gain will be smaller than on dev.
* R1 overrides a confident choice of the trained network. If the hold was in fact deliberate (for example waiting
  for EW left arrivals that are still upstream), R1 costs a few vehicle-seconds per episode.
* The E rules make the network choose among non-empty phases only when no upstream vehicle is moving. While the
  last vehicles are still approaching, the original policy (and its habit of switching to empty phases) remains.
* The detectors use very large weights (up to 600), so the edited units are fully saturated and give almost no
  gradient. That is fine for evaluation but makes them unsuitable for further PPO training.

## Files

`build2.py` (construction), `verify2.py` (truth tables and decision changes), `loop_built.py`, `loop_up.py`,
`loopcheck*.py` (frozen-state service checks), `rulesim*.py` (rule prototypes as plain logit penalties),
`analyse*.py`, `probe.py`, `ties.py`, `units*.py`, `flips.py` (analysis), `check_f32.py`.
Earlier candidates: `r1.npz`, `r3.npz`, `r4.npz` (= final), `r5.npz`.
