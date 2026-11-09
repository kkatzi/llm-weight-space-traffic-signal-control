# Edit of start_weights.npz: "never serve an empty phase" patch

## What I built and why

**Diagnosis.** On its recorded states (`states.npz`), the trained network behaves as a phase *selector*. It requests a
switch at every honoured decision, so every green lasts the minimum 15 s (12 s green + 3 s yellow), and in normal
traffic it picks a sensible next phase. It usually picks the phase with the most halting vehicles or close to it. The
exceptions are mostly deliberate left-pocket clearances. It never left a phase with vehicles waiting for long. Its
weak spot is **near-empty states**. There, its logits are biased toward phase 2 (EW_through), even when phase 2 has no
vehicles at all.

In 1 of its 6 recorded episodes (episode 2) this produced a **deadlock at the end of the episode**. At t = 3625 s,
phase 2 was green with nothing on its lanes, while one car waited on each of N0, N1 and W2. The network kept choosing
phase 2: logits 2.1 for phase 2 against 1.2 for phase 3 and -1.2 for phase 0. It held phase 2 until the 30-minute
drain cap (about 1780 s), which added roughly 1.5 veh-h of TTS to that episode. The same bias shows up in synthetic
follow-up states: with phase 3 or phase 2 green and cars only on N0 and N1, the original network picks phase 2, which
is empty. Recorded drains of the other controllers also show it choosing empty phase 2 (14 such picks).

**Fix (built by hand, nothing fitted).** I added an additive penalty to the logits. For each phase p:
`logit_p -= M` exactly when phase p has **no vehicle** on any of its stop-line or upstream lanes (density sum = 0).

- When some phase is empty and another is not, the controller can no longer choose (or stay on) an empty phase. Among
  the phases with vehicles, the original network's preference decides.
- When nothing is empty, the penalty is zero for every phase, so the network is unchanged apart from replacing the
  units it uses (see the checks).
- When everything is empty, every phase gets the same penalty, so the decision is unchanged.

Phase lanes. The upstream lane 1 feeds both the straight lane and the left pocket, so it counts for both phases:

| phase | stop-line lanes | upstream lanes |
|---|---|---|
| 0 NS_through | N0, N1, S0, S1 | N0u, N1u, S0u, S1u |
| 1 NS_left | N2, S2 | N1u, S1u |
| 2 EW_through | E0, E1, W0, W1 | E0u, E1u, W0u, W1u |
| 3 EW_left | E2, W2 | E1u, W1u |

## Which units and weights do what

Script: `work/build.py` (function `build()`). Units were chosen as the least important by single-unit mean-ablation
(`work/importance.py`): ablating any one of them changes 0.1–0.3 % of the honoured decisions.

| phase p | layer-1 detector J[p] | layer-2 flag K[p] |
|---|---|---|
| 0 | 60 | 37 |
| 1 | 57 | 0 |
| 2 | 32 | 11 |
| 3 | 13 | 5 |

- **Layer-1 detector J[p]** (`W1[J[p]]`, `b1[J[p]]`):
  - weight -300 on the density of each of phase p's stop-line and upstream lanes;
  - the old weights on the same lanes' halting inputs are kept, because they are zero whenever the density is zero;
  - every other input weight is 0, and the bias is -1.
  - Phase p empty: pre-activation = -1 exactly, so h = -0.762.
  - At least one vehicle (one upstream car = density 0.029): pre-activation ≤ -9.8, so h = -1.000.
- **`b2` (all 64 entries)**: mean compensation for the four replaced layer-1 units. They used to output their recorded
  mean and now output -1 in normal states, so `b2 += W2[:, j] * (mean_old_h1_j + 1)`.
- **Layer-2 flag K[p]**:
  - `W2[K[p], J[p]] = 83.9`; the other 63 inputs keep their old weights.
  - `b2[K[p]]` is set so the pre-activation is about -10 in normal states and +10 when phase p is empty.
  - The flag is therefore exactly ±1. It was verified on all recorded states and on 80,000 synthetic states.
- **`W3[:, K[p]]`** = (0, 0, 0, 0) except `W3[p, K[p]] = -12`. That gives +12 on logit p normally and -12 when p is
  empty: a relative penalty of M = 24, larger than any logit gap of the original network (max |logit| ≈ 9.8).
- **`b3`**:
  - adds `W3_old[:, K] @ mean(h2_old[:, K])`, so the four replaced layer-2 units are replaced by their means;
  - subtracts 12, which cancels the uniform +12.

**Changed parameters: 252 of 7428** (limit 371): W1 160, b1 4, W2 4, b2 64, W3 16, b3 4.

## Checks I ran

Scripts: `check.py`, `check2.py`, `synth_check.py`, `deadlock_check.py`.

- **Fidelity in normal states** (honoured decisions, no phase empty, 4744 recorded states): **99.07 %** of decisions
  are the same as the original. The 44 that changed were near-ties: the original's logit gap between its own pick and
  the new pick has median 0.10 and maximum 0.50. Agreement is similar for the network's, fixed-time and actuated
  states (98.9 / 99.2 / 99.2 %).
- **States with some phase empty** (672 honoured recorded states):
  - the new network always picks a phase with vehicles (the original did so in 45 %, mostly because of the deadlock);
  - it matches the original's best choice among the phases that have vehicles in 99.85 % of them.
- **All-empty states**: 100 % unchanged.
- **Synthetic stress test**: 80,000 random observations, from nearly empty to nearly full lanes, plus 20,000
  drain-like ones.
  - Flags: 0 errors, every flag value exactly ±1.000.
  - Whenever at least one phase had vehicles, the chosen phase had vehicles in 100 % of states.
- **Recorded deadlock state** (t = 3625 s): the new network picks phase 3 (serves W2). In the follow-up state, cars
  only on N0 and N1 with phase 3 or phase 2 green, it picks phase 0. The original picks the empty phase 2 in both.
- **Float32 round trip**: identical decisions.
- **Simulator, 1 call of 5** (`work/v1.npz`, the same file as `weights.npz`; sha1 `16304574f79a`):

  | | low | nominal | high | all 9 |
  |---|---|---|---|---|
  | edited network | 34.7 | 44.8 | 54.6 | **44.70** |
  | original (reference) | 34.7 | 44.9 | 54.4 | 44.67 |

  - Per-episode TTS: 35.4, 33.7, 35.0, 46.0, 43.9, 44.4, 56.0, 53.4, 54.5.
  - Vehicles left at the drain cap: 0 in every episode. Switches per episode: 242–245.
  - The original apparently did not deadlock on these 9 dev episodes either, so on dev the patch is neutral, within
    noise.
  - I used no further calls. I had no other change with a solid enough prior to justify selecting it on 9 noisy
    episodes.

## Ideas I considered and rejected

- **Longer greens.** At its honoured decisions the network switches as soon as the current phase's halting queue has
  cleared. Recorded upstream queues stay small (p99 ≤ 7 vehicles per lane, no spillback to the entries), so the junction
  is under-saturated. Both PPO and the tuned actuated controller settled on minimum greens, and a rough delay
  estimate for a 5 s extension came out break-even. Not pursued.
- **Max-halting phase choice.** The network already picks within about 2 halting vehicles of the maximum in 90 % of
  cases. The larger gaps are left-pocket clearances, which are deliberate. Not pursued.
- **b3 phase biases.** These would have been free, since b3 is already changed, but I had no principled direction, so I
  did not use them.

## What I expect to go wrong

- **The gain on test is small and depends on deadlocks occurring.** The fix only matters in episodes where the
  original would have got stuck, or chosen an empty phase, near the end of the drain. That happened in 1 of 6
  training-like recorded episodes and apparently in 0 of 9 dev episodes. Each such episode costs the original
  about 1.5 veh-h or more. Expected gain on the 30-episode mean: roughly 0–0.3 veh-h.
- **Noise from the 0.9 % of changed near-tie decisions** can move the test TTS by about ±0.2 veh-h either way. On dev
  this showed as +0.2 in high and -0.1 in nominal. The test result could therefore come out marginally worse than the
  original if no deadlock occurs in the 30 test episodes.
- **Stuck modes the patch does not cover.** A phase counts as "non-empty" as soon as any vehicle is on its lanes,
  including a car far upstream, or a left-turner in the shared upstream lane 1 that will need the left phase. A stuck
  mode in which the current phase is non-empty but its vehicles cannot move is not covered. I found no such case in the
  data.
- **Further PPO training would be harder.** The trainability diagnostics are worse: max |logit| is now 32,
  `sat_l1` 0.06 and `sat_l2` 0.15, because the flag units are deliberately saturated. Decisions are unaffected.
