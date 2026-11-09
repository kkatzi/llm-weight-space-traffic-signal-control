# Notes: patch of net1 (budget 371 parameters)

## What I found about the trained network

- It is a **phase picker, not a green-time controller**. On its own recorded episodes it switches at the first
  honoured decision (green age 12 s) 95 % of the time. Queues on the served lanes are gone by about 7 s of green.
  So the only decision that matters is *which phase to serve next* in each 15 s slot (12 s green + 3 s yellow).
- Its choices are sensible. It never switches to an empty phase while others wait. It gives priority to full
  left pockets. It agrees about 81 % of the time with an exchange-argument index (below) with G ≈ 10 s.
- The remaining problem is **noise in close calls**. 12–15 % of honoured decisions change when the junction is
  rotated 180° (N↔S, E↔W). Test demand is symmetric under that rotation, so at least one decision in each such pair
  is suboptimal. About 8 % of honoured decisions have a top-1 vs top-2 logit gap below 0.5.

## Principle used

Exchange argument for two phases served in consecutive 15 s slots: serving p before q is better when
`H_p - G*lambda_p > H_q - G*lambda_q`. Here `H` is the number of vehicles waiting for the phase and `lambda` is
its arrival rate. A phase with heavy arrivals is better served second, because its green then also covers its new
arrivals.

- `H_p`: vehicles on the phase's stop-line lanes (density × 2.99) + halted vehicles on its own upstream lanes
  (× 34.1). The shared upstream lane 1 (straight + left) counts half for the through phase and half for the left
  phase.
- `lambda`: nominal rates 0.175 / 0.075 / 0.292 / 0.125 veh/s for phases 0..3.

Evidence from the recorded data (statistics only: regression-adjusted comparisons of the next 60–120 s of
vehicle counts, with state controls and episode × time fixed effects; nothing was fitted to weights):

- On the network's own states, decisions where it deviated from the index were followed by more delay. The effect
  was about −30 to −75 veh·s per decision in favour of the index, 2.5–5 standard errors.
- The effect was strongest where the network's own preference was weak (logit gap < 1–2): about −100 to −240
  veh·s.
- Honoured "hold" decisions were followed by more delay (+37 ± 15 veh·s at 60 s).
- Checks that did not support other ideas, so I dropped them:
  - "Always serve the left phase when the pocket blocks the shared lane" was not supported.
  - A count-only index with a +λ term (serve busy phases first) was contradicted.

## What I built

A capped nudge added to the logits:

    logit_p += 2.5 * tanh(0.3 * s_c) projected on 3 phase contrasts  -  0.75 * 5 * lambda_p

For small `s` this is `0.75 * (H_p - 5*lambda_p)`: 0.75 logit per waiting vehicle, with G = 5 s. G = 5 is between
the network's own G ≈ 10 and G = 0. It halves the bias toward low-demand phases when the junction is empty. Each
contrast is capped at ±2.5 logits, so confident decisions are kept. In congestion the network's own reading of
blocking still dominates.

Three contrast channels:

| channel | signal s | logit direction | layer-1 unit (signal) | layer-2 unit (relay) |
|---|---|---|---|---|
| A | (H0 − H2)/2 (NS thr vs EW thr) | (+1, 0, −1, 0) | h1 #31 | h2 #25 |
| B | (H1 − H3)/2 (NS left vs EW left) | (0, +1, 0, −1) | h1 #5 | h2 #46 |
| C | (H0 + H2 − H1 − H3)/4 (through vs left) | (+1, −1, +1, −1) | h1 #27 | h2 #4 |

- **Layer-1 unit k:** `W1[k] = 0.01 × (coefficients of s on the inputs)`, `b1[k] = 0`. Its output is
  `tanh(0.01 s)`, which is almost linear and small. Its leak into the other layer-2 units through the unchanged
  `W2[:, k]` is therefore negligible.
- **Layer-2 relay j:** `W2[j, :] = 0` except `W2[j, k] = 30`, `b2[j] = 0`, which gives `h2_j ≈ tanh(0.3 s)`.
- **Output:** `W3[:, j] = 2.5 × direction`.
- **`b3`:**
  - adds the old mean contribution of the three replaced relay units (`W3_old[:, j] × mean h2_j`);
  - adds the constant `−0.75 × 5 × lambda`.
- **`b2`:** 19 entries compensate the largest mean input lost from the three replaced layer-1 units.

I chose the six repurposed units because removing them disturbs the original decisions least. This was a discrete
choice made with forward passes on the recorded states.

**Parameters changed: 371 / 7428** (W1 138, b1 3, W2 192, b2 22, W3 12, b3 4). Checked on the reloaded float32
file.

Scripts:

- `build.py`: the construction.
- `make_weights.py`: final build and save.
- `checks.py`: verification.
- `select_units*.py`: unit choice.
- `explore*.py`: the analysis behind the decisions above.

## Checks run (`work/checks.py`)

- `edit_size` = 371 (limit 371). All values are finite. Max |logit| went from 12.2 to 14.2.
- Honoured decisions changed vs the original: 6.3 % (own states), 6.0 % (fixed-time states), 6.0 % (actuated
  states).
  - The ideal (exact) nudge would change 5.7 %.
  - The built network differs from the ideal on 1.0 % of honoured states. This is collateral from the repurposed
    units, and it falls in near-ties.
- Of the 96 changed decisions on the network's own states:
  - 89 % move to a phase with more waiting vehicles (+2.2 vehicles on average);
  - the median original logit gap of these decisions was 0.55.
  - Holds fall from 71 to 44.
  - Phase-pick mix: [362 273 501 385] → [401 271 502 347].
- 180° rotation inconsistency on honoured states: 0.123 → 0.101.
- Synthetic states behave as before:
  - empty junction;
  - a full NS through queue → 0;
  - full EW pockets → 3;
  - full NS pockets → 1;
  - a long EW upstream queue → 2;
  - everything full → 2.
- Not run: any simulation (not allowed). The effect on TTS is therefore an estimate, not a measurement.

## What I expect to go wrong / risks

- **The gain is small and uncertain.** I expect about 0.2–0.5 % lower TTS: about 6 % of honoured decisions
  change, each worth a few tens of veh·s if the evidence is right. Changing any decision makes the trajectories
  diverge, so the measured difference on 30 episodes includes noise of a similar size (±0.3 % or so). The result
  could come out slightly worse even if the nudge helps on average.
- **The evidence is observational.** The regression controls are linear and selection bias cannot be ruled out.
  The network may deviate from the index in states that are hard for reasons the controls miss.
- **The index ignores blocking dynamics.** These are left-turners spilling out of the 3-car pocket and blocking
  the shared upstream lane. The ±2.5 cap leaves those cases to the network, but moderate cases can be nudged the
  wrong way.
- **The λ term uses nominal rates.** At low demand it slightly over-favours left and NS phases when queues are
  tiny. At the empty start of an episode (t = 10 s) the network already picked NS left; the nudge makes that
  stronger, which costs a few seconds once per episode.
- **Fewer holds.** If some of the network's holds were deliberate platoon-serving (the H index ignores moving
  vehicles upstream), dropping them costs a little at low demand.
- **Collateral changes.** Repurposing six hidden units changes about 1 % of decisions in ways unrelated to the
  index.
