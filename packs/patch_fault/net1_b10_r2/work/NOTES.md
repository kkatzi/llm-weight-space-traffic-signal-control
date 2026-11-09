# Repair of the network for the failed W2 detector

Final file: `work/weights.npz` (variant **C3**, built by `python work/build.py C3 work/weights.npz`).
**329 of 7428 parameters changed** (limit 371). W3 and b3 are untouched.

Dev result (eval call 4): **mean TTS 45.0 veh-h**, with low 34.7, nominal 45.2 and high 55.1. No vehicles were left
at the drain cap. For comparison: the start network with the failed detector scored 214.3, fixed-time 48.3, actuated
49.5, and the network with every detector working 44.5.

## What went wrong

From the recorded start-network episodes (`states.npz`): at medium and high demand the network locks into phase 2
(EW_through) forever. In that state the left pockets E2 and W2 are full, and left-turners waiting for a pocket block
the upstream through+left lanes E1u and W1u, which fill to 1.0. N and S are full too. The network never sees
W2 > 0. With `q_W2 = d_W2 = 0` it reads "W1u is jammed" as "serve EW through", which cannot drain W1u because a
left-turner sits at its head. Setting `q_W2 = d_W2 = 1` in those recorded states flips 535 of 539 decisions to
phase 3, so the trained network does know what to do once it gets the information.

A second, smaller failure: when traffic is light, the network parks in phase 1, 2 or 3 indefinitely. Cars in W2 are
invisible, so they can wait until the 30-minute drain cap.

## What I built (all by construction; no fitting)

1. **Mirror of the opposite pocket (64 params, W1 column `d_E2`).** E2 has the same demand as W2 and is served by
   the same phase 3. I use `q_W2_hat = d_W2_hat = d_E2`. Because layer 1 is linear in x, this is exactly
   `W1[:, d_E2] += W1[:, q_W2] + W1[:, d_W2]`. A two-column version (q from q_E2, d from d_E2; variant A, 128
   params) agrees with this one-column version on about 99% of recorded decisions. I compressed it to free budget.
   The dead columns 11 and 23 were left as they are, since their inputs are always 0.
2. **Three layer-1 units repurposed as detectors.** I picked the least important units: mean-ablating any one of
   them changes only 16–17 of 4683 recorded decisions. For each unit I overwrite only 2 input weights and the bias,
   using large gains. The unit's other old input weights stay and act as a bounded residual (|residual| < 3 on the
   recorded states), which the gains swamp, and the bias is centred on that residual. Each unit's outgoing W2
   column is replaced. Its old mean contribution, plus half the injection, is folded into b2. So a unit adds
   nothing when off (h = -1) and adds its injection vector to the layer-2 pre-activations when on (h = +1).
   - **Unit 31, timer** (`t_green` weight +120, `phase3` weight -200, bias at 29.5 s): fires when a green of phase
     0, 1 or 2 is older than 29.5 s. Its injection is the mean change in layer-2 pre-activations that `q_W2 = d_W2 = 1`
     would cause, averaged over recorded fixed-time and actuated states plus idle states. In plain words: "if
     nothing has happened for 30 s, assume the W2 pocket has filled", so idle periods get a phase-3 visit.
     The network itself almost always switches at the 12 s minimum, so this unit is almost never on in busy
     traffic.
   - **Unit 0, phase-3 maximum green** (`t_green` +120, `phase3` +200): fires when phase 3 has been green for more
     than 29.5 s. Its injection lowers logit 3 along the part of W3[3] that is orthogonal to W3[0..2]. In practice it
     saturates layer 2 and sends the signal to phase 0. That is crude but robust: a gentler version starved more
     single-car states in `starve.py`, and one with equal starvation was no better. Without this unit, the timer
     above would leave an idle junction parked in phase 3, where single cars on N1, E0, S1, W0 or W1 can wait
     forever.
   - **Unit 4, spillback detector** (`q_W1u` +400, `phase2` +500): fires when phase 2 (EW through) is green and at
     least one vehicle halts on W1u, the upstream lane that feeds W1 and the W2 pocket. On the approaches with working
     detectors, the same signature (through phase green, at least one halting car on lane 1u) means the left pocket
     is full in 96–100% of states. Its injection is 0.75 × the "W2 = 1" effect over recorded phase-2 states.
     I picked 0.75 from {1, 0.75, 0.5, 0.35} by agreement with the start network fed `q_W2 = d_W2 = 1` on the
     recorded states where the unit fires. The full 1.0 double-counts, because the mirror usually already reads 0.67
     or more there. This is one hand-chosen scale, not a fit.
3. **b2 (64 params)** takes the compensation terms above.

## Checks run

- `tools.edit_size`: 329 changed (W1 70, b1 3, W2 192, b2 64).
- `work/check.py`: agreement with the earlier variants on recorded states with green under 27 s (above 99% between
  C and C3). The recorded deadlock states now give phase 3 in 539 of 539. I also swept idle states and checked
  single-lane "hold" states.
- `work/starve.py`: a closed-loop replay of the switching rules on frozen synthetic drain states (1–3 cars on one
  lane, or one car on each of two lanes, from every phase). The start network starves 224 of 816 cases (it never
  serves the lane). A (the mirror alone) starves 198, and B, C and C3 starve 10. The 10 remaining are all "2–3 cars
  halting on N1 and nothing else": the original network prefers phase 2 for N1-only queues, which causes a
  2 → 3 → 2 cycle. That quirk is inherited from the trained network and appears there with all detectors working.
- float32 and float64 forward passes agree on 100% of argmax decisions (max logit difference 2.6e-4). This matters
  because unit 0's column is large (|W2[:,0]| up to about 544, with the same magnitude cancelling in b2).
- Simulator calls (5 of 10 used, all 9 dev episodes each):

| call | variant | params | low | nominal | high | mean | vehicles left at cap |
|---|---|---|---|---|---|---|---|
| 1 | A: mirror (2 columns) | 128 | 34.8 | 45.6 | 57.5 | 46.0 | 2 and 5 in two episodes |
| 2 | B: mirror1 + timer + phase-3 max green | 348 | 34.9 | 45.0 | 56.1 | 45.3 | 0 |
| 3 | C: B made cheaper + spillback unit | 329 | 34.7 | 45.2 | 55.3 | 45.0 | 0 |
| 4 | **C3**: C with spillback injection × 0.75 | 329 | 34.7 | 45.2 | 55.1 | **45.0** | 0 |
| 5 | C4: C3 with the timer at 19.5 s | 329 | 34.8 | 45.1 | 55.8 | 45.2 | 0 |

The rejected variant C2 (gating the spillback unit on d_E2 < 0.5) was never evaluated in the simulator. One tanh
unit cannot AND three conditions over these input ranges, and the offline check showed it firing almost everywhere.

## What I expect to go wrong

- The mirror is only a proxy. When W2 is full and E2 empty (and no spillback has formed yet), W left-turners wait
  longer than with a working detector. When E2 is full and W2 empty, phase 3 is served a little more than needed.
  The remaining gap to the all-detector network, about 0.5 veh-h and mostly at high demand, is that.
- Isolated drain states with only 2–3 cars on N1 can still cycle 2 ↔ 3 (an inherited quirk). If the last vehicles
  of an episode happen to be like that, they wait until the drain cap, roughly 0.5 veh-h per vehicle. It never
  happened in the 36 dev episodes I ran with B, C, C3 and C4.
- The timer interrupts any green of phase 0, 1 or 2 longer than about 30 s with a phase-3 visit. If test traffic
  ever made long greens useful, that would cost a little. With this network, greens are almost always 12 s.
- Dev has only 3 episodes per scenario. Differences of 0.2–0.5 veh-h between C, C3 and C4 are within episode noise.
  For the predictions I added a small margin to the dev numbers for tail risk.
