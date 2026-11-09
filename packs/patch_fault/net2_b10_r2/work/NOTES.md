# Repair of the network for the failed W2 detector

## What I built and why

The W2 detector (west left-turn pocket) now always reads 0, so x[11] (q_W2) and x[23] (d_W2) are dead. The trained
network leans on these inputs heavily: on the recorded states, moving them from 0 to 1 car (0.33) raises the share of
phase-3 picks from 25% to 41%. With the inputs dead, the network thinks the W left pocket is always empty, so it
under-serves phase 3 (EW_left). The recorded data shows the result: under the deployed network, the upstream lane
W1u (which feeds W1 and W2) has halting vehicles about twice as often as its twin E1u (1003 vs 553 states), with
W1 itself nearly empty. That is W2 left-turners spilling back and blocking the main road's straight lane.

The repair puts a **replacement estimate of the two dead inputs into the first layer**. The first layer is linear
before the tanh, so any estimate that is linear in working inputs can be built in exactly. For a proxy input j with
coefficient a (for q_W2) and c (for d_W2), I set
`W1[:, j] += a * W1[:, 11] + c * W1[:, 23]` (and the same into `b1` for a constant). The edited network on the
broken inputs then equals the original network fed the estimate (checked to 1e-14).

The estimate (final, `c06_shrink_phaseaware`):

    q_W2_hat = 0.5*q_E2 + 6*(q_W1u - q_W0u) + 0.15*(1 - phase3)
    d_W2_hat = 0.5*d_E2 + 6*(q_W1u - q_W0u) + 0.20*(1 - phase3)

The three terms:

1. **Opposite pocket, shrunk: 0.5 x E2.** E and W have the same demand and both left pockets are served by phase 3,
   so E2 is the natural stand-in. It is only a partial predictor, though. On the analogous N2/S2 pair (working
   detectors, same phase structure), E[S2 | N2] is about 0.15 + 0.35*N2 outside their phase and about 0 inside it.
   So I used a slope of 0.5, between that conditional mean and a full mirror, plus a constant prior.
2. **Prior 0.15 / 0.20 outside phase 3, 0 in phase 3.** A left-turner can wait in W2 while E2 is empty. Without a
   prior, such a lone vehicle is invisible: in an empty junction the network cycles 0->1->0 and the vehicle is stuck
   until the drain cap. That happened in dev episode low_1103 in call 2. In phase 3 the opposite pocket drains to
   about 0, so the prior is removed there. This also avoids lengthening phase 3, which is what wrecked call 4.
   It is implemented as +b in `b1` and -b in the phase3 column (W1[:,43]), on the same 57 rows, so it cancels
   exactly in phase 3. The 57 rows are those where the W2 inputs have the most influence (|W1[:,11]|+|W1[:,23]|,
   times mean tanh slope, times the norm of the unit's outgoing column). That is about 90% of the influence. The
   other rows are left alone to stay within budget (agreement with the full version: 99.8% of honoured decisions).
3. **Spillback detector: 6 x (q_W1u - q_W0u).** If W1u has halting vehicles while W0u flows, a left-turner
   is blocking W1u because the W2 pocket is full. Using the *difference* is essential. A plain 6*q_W1u term (calls
   2-4) feeds itself. When phase 3 is held, W straight is red, W1u fills with straight vehicles, the estimate
   rises, phase 3 is held again, and the junction gridlocks. That happened in call 4: 1317 vehicles left, TTS 1005.
   In such a jam both W upstream lanes fill, so the difference cancels. In jam probes the c02-c04 networks held
   phase 3 in about 99% of congested states (recorded states forced into phase 3, E2 drained, upstream +0.3); the
   final network holds it in 58%, close to the original network's own 55% when it correctly sees W2 = 0.

## Which weights changed (370 of 7428)

| parameters | count | role |
|---|---|---|
| W1[:, 5] (q_E2) | 64 | +0.5*W1[:,11]: opposite-pocket stand-in for q_W2 |
| W1[:, 17] (d_E2) | 64 | +0.5*W1[:,23]: opposite-pocket stand-in for d_W2 |
| W1[:, 31] (q_W1u) | 64 | +6*(W1[:,11]+W1[:,23]): spillback detector |
| W1[:, 30] (q_W0u) | 64 | -6*(W1[:,11]+W1[:,23]): cancels the detector in symmetric jams |
| b1, 57 rows | 57 | +0.15*W1[:,11]+0.20*W1[:,23]: prior that W2 holds something |
| W1[:, 43] (phase3), same 57 rows | 57 | the negative of that: no prior while phase 3 is green |

The columns W1[:,11] and W1[:,23] themselves are unchanged (their inputs are always 0). Layers 2 and 3 are unchanged.

## Checks

- Exactness: edited net on broken inputs vs original net on imputed inputs, max logit error < 1e-14 (build.py).
  `tools.edit_size` = 370 after the float32 save.
- Probes (work/probe_empty.py, probe_jam.py, probe_jam2.py):
  - Empty junction with only W1u halting: the mirror-only net sits in phase 2 even with 8 vehicles waiting; the
    final net switches to 3 with 1-2 halting vehicles from phase 1 or 2, and with 5 from phase 0.
  - Empty junction: the cycle 0->1->3->0 now includes phase 3.
  - Parametric jams (stop lines full, upstream at 0-1, W1u up to +0.3): the final net always leaves phase 3.
- Hold rates on recorded states (honoured decisions): phase-3 holds 20.8% vs 22.5% for c05 and 18.5% for the start
  network.
- Offline decision changes vs c05: 4.1% of honoured decisions.
- Simulator, 6 of 10 calls (dev mean TTS; low / nominal / high):

| call | candidate | mean | low | nom | high | note |
|---|---|---|---|---|---|---|
| 1 | mirror E2 (x11=q_E2, x23=d_E2), 128 params | 45.9 | 36.5 | 45.8 | 55.5 | 8 vehicles stuck in low_1102 (W2 + W1u, rest in phase 2) |
| 2 | + 6*q_W1u, 192 | 45.3 | 35.3 | 45.3 | 55.2 | 1 stuck (empty 0<->1 cycle) |
| 3 | + 0.2*phase1 prior, 256 | 45.1 | 35.2 | 45.3 | 54.7 | none stuck |
| 4 | shrink 0.5 + constant 0.16 + 6*q_W1u | 150.8 | 35.0 | 45.2 | 372 | **gridlock** in high_1102 |
| 5 | c03 with 6*(q_W1u - q_W0u), 320 | 45.1 | 35.2 | 45.3 | 54.7 | same as c03, jam-safe |
| 6 | **final**: shrink 0.5 + phase-aware prior + difference, 370 | **45.0** | **34.9** | **45.3** | **54.7** | none stuck |

References: deployed broken network 47.9 (35.9 / 46.9 / 60.9); same network with all detectors 44.8
(34.5 / 44.9 / 55.0); fixed-time 48.3; actuated 49.5.

No optimiser, regression or search was used. Coefficients come from construction: symmetry, mean levels, and the
N2/S2 conditional means read off as a slope and offset. The spillback gain of 6 was set so that 3 halting vehicles
(0.088) count as about half a pocket.

## What I expect to go wrong

- **Lone W2 vehicles while the junction rests in phase 2.** An empty junction in phase 2 stays there, and nothing
  observable shows 1-3 left-turners waiting in W2 until they spill into W1u. Each one stranded at the end of the
  demand costs up to 0.5 veh-h (until the 30-min cap). Fixing it needs a phase-2 prior of about 0.4, which
  changes 25% of normal phase-2 decisions, so I did not do it. It did not happen in the 9 dev episodes with the
  final net; across 30 test episodes it may happen once or twice.
- **Asymmetric jams.** The difference term is only small when both W upstream lanes are jammed. If W1u is far more
  jammed than W0u for a long time, phase 3 can be held longer than it should. In probes it still releases by 60 s.
- **Negative estimates.** If W0u halts while W1u does not (rare: 0.5% of recorded states), the estimate goes
  negative and phase 3 is suppressed until phase 2 clears W0u.
- The dev gains over the c03/c05 variants (about 0.1 veh-h) are within the episode-to-episode noise. The large,
  real gains are against the deployed network in high demand (60.9 -> 54.7) and nominal (46.9 -> 45.3).

## Files

- `weights.npz`: final network (same as `c06_shrink_phaseaware.npz`).
- `build.py`: folds a linear imputation into W1/b1 (full columns). `build2.py` adds the row-limited prior.
- `analyse.py`, `probe_empty.py`, `probe_jam.py`, `probe_jam2.py`: the offline checks.
- `c0*.npz` / `c0*.json`: every candidate sent to the simulator and its imputation spec.
- `predictions.json`: predicted test means.
