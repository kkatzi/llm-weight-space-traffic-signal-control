# Patch for the failed W2 detector

## What I built and why

The W2 detector (west left-turn pocket) always reads 0, so inputs x[11] (q_W2) and x[23] (d_W2) are dead. The
network was trained with them. Without them it under-serves phase 3 (EW_left). In the recorded start-network runs,
4 of 6 episodes reached the 30-minute cap. They all failed the same way. West left-turners filled the 3-car pocket
and backed up into W1u, which blocked the west straight traffic. The east pocket E2 filled and backed up into E1u
in the same way. The network then held phase 2 (EW_through) indefinitely (`t_green` = 1) with the E/W through
stop-line lanes empty, so nothing moved. One lighter episode also stalled at the end, holding phase 2 with one car
in N0 and one in E2.

**Fix: mirror imputation.** E and W carry the same demand (750 veh/h, 30 % left). Phase 3 serves both left pockets
at once, so they fill and empty together. The east pocket E2 is therefore the best single stand-in for the
invisible west pocket. I add the dead W2 weights to the E2 weights:

    W1[:, 5]  += W1[:, 11]     # q_E2 column now also carries the q_W2 weights
    W1[:, 17] += W1[:, 23]     # d_E2 column now also carries the d_W2 weights

This is exactly the original network evaluated on observations with q_W2 := q_E2 and d_W2 := d_E2. I checked that
the logits match to within 1.4e-7. Nothing is fitted; the change follows from the E/W symmetry.

**Parameters changed: 128 of 7428** (2 columns of W1). No other weights, biases or layers were touched.

## Which units and weights do what

- Only the 64 first-layer units' weights on q_E2 and d_E2 change. Each unit now gets
  `W1[j,5]·q_E2 + W1[j,11]·q_E2` (and the same for d). So the network reads "the W pocket holds what the E pocket
  holds", and all its learned logic (switch timing, phase choice) stays as trained.
- The average sensitivity of the logits to the E2 inputs roughly doubles toward phase 3 (q_E2 +1.7 → about +3.4,
  d_E2 +2.8 → about +5.7 on the phase-3 logit). The network already gave phase 3 a positive response to
  W1u queue/density (+2.0/+3.2), and that stays as a partial backup cue for west spillback.

## Checks run (scripts in `work/`; output in `checks_output.txt`)

1. `make_weights.py`: `tools.edit_size` = 128 changed. Patched logits on the faulty inputs equal the original
   logits on mirror-imputed inputs (max diff 1.4e-7, argmax agreement 100 %).
2. Gridlock states: in the start network's failed episodes there are 2123 states with phase 2 green ≥ 30 s and
   the E/W through lanes empty. The start network holds phase 2 in 2115 of them; the patched network picks phase 3
   in 2120. At the onset of the episode-3 gridlock (t = 495–530 s), phase 3 now wins by about 3–4 logits.
3. "Wasted holds" (ok = 1, current phase's stop-line lanes empty, other lanes queued, network keeps the phase):
   2200 → 31 over all recorded states. Only 1 of the 31 happens with green ≥ 30 s; the rest are single extra 5 s
   steps with vehicles approaching upstream.
4. Realistic sparse states (only lanes of other phases occupied): the patched network wrongly holds in 2 of 528.
5. Imputation variants compared: mirror, shrunk 0.2+0.5·E2, 0.1+0.75·E2, 1.2·E2. The shrunk version has phantom
   W2 demand when E2 is empty. It caused 3× more wasted holds (34 of them long) and smaller anti-gridlock margins.
   Pure mirror is the best simple choice.
6. Analogue test on the N/S left pockets, where both detectors work: hide S2, reconstruct it from N2, and compare
   with the decision on the true state. Zero-fill (the failure) under-calls the left phase by 3–4×. Mirror agrees
   with the true decision 87–93 % of the time and over-calls the left phase by 0–35 %. That over-call is the safe
   direction of error.
7. Frozen-queue decision chains on sparse synthetic states: 25 → 20 permanent holds. All the rest are original
   network quirks that the full-detector network shares too (e.g. a lone N1 queue while phase 2 is green).
8. `tools.trainability`: no dead units, saturation about the same as before.

Things I considered and rejected:
- Spillback detector unit. Phase 2 green, W1 empty and W1u halted means the W2 pocket is full; on the observable
  sides this signature is 99 % reliable. But even the full-detector network mostly holds phase 2 in those states
  when E2 ≤ 1 car. Building the unit would also mean giving up an existing hidden unit and about 175 more changes.
- Fixing the other sparse-state quirks: they are unrelated to the fault.

## What I expect to go wrong

- The W pocket is only estimated. When W left arrivals run ahead of E arrivals, phase 3 is called late and west
  left-turners can briefly block W1u (west straight traffic). It should clear when E2 fills, about 15–40 s later.
  This should cost a few veh-h at high demand, not cause gridlock.
- Slight over-calling of phase 3 when E2 has cars but W2 is empty: some lost through capacity at high demand.
- Residual risk of a gridlock episode at high demand. The causal states are fixed with wide margins, but I could
  not run closed-loop tests, so I can't rule out other paths into a jam. One gridlocked episode would add about
  50 veh-h to the high-scenario mean.
- End-of-episode stalls from the original network's sparse-state quirks (e.g. a lone car in N1 while phase 2 is
  held) can still happen occasionally.

Predicted mean test TTS: low 35.5, nominal 46.0, high 58.0 veh-h. The full-detector network is about
34.5/44.6/54.4, and fixed-time is 37.7/48.4/58.7.
