# Repair of start_weights.npz for the failed W2 detector

Final file: `work/weights.npz` (same values as `work/c_mirror_brk.npz`, built by `python work/build.py work/c_mirror_brk.npz '{}'`).
It changes **292 of 7428 parameters**: W1 212, b1 4, W2 4, b2 64, W3 4, b3 4.
Dev TTS (simulator call 2): **45.0 veh-h** (low 34.9, nominal 45.3, high 54.9), with 0 vehicles left at the drain cap
in all 9 episodes. The start network got 47.9 (35.9 / 46.9 / 60.9), and the network with every detector working got 44.8.

## What was wrong (found from the recorded states and forward passes)

1. **W2 is invisible.** Phase 3 (EW_left) was chosen only on the E2 pocket's evidence. When W2 filled up while E2
   did not, W left-turners spilled back into W1u and blocked the W straight traffic too. This cost the most in high demand.
2. **Deadlocks of the deterministic policy.** These appear in near-empty states (`work/probe.py`):
   - empty junction: the network cycles 0↔1, or holds 2 forever. A hidden W2 car is never served (this is what
     happened in the actuated and start-network episodes that hit the drain cap);
   - a lone car on N0, S0 or S1 while phase 2 is green: phase 2 is held forever;
   - a lone car on W0: the network cycles 3→0→1→3 and never gives phase 2;
   - halted cars in W1u (left-turners queued behind a full, invisible W2) while phase 2 is green: phase 2 is held forever.
   Baseline dev run: 9 vehicles stuck at the cap over 4 episodes (6 in one high-demand episode).

## What I built

**Part A: impute W2 from the opposing pocket E2 (128 params, W1 columns q_E2 and d_E2).**
E2 and W2 are served by the same phase, have identical geometry and identical demand (the directional bias scales E
and W together). So the dead inputs are replaced by their mirror: `W1[:,q_E2] += W1[:,q_W2]` and
`W1[:,d_E2] += W1[:,d_W2]`. The network now computes exactly what it would compute if q_W2 = q_E2 and d_W2 = d_E2.

**Part B: drain/deadlock breaker (164 params).** Four layer-1 units are rewritten as triggers, one per phase p:
- units 50, 42, 48 and 51 are u_0, u_1, u_2 and u_3. Each row gets: phase_p +30, ok +30, bias −60 + 9,
  t_green +12, every visible stop-line density −3·(22.4/7.5) per lane, every upstream density −3·(256/7.5)
  per lane (so the pre-activation is about 3·(3 − vehicles visible + 4·t_green)), and the phase's own stop-line
  lanes an extra ×7 penalty. A unit is ON (+1) only when phase p is green, ok = 1, phase p's own stop-line lanes
  are empty, and the junction holds at most about 2–3 visible vehicles. With a very long green (t_green ≈ 1) it
  tolerates up to about 7 vehicles. Its other old weights are left as small noise.
- Layer-2 units 25, 14, 52 and 63 are the routers r_0..r_3: `W2[r_p, u_p] = 10`, so r_p rests at −1 and flips to
  +1 when u_p fires. `W3[(p+1)%4, r_p] += 10` and `b3[(p+1)%4] += 10`, so when it fires the next phase in the
  0→1→2→3 rotation gets about +20 on its logit, and nothing changes while it rests.
- Compensation: the pinned units used to vary around a mean. `b2 += W2[:,u]·(mean(h_u)+1)` and
  `b3 += W3[:,r]·(mean(h_r)+1)` (means taken on the recorded states) keep the rest of the network's input unchanged on average.
- Units were picked as the ones whose pinning changes the fewest decisions (ablation on the recorded states).

During the main hour the visible load never falls below about 7 vehicles (1% quantile, recorded states), and the
network never holds a green longer than 37 s. So the triggers never fire between t = 60 s and 3600 s (checked: 0 of
12744 states; the closest pre-activation was −0.55). They act only in the first minute and in the drain, where they
turn any stuck state into a short-green rotation that serves every lane, including the invisible W2.

## Checks run

- `work/check.py`: edit size; decision agreement on the recorded states. In the main hour, 98.2% of decisions match
  the original network fed the mirrored input; the 1.8% that differ come from pinning, not from triggers. Synthetic
  deadlock probes: lone 1–3 cars in every lane from every phase, the empty junction (must reach phase 3), and 1–8
  halted cars in W1u. There are no deadlocks left, against 24 probe cases for the start network.
- Simulator calls (6 of 10 used):

| call | weights | mean | low / nom / high | left at cap |
|---|---|---|---|---|
| 1 | start (baseline) | 47.9 | 35.9 / 46.9 / 60.9 | 9 |
| 2 | **mirror + breaker (final)** | **45.0** | 34.9 / 45.3 / 54.9 | 0 |
| 3 | breaker only | 46.7 | 35.8 / 46.7 / 57.7 | 0 |
| 4 | 0.5·E2 + 0.5·mean imputation + breaker | 45.6 | 35.3 / 45.7 / 55.9 | 0 |
| 5 | 1.25·E2 + breaker | 45.3 | 35.1 / 45.2 / 55.5 | 0 |
| 6 | mirror + 3·q_W1u spill-back term + breaker | 45.0 | 35.0 / 45.3 / 54.7 | 0 |

  The breaker is worth about 1.2 veh-h and the mirror about 1.7. α = 1 (exact symmetry) beat both 0.5 and 1.25.
  The W1u term gave no gain, cost 64 more parameters and mixes two causes, so it was dropped.
- No optimiser, regression or search on weights: every value is set by construction. The only data-derived numbers
  are means of unit activations and the choice of α from three hand-picked values.

## What I expect to go wrong

- The mirror is only right on average. When the two pockets differ by chance (Poisson arrivals), W2 is still over-
  or under-estimated, so the result stays a little behind a working detector in low and nominal demand.
- In very light traffic (≤ 3 visible cars while the green phase is empty), the rotation forces the *next* phase even
  if it is empty, which wastes about 12 s. This does not happen in the recorded main hours; it is possible in quiet
  moments of the low scenario.
- Pinning four units in each layer changes about 1.8% of main-hour decisions. These are mostly near-ties, but each one
  is a small unknown.
- The dev gains could be about 0.2–0.3 veh-h optimistic, since the final candidate was picked from four variants on
  the same 9 episodes.
- The network is now very confident (trigger logits up to about 28, saturated routers), so it would be a poor
  starting point for more PPO training. That was not required here.
