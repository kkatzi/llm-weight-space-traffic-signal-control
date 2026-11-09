# Repair of the PPO signal controller after the W2 detector failure

Delivered: `work/weights.npz` (built by `work/make_final.py`, written by `work/finalize.py`).
It is identical to the file scored in simulator call 4. **299 of 7428 parameters changed** (limit 371).

| | low | nominal | high | all 9 |
|---|---|---|---|---|
| start network, failed detector (reference) | 35.5 | 46.8 | 61.8 | 48.0 |
| **delivered network** (dev, call 4) | **35.0** | **45.6** | **55.7** | **45.4** |
| trained network, all detectors working (reference) | 34.7 | 44.9 | 54.4 | 44.7 |

## What I built and why

Two separate faults explain the loss.

1. **The network is blind to the W left-turn pocket.** It is very sensitive to the W2 inputs: setting q/d_W2 to 1
   on recorded states raises the share of phase 3 decisions from 17 % to 65 %. With W2 stuck at 0, phase 3 is
   picked only when the *E* pocket has a car. If E2 is empty, phase 3 is never chosen.
   **Fix: read W2 from E2.** E and W have the same demand, the same geometry, and both pockets are served by the
   same phase (3), so `q_W2 := q_E2` and `d_W2 := d_E2` is the symmetric estimate. Since `x11` and `x23` are
   always 0, this folds exactly into layer 1: `W1[:,5] += W1[:,11]` and `W1[:,17] += W1[:,23]` (128 parameters).
   The patched network on failed inputs equals the original network on inputs with W2 replaced by E2.
   Simulator call 1 with only this change gave 48.0 → 45.6.

2. **The network stalls once the junction is nearly empty.** In an idle junction it holds phase 2 forever.
   Synthetic single-car tests show it never serves one N/S through car (lanes N0, N1, S0, S1; it needs 2–3 cars,
   and even a full S1 lane is not enough). It also never serves a lone W-left car, because that car is now
   invisible. This already happens with every detector working: a single W-left car gives logit 3.07 against
   3.32 for phase 2. During the drain, the last such car waits until the 5400 s cap, about 0.47 veh-h per car.
   Calls 1–2 left 5 and 3 vehicles at the cap, and two recorded failed-network episodes hold phase 2 for about
   28 minutes with 9 W-left cars waiting.
   **Fix: a stall breaker.** Layer-1 unit 57 is the least used unit: replacing it by its mean changes 3 of
   1470 main-hour decisions. I repurposed it as a sharp detector,
   `h1_57 = tanh(200*(t_green + phase2 - 1.975))`, which is +1 only when phase 2 has been held for 60 s or more
   (`t_green` clipped at 1) and −1 otherwise. No recorded main-hour state from any controller reaches that.
   Its layer-2 column `W2[:,57]` is set to the layer-2 shift the original network would get, in the idle phase-2
   state, if it saw 2 W-left cars (`vW = 0.67` on q/d_W2) and about one extra car in each N/S through lane
   (`vN = 0.35` on q/d of N0, N1, S0, S1). That is computed by two forward passes, not fitted. `b2` absorbs the
   old unit's mean (0.456) and the detector's OFF level. `vW` and `vN` were chosen from a table of logit margins
   on synthetic idle states. The result: an empty junction visits phase 3 (+1.2 logit margin) after 60 s, while
   one waiting N/S through car makes it visit phase 0 instead (+2.0). Afterwards the network returns to phase 2
   by its own rules.
   Cost: row 57 of W1 (44 live entries, 2 shared with fix 1), b1[57], W2[:,57] (64), b2 (64), so 171 more
   parameters.

## Which units or weights do what

- `W1[:,5]` and `W1[:,17]`: the original E2 weights plus the original W2 weights, so the network reads E2 twice,
  once as itself and once as the W2 estimate.
- `W1[:,11]` and `W1[:,23]`: unchanged. They multiply an input that is always 0.
- Unit 57 (layer 1): the 60-s phase-2 hold detector. `W1[57,42] = W1[57,45] = 200`, `b1[57] = -395`, all other
  live inputs 0.
- `W2[:,57]` and `b2`: inject the "2 W-left cars + 1 N/S through car per lane" shift when the detector is ON;
  otherwise they reproduce mean-replacement of the old unit 57.
- W3 and b3: unchanged.

## Checks I ran

- Exact equivalence (to 1e-9) of the folded network against the original network on imputed inputs
  (`check_ramp.py`, `check_patch.py`). After the unit-57 swap, 99.74 % of honoured decisions still match the
  original network on imputed inputs (`finalize.py`).
- The detector is ON in 677 recorded states, all of them phase-2 holds at `t_green` = 1 during the drain, and
  in 0 main-hour states. Its OFF level is −1.000 to −0.9999.
- Closed-loop single-car tests on synthetic states, every lane and every start phase (`check_final.py`):
  every lane gets served, where previously N0, N1, S0 and S1 never were. The empty junction cycles 2 → 3 → 2.
- Simulator calls (dev TTS mean; vehicles left at the cap):
  1. W2 := E2 only: 45.6 (5 left).
  2. Plus a linear t_green ramp on W2: 45.8 (3 left; the stuck cars were N/S through, not W-left).
  3. Linear ramps on W2 and N/S through: 45.6 (0 left, but worse in the main hour).
  4. **Delivered: W2 := E2 plus the 60-s stall breaker: 45.4 (0 left).**
  5. Plus 1.5·q_W1u spill-back term (362 params): 45.5.
  6. Shrinkage imputation W2 = μ + 0.3(E2 − μ) (ρ = 0.3 from corr(N2, S2)): 45.5 (high 56.3, one episode 59.1).
  7. Same with ρ = 0.6: 45.4.

  Variants 4–7 are within dev noise of each other. I kept 4 because it is the simplest, has the fewest
  parameters and the best high-demand result. 3 of 10 calls were left unused.
- Trainability is about unchanged (median max-prob 0.981; no dead units). sat_l1 rises to 1.6 % because the
  detector unit is saturated by design.

## What I expect to go wrong

- Phase 3 is still triggered by E-left cars only. A W-left queue while E2 is empty waits for an E-left car, and
  phase 3 can end while W cars are still queued. The recorded statistics show the E2 estimate reads low when W1u
  is queued (≈ 0.55 against ≈ 0.8). This should matter most at high demand, which is where the remaining gap to
  the full-detector network is largest (≈ 1.3 veh-h).
- High-demand episodes are chaotic: single episodes moved by ±1–3 veh-h between near-identical variants, so
  the test means per scenario may differ from dev by a few tenths.
- The stall breaker serves hidden or neglected cars only after a 60 s phase-2 hold. In light traffic a lone
  N/S through car or W-left car can still wait up to about a minute. If a W-left car and an N/S through car
  are both waiting, phase 0 goes first and the W-left car waits another cycle.
- Unit 57's original (small) contribution is gone; in about 0.3 % of decisions this changes the choice.
