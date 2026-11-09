# Repair of the PPO signal controller after the W2 detector failure

Delivered: `work/weights.npz`, built by `work/build.py` (run it to reproduce). **336 of 7428 parameters changed**
(limit 371): W1 190, b1 2, W2 80, b2 64, W3 0, b3 0. Nothing was fitted. Every number comes from a design
rule, symmetry, or a mean of recorded activations, and was checked with forward passes only.

## What went wrong

The network decides on EW_left (phase 3) using both left pockets together. Its local sensitivity to q_E2/q_W2
and d_E2/d_W2 is almost symmetric (about +1.5 and +2.4 on the phase-3 logit). With q_W2 = d_W2 = 0 it sees
only half of the left-turn demand. Under heavy traffic the E and W pockets (3 cars each) fill up. Left-turners then
queue at the head of upstream lane 1 (which also carries straight traffic), so E/W straight traffic is blocked too.
The network keeps EW_through green with nothing moving and N/S queues full, and gridlocks. In the recordings,
4 of the 6 start-network episodes ran to the 30-min cap (one of them stuck at the end with 2 visible cars). About
2,370 recorded states look like that.

## What I built

1. **Stand-in for the dead input: W2 := E2** (128 parameters).
   `W1[:,5] += W1[:,11]` and `W1[:,17] += W1[:,23]`, so layer 1 now sees q_W2 = q_E2 and d_W2 = d_E2.
   Both pockets are emptied by the same phase and filled by equal demand (E and W share the directional factor),
   so E2 is the natural mirror estimate and it follows the demand level on its own. A constant (mean)
   substitute could not do that. With this patch the network picks EW_left in the recorded gridlock states.

2. **Empty-junction watchdog, layer-1 unit 31** (19 W1 entries, b1[31], 16 W2[:,31] entries, b2).
   A lone car in W2 is now invisible. If every visible lane is empty, the network holds its phase and the car waits
   until the 30-min cap. The actuated controller's recorded episode 5 shows exactly this: about 1,650 s of
   "empty" junction before the cap. Unit 31 is redefined as
   `5 - 35*sum(stop-line densities) - 350*sum(upstream densities)`. It is +1 only when nothing visible is on any
   approach, and -1 as soon as one car is visible (one car contributes at most about -10). The unit had the least
   effect on decisions when removed. Its old mean output went into b2. Sixteen layer-2 units, chosen by their
   W3 leverage for phase 3, get W2[j,31] = ±6, so when the watchdog fires they saturate toward EW_left. EW_left
   then wins every empty state by at least 3.1 logits. Empty visible lanes happen in the recordings only at t = 0
   and in the last ~30 s while cars leave on the unobserved exits, so this costs nothing in normal running.

3. **W-side blockage detector, layer-1 unit 0** (whole W1 row 0, b1[0], W2[:,0], b2).
   It fires when EW_through has been green for at least 10 s, at least one car is halting in W1u, and the straight
   lane W1 is empty. On the mirror E side, that signature means the pocket is full in 98–100 % of 241 recorded
   states: a left-turner at the head of lane 1 cannot enter the full pocket. Gates use large weights (300 on
   q_W1u, threshold half-way between 0 and 1 car; -1000 on d_W1; +320 each on phase2 and ok). When it fires, it
   adds to layer 2 the mean shift that "W2 full" would cause compared with the E2 substitute at 1 car. I measured
   that shift on the 241 E-side signature states after a 180° rotation. Unit 0's old mean output went into b2.
   This covers the case patch 1 cannot see: W pocket full while E2 is not.

## Checks run (`work/check.py`, `work/check_strand.py`; output in `work/check_output.txt`)

- edit_size = 336 ≤ 371. float32 save vs float64 build: identical decisions, max logit difference 2e-7.
- Watchdog on 15,014 recorded states: ≤ -0.99996 on every non-empty state and ≥ 0.99986 on the 424 empty ones.
  It stays off on all 2,128 synthetic one-car states, whatever the phase and t_green.
- Blockage unit fires on exactly the 2,717 recorded states that match its signature (0 mismatches). It saturates
  everywhere (|h| ≥ 0.9997). On one-car synthetic states it fires only for a halting car in W1u during EW_through.
- Honoured decisions match the patch-1-only network on 98–99 % of recorded states. The difference is the
  blockage unit firing plus about 0.4 % from giving up the old roles of units 0 and 31.
- Gridlock-like recorded states (EW_through ≥ 10 s, E/W stop-line through lanes empty, E1u queued): start
  network picks EW_through in 2354/2372, final network picks EW_left in 2372/2372. Before patch 3 the deep
  gridlock states were close to a tie.
- Rotated scenarios (E-side states turned 180°, so the true W2 is known): agreement with the full-detector
  decision for E2 = 0 / ⅓ / ⅔ / full:
  W-spillback states: start 0.64/0.46/0.46/0.47 → final 0.76/0.82/0.83/0.94;
  W-blockage states: start 0.59/0.40/0.49/0.53 → final 0.79/0.97/0.85/0.91.
- Analogue test on the working S2 detector (pretend it failed): the mirror substitute and the mean substitute
  both keep about 94 % of decisions. That test barely separates the options, so I chose the mirror for its
  closed-loop behaviour: it resets when phase 3 is served.
- One-car "stuck" probes (halting car, unserving phase ≥ 60 s): the final network has no new stuck cases. It has
  28, against 30 for the start network. The remaining ones are quirks of the original policy.

## What I expect to go wrong

- With E2 empty, the network still cannot see one or two W left-turners until E2 fills, W spills back into W1u,
  or the junction empties. Expect a little extra W-left delay, mostly at low demand.
- The E2 stand-in effectively doubles E2's weight, so random E/W imbalance adds noise to EW_left timing. EW_left
  may sometimes run with only E-left cars, or start a bit late.
- Units 0 and 31 gave up their old roles, so about 0.4 % of decisions differ from the network's own choices.
- The blockage shift is an average. At E2 = full it pushes EW_left slightly harder than the true network would
  (0.91 agreement there).
- The worst case is the high scenario: if a gridlock still forms in some episode, that episode's TTS could jump
  by hundreds. The gridlock states now all pick EW_left, but I could not run the closed loop. The high prediction
  includes a small margin for this.
- The original policy's quirks remain: a single halting car can wait under some unserving phases.

## Predictions (mean TTS, veh-h)

low 35.8, nominal 46.3, high 58.0. These sit a little above the full-detector network on the dev episodes
(34.5 / 44.6 / 54.4) and below the fixed-time controller (37.7 / 48.4 / 58.7) at low and nominal.

## Files

`build.py` (construction), `check.py`, `check_strand.py`, `check_output.txt` (verification),
`analog_s2.py`, `mirror_test.py`, `scan_watchdog.py`, `scan_blockage.py` (exploration that led to the choices).
