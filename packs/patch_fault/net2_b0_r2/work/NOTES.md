# Repair of the PPO signal controller for the failed W2 detector

## What I built and why

The W2 detector (west left-turn pocket) always reads 0, so x[11] (q_W2) and x[23] (d_W2) are 0. Forward passes show
that the trained network uses the two left pockets roughly through their **sum**. P(pick phase 3) for each
(E2, W2) level pair is close to symmetric (`work/explore.py`, table 1). With W2 stuck at 0, the network sees about
half of the real left-turn pressure. It calls phase 3 (EW_left) too late, and the W2 pocket (3 cars) overflows into
upstream lane W1u, where the waiting left-turners also block W straight traffic. The recorded start-network episodes
show this. With phase 3 not green, "W1u has a halted vehicle while E2 is not full" occurs in 11.8 % of states. The
E-side mirror of that event occurs in 0.2 %.

I made two edits, both by construction (no fitting, no search):

1. **Impute W2 from its mirror lane E2** (128 parameters). E and W have the same demand and the same 30 % left
   share, and both pockets fill and empty under the same phase 3. So x11 := x5 (q_E2) and x23 := x17 (d_E2). This is
   folded into layer 1: `W1[:,5] += W1[:,11]`, `W1[:,17] += W1[:,23]`. The dead input's own columns are left
   unchanged; they multiply a 0.
2. **A "W2 is spilling back" detector in layer-1 unit 50** (175 parameters, 2 of them overlapping edit 1, so 301
   in total).
   - Mirror evidence: on the E side, when a vehicle is halted on E1u and phase 3 is not green, E2 is full in 97.8 %
     of start-network states and 96.7 % of actuated states.
   - Unit 50 now computes `s = tanh(120*q_W1u - 130*phase3 - 1.75)`: about -0.94 off, and about +0.94 as soon as
     one vehicle (q = 1/34.13) halts on W1u while phase 3 is not green. All its other input weights are 0.
   - Its outgoing weights `W2[:,50]` are the mean layer-2 pre-activation shift that the **original** network shows
     when (q_W2, d_W2) moves from the imputed (q_E2, d_E2) to (1, 1). The mean is over the 1133 recorded states
     where the detector fires and E2 is not full, divided by (s_on - s_off). This is a forward-pass statistic.
   - Unit 50 was chosen because replacing it by its mean activation changes the fewest honoured decisions (0.17 %,
     `work/analyse_units.py`). Its old mean contribution is put back through `b2`, which also absorbs the detector's
     off level.

## Which weights do what

| weights | role |
|---|---|
| `W1[:,5]`, `W1[:,17]` | original E2 weights plus the W2 weights: E2 counts for both pockets |
| `W1[50,:]`, `b1[50]` | spillback detector (q_W1u > 0 and not phase 3) |
| `W2[:,50]` | "W2 is full" shift into layer 2 when the detector fires |
| `b2` | keeps old unit 50's mean contribution, cancels the detector's off level |

Everything else is unchanged. `tools.edit_size` gives 301 of 7428 changed (4.05 %, limit 371).

## Checks run (`work/check.py`, `work/explore.py`)

- Edit size 301; the file reloads from float32; shapes and finiteness pass in `tools.save`.
- The detector is on exactly when phase is not 3 and q_W1u > 0 (100 % match on recorded states; never in between).
- On detector-off states, the new net matches "start net with W2 := E2" in 98.8 % / 99.4 % / 99.9 % of honoured
  decisions (start / fixed / actuated data). The rest comes from removing old unit 50.
- On spillback states, the reference is "start net with W2 = full". The new net matches it 92.4 % / 87.7 % / 91.6 %
  of the time; the start net matches it 48 % / 80 % / 50 %. By E2 level the match is 82 % (E2 = 0), 92 %, 94 % and
  92 % (E2 = 1).
- Phase-3 share of honoured decisions on start-network states: 25 % before, 44 % after. That is the copy effect; the
  detector adds about 1.5 points. Deployed, the share should fall again, because the pockets will be emptier.
- Sweeps: P(pick 3) rises with E2 level and with W1u queue. During phase 3, a W1u queue barely affects staying in
  phase 3 (19 % to 25 %, about the same as the start net), so the detector does not extend phase 3.
- Trainability is somewhat worse: layer-2 saturation 0.079 to 0.136 (mostly from the copy), layer-1 saturation
  0.005 (the detector unit by design), median top probability 0.993 to 0.998. No dead units.

## What I expect to go wrong

- Copying E2 doubles its noise. When E2 is full and W2 is empty, phase 3 is called a little too eagerly (about 0.95
  against about 0.67), wasting some green at low demand. When E2 is empty and W2 holds 1–2 cars, nothing reacts
  until W2 spills back.
- The detector assumes "halted vehicle on W1u means W2 is full". That held at 97–98 % under short-cycle control, but
  only 54 % under the long fixed-time greens. If the repaired network ever runs long greens, at very high demand,
  straight-lane overflow on W1u will also push phase 3.
- During phase 3 the network still cannot see leftover W2 queue (the detector is off there), so it may leave phase 3
  a bit early after a W spillback.
- The layer-2 shift is an average; when the detector fires with E2 = 0 it under-shoots (picks 3 in 25 % of cases;
  the reference picks 3 in 42 %).
- All checks are decision agreement on recorded states, not closed-loop TTS; the closed-loop gain is uncertain.

## Predictions

The start net with the failed detector scores 35.9 / 46.9 / 60.9 on dev; with every detector working it scores
34.5 / 44.9 / 55.0. I expect to recover roughly half to two thirds of that gap, most of it in the high scenario:
low 35.2, nominal 45.8, high 57.5 (`predictions.json`).
