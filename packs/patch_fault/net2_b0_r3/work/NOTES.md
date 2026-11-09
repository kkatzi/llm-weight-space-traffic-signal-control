# Repair of the PPO signal network for the failed W2 detector

Deliverable: `work/weights.npz`, built by `work/make_weights.py`. It changes **273 of 7428 parameters**
(W1 195, b1 64, W2 2, W3 8, b3 4; the limit is 371). No optimiser, regression or search was used. Every value
comes from a formula, the network's own weights, or a few plain statistics of the recorded states (means of E2
and a correlation used as a cross-check).

## What I found

1. **The network relies heavily on W2.** On the recorded states, setting q_W2 = d_W2 to one car (0.335) instead
   of 0 changes 15 % of the honoured decisions and raises the share of phase 3 picks from 21 % to 34 %. With the
   detector stuck at 0, the network believes the W left pocket is always empty. So it under-serves phase 3
   (EW_left), and W left-turners back up into W1u, which also blocks the W through traffic.
2. **The trained network has drain-out deadlocks of its own.** In 2 of the 6 recorded start-network episodes, one
   car waited about 1770 s, until the 30-minute cap:
   - a lone N0 car while phase 2 was held for ever;
   - a lone W0 car while the network looped 3→0→1→3 and never picked phase 2.

   In static checks, a lone car on many lanes never gets its phase. An empty junction never visits phase 3, so a
   W left-turner that is last to arrive sits unseen on the failed W2 detector and waits until the cap. The
   actuated recording shows exactly this: in one episode it ran to the cap with an empty-looking network. During
   the demand period, phase 2 never lasted 60 s in the recordings, and phase 0 and phase 1 always ended at 12 s.

## What I built

### Step 1: impute W2 from its mirror lane E2 (`build.py`, 256 parameters)

The E and W approaches have the same demand, and the directional bias applies to both alike. Their left pockets
are also served together by phase 3. So, by symmetry, E2 is the natural proxy for W2:

    W2_hat = 0.8 * E2 + 0.2 * mu(phase)
    mu = E2 mean on the start-network states: (q 0.407, d 0.457) in phases 0-2; (0.056, 0.219) in phase 3

Because x11 = x23 = 0, this folds exactly into layer 1:
- `W1[:,5] += 0.8*W1[:,11]` and `W1[:,17] += 0.8*W1[:,23]`;
- `b1 += 0.2*(W1[:,11]*0.407 + W1[:,23]*0.457)`;
- `W1[:,43]` (the phase-3 one-hot) gets the phase-3 correction.

Columns 11 and 23 are left as they were.

I checked the method on the N/S pair, where the truth is known (`analog_test.py`, `analog_load.py`). I zeroed S2
and imputed it from N2 the same way, then compared against the network with S2 visible:
- With no fix, phase 1 is chosen in 11.7 % of states; the truth is 23.3 %.
- With imputation and ρ = 0.7–1.0, the share is back near 22–25 %, and agreement rises from 0.851 to 0.88.
- By load tercile, ρ = 0.7 under-serves at high load and ρ = 1.0 over-serves at mid load. ρ ≈ 0.85 fits best, so
  I chose 0.8.

The constant part also creates a small phantom left-turn demand in an otherwise empty junction. That is useful:
it makes the idle network visit phase 3, which clears any W2 car the detector cannot see.

### Step 2: two max-green rules (`pathways.py`, `build_full.py`, 17 more parameters)

Each rule is one detector unit in layer 1 plus one relay unit in layer 2, built from the least important units:
- **Detector:** I add three or four huge input weights (K = 400) to an existing layer-1 unit, so it saturates at a
  fixed level everywhere except when its condition holds. Its old inputs no longer matter.
- **Relay:** one weight of 20 from the detector makes a layer-2 unit copy it (−1 normally, +1 under the
  condition).
- **Push:** the relay's W3 column is replaced by a push vector P. b3 absorbs the constant normal-state term, so
  normal logits equal the old logits with that unit held at its mean.

The two rules:
- **A** (layer-1 unit 50, layer-2 unit 1): phase 2 green ≥ 60 s (t_green = 1) → phase-2 logit −12.
- **B** (layer-1 unit 38, layer-2 unit 63): phase 3 green ≥ 27 s **and** E2 empty (q_E2 = d_E2 = 0) → phase-2
  logit +12, phase-3 logit −12. The E2 weight (2.5 K) is large enough to beat the t_green ramp, so a single E2 car
  always blocks the rule.

Units 50 and 1 change the fewest decisions when held fixed (0.15 % and 0.2 %). Unit 63 is also low (0.29 %); unit
38 changes 0.84 % when pinned at −1.

## Checks run (all forward passes; no simulator)

- `tools.edit_size`: 273 changed. The saved float32 file gives the same decisions as the float64 build on all
  14,337 recorded states.
- **Detector saturation:**
  - On recorded states, |h1| > 0.99999 for both detectors outside their conditions and the relays sit at −1.
  - A worst-case check over the whole input box (`bounds.py`, `check_margins.py`) gives pre-activation margins
    ≥ 3 everywhere. The one exception is the green age of 25 s, which only happens at episode start and sits right
    at the threshold anyway.
- **Side effects:** outside the rule conditions, the two rules change 1.4 % of the honoured decisions against the
  imputation-only network. All of those are near-ties (median logit gap 0.09, max 0.5).
- **Rules in normal traffic:** rule A never fires during the demand period in the recordings. Rule B fires in 5 of
  the demand-period states.
- **Phase-3 share:** on demand-period honoured start-network states, phase 3 is picked 45 % of the time instead of
  22 %. That doubling matches the N/S analog, where restoring the hidden pocket also doubled the serving phase's
  share.
- **Deadlock chains** (`chain.py`, `chain_pairs.py`): these freeze the traffic part of the state and step only the
  signal timer.
  - Lone car, 1 or 2 vehicles, halting or moving, every lane, start phase and green age: the start net fails 149
    chains, the repaired net 0 (each lane's phase is reached within 240 s).
  - Two cars on any two lanes: the start net fails 98 of 1616, the repaired net 0.
  - Empty network: the repaired net cycles 0→1→3→2 from every start, so a hidden W2 car is served.

## What may go wrong

- The W2 estimate is only as good as E2. The opposing pockets correlate only about 0.3, so states where W2 is full
  but E2 is empty (or the other way round) still get wrong decisions. Recovery of the 3 veh-h gap will be partial,
  and I expect it to be weakest in the high scenario.
- The phantom 0.2·mu offset adds some phase-3 visits when W lefts are truly absent. This could cost a little in
  the low scenario.
- Rule B cuts phase 3 at 27 s when E2 is empty, even if W2 (unseen) or W1u is still discharging left-turners. This
  was rare in the recordings.
- The deadlock checks use frozen states, not real traffic dynamics.
- The 1.4 % near-tie side effects are untested in closed loop.
- PPO trainability is slightly worse: the detector units are saturated by design.

## Predictions (mean TTS, veh-h)

| scenario | prediction | start net, failed detector (dev) | healthy net (dev) |
|---|---|---|---|
| low | 35.2 | 35.9 | 34.5 |
| nominal | 46.0 | 46.9 | 44.9 |
| high | 58.0 | 60.9 | 55.0 |

These assume about 55–65 % of the gap to the healthy network is recovered, plus about 0.1 veh-h saved per
episode from fewer drain-out deadlocks.
