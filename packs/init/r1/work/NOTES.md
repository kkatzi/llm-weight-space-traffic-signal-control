# Hand-built PPO starting network for junction J0

## What I built and why

The network implements a **weighted-queue (max-pressure-style) controller with a small keep bonus**. At every decision
it scores each phase by the vehicles it would serve, in vehicle units, and picks the highest score:

    S_p = Σ stop-line lanes of p   3.0·d_lane              (every vehicle within 22 m: halting or about to halt / cross)
            (left pockets × 1.25)
        + Σ upstream lanes of p    34.1·[ 1.0·halting·share_h + 0.25·moving·share_m ]
        + 0.75 · [p is the current green]

- Upstream lane 0 (right + straight) belongs to the through phase only (share 1). Upstream lane 1 (straight + left)
  is shared. Its moving vehicles are split 50/50 between through and left. Its halting vehicles go 35 % to through
  and 65 % to left, because in the recorded data a queue on upstream lane 1 usually comes with a full left pocket
  and a non-full straight lane: left-turners are blocking the lane. E/S/W: 56–71 % of cases; during that
  approach's own through green, 80–90 %.
- **Left pockets × 1.25.** The pockets hold only 3 cars. Once a pocket is full, every new left-turner blocks the
  shared upstream lane, so a full pocket is worth slightly more than three ordinary queued cars.
- **Moving upstream vehicles × 0.25.** If a phase is served now (after 3 s of yellow and at least 12 s of green),
  its vehicles still on the 256 m upstream lane mostly pass without stopping. If it is served one slot later, they
  arrive and wait, but on average about half as long as a car already queued. That argument gives ≈ 0.5. I used
  0.25 because a weight of 0.5 pushed most decisions to EW-through (55 %) and starved the left phases, and short
  left pockets make starvation costly.
- **Keep bonus K = 0.75 vehicle.** It is just under one vehicle. An empty green is left as soon as a single vehicle
  waits elsewhere, and a green that still has vehicles on its lanes is kept until another phase clearly has more.
  At ok=1 decisions (12 s of green) the current phase's stop-line lanes are almost always empty (recorded mean 0.0
  halting, 0.5 vehicles), so in practice it switches at minimum green unless a platoon or queue is still
  discharging, or nobody is waiting.

Why this rule: the tuned actuated controller (45.3 veh-h on dev) turns out to be almost exactly "at minimum green,
switch to the phase with the most halting vehicles": 96 % agreement on its own recorded ok=1 states. My rule keeps
that core and adds three things: vehicles that are about to arrive, a mild extension when the current phase is
still discharging, and spillback-aware handling of the left pockets and shared lane. On the actuated controller's
recorded ok=1 states my network agrees with the actuated decision 82 % of the time. It keeps the current green 3 %
of the time (actuated: 3 %). Its choices are NS-through / NS-left / EW-through / EW-left = 22 / 13 / 41 / 25 %
(actuated: 29 / 16 / 33 / 22 %).

`t_green` and `ok` are not used. A pick made while ok=0 is ignored by the junction anyway.

## Which units / weights do what

Everything is built by `work/build.py` (parameters in `work/params.json`). Nothing was fitted to data.

- **Layer 1, units 0–15** (4 per phase: units 4p…4p+3): `h = tanh(α_j·(S_p − μ_j))` with (α, μ) = (0.30, 1),
  (0.20, 6), (0.12, 13), (0.06, 26). Each row of W1 is `α_j ×` the phase's score vector (table below), and
  `b1 = −α_j·μ_j`. All four phases use the same α_j and μ_j, so the summed response is one increasing function
  of S_p. It is steep for 0–10 vehicles (≈ 0.8 logit/vehicle) and has slow tails: 0.3 logit/vehicle at 15–20 and
  0.07 at 30–40, so heavily congested phases are still ranked correctly.
- **Layer 2, units 0–7** (2 per phase: 2p, 2p+1): `tanh(0.2·Σ_j h1_pj ± 0.2)`, which stays near-linear
  (sat_l2 = 0). Their W2 entries from all other layer-1 units are 0. Those zero weights still receive gradient,
  because the source units are active.
- **Read-out:** `W3[p, 2p] = W3[p, 2p+1] = 8.0`; my layer-2 units' entries for the other phases are 0; b3 = 0.
- Because the whole chain is the same increasing function for every phase, `argmax(logits) = argmax(S_p)`.
- **Untouched:** layer-1 units 16–63, layer-2 units 8–63, and the read-out of those units keep their random initial
  values (W3 entries ~0.001, so they add only ~0.006 noise to the logits). They are alive (dead_l1 = dead_l2 = 0)
  and available to PPO. Changed entries: 1304 of 7428 (W1 rows 0–15, b1[0:16], W2 rows 0–7, b2[0:8], W3[:, 0:8]).

Score weights per raw input (vehicle units; `S_p = Σ coef·x`):

| phase | inputs and coefficients |
|---|---|
| 0 NS_through | d_N0, d_N1, d_S0, d_S1: 2.99 · q_N0u, q_S0u: 25.6 · q_N1u, q_S1u: 7.68 · d_N0u, d_S0u: 8.53 · d_N1u, d_S1u: 4.27 · phase0: 0.75 |
| 1 NS_left | d_N2, d_S2: 3.73 · q_N1u, q_S1u: 17.9 · d_N1u, d_S1u: 4.27 · phase1: 0.75 |
| 2 EW_through | as phase 0 with E/W lanes · phase2: 0.75 |
| 3 EW_left | as phase 1 with E/W lanes · phase3: 0.75 |

## Checks I ran (forward passes and descriptive statistics only; no simulator, no fitting)

- `tools.trainability(w)` on states.npz:
  **median_max_prob 0.865, share_max_prob_above_0.99 0.168, max_abs_logit 8.50**, median_logit_gap 2.14,
  sat_l1 0.004, sat_l2 0.000, dead_l1 0, dead_l2 0. All required limits are met (≤ 0.95, ≤ 0.30, ≤ 15) and so are
  the recommended ones. On ok=1 states only: 0.888 / 0.184. This is close to the 600k-step PPO reference
  (0.93 / 0.16 / 6.5).
  The read-out scale (GAMMA = 8) was chosen by scanning these statistics (`temperature.py`, `slope*.py`).
- Network argmax vs the rule's argmax (`build.py`, `robust.py`): 99.2 % on the recorded states. Agreement is 100 %
  whenever the rule's top-2 margin is ≥ 0.05 vehicle; the remaining disagreements are exact or near ties broken by
  the tiny random read-out. On 8,874 synthetic congested states (upstream counts × 1.5–3.5, stop-line lanes
  filled, scores up to 69 vehicles), agreement is 99.94 % and 100 % for margins ≥ 0.05.
- Behaviour on recorded ok=1 states (`check.py`): see the numbers above. On the fixed-time controller's states
  (long reds) it keeps only 2 % of the time and mostly picks the phase starved by the 75 s cycle.
- Synthetic probes (`check.py`):
  - Empty junction → keep the green.
  - One vehicle waiting elsewhere → switch.
  - Both EW pockets full → EW_left (p = 0.998).
  - Both pockets full plus 6 cars queued upstream on E0u → still EW_left (7.5 vs 6.75), with p(EW_through) = 0.33,
    so PPO can explore the alternative.
  - NS_left green with 9 EW cars waiting → EW_through.
- Exploration scripts (`explore.py`, `phase_stats.py`, `spill_stats.py`, `variants*.py`, `deviations.py`)
  compute descriptive statistics and compare candidate rules on the recorded states. Some of them (`slope.py`,
  `temperature.py`) were written for earlier parameter shapes and may not run with the final `P`.

## What I expect to go wrong

- **Untested in simulation.** The weights were chosen from reasoning and recorded statistics only. A wrong left /
  through balance is the biggest risk: too little left service means pocket overflow, a blocked upstream lane 1, and
  lost through capacity, most of all in the high scenario. The 1.25 pocket weight and 65 % attribution are guesses.
- **The shared lane is ambiguous.** Upstream lane-1 halting is split 35/65 by a fixed rule. When the straight lane
  (not the pocket) is the blocker, typically on the N approach, the left phase gets credit for vehicles it cannot
  serve, and a left green may be held or chosen with few real left-turners.
- **No max green and no notion of waiting time.** A small queue on a low-volume movement, such as a single NS-left
  car, can wait several cycles when other phases always have more vehicles. Total delay should still be fine, but
  individual waits can be long.
- **Upstream counts carry no position.** The 0.25 weight treats a car 250 m away like one 30 m away, so extension
  decisions for platoons are coarse.
- **Training-time behaviour.** With median max-prob ≈ 0.87, PPO's sampled rollouts deviate from the greedy action
  on roughly 10–15 % of decisions. Early training returns will therefore look worse than the deterministic
  controller, as with any exploratory start. Logits are bounded (|logit| ≤ ~10 even in heavy congestion) and
  layer 2 is unsaturated, so early updates should be smooth.
- **Expected TTS:** about the tuned actuated level: low 34.4, nominal 45.0, high 56.8 veh-h (`predictions.json`).
  Uncertainty is roughly ±2 veh-h, skewed upward in the high scenario because of the spillback risk.
