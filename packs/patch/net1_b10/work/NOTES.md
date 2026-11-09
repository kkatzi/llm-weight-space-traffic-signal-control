# NOTES: patch of start_weights.npz (net1, budget 10 evaluations)

## What I built and why

The final network is the original network plus a **max-green safety gate** for phases 2 (EW_through) and 3 (EW_left).
If either phase has been green for about 47 s or more, the gate pushes its logit down, so the network switches to
whichever of the other phases it prefers. In normal traffic the gate is off: greens in the network's own recorded
runs never last longer than 37 s.

Why this edit:

1. **What the network does.** At each honoured decision it switches at the minimum green (12 s) about 95% of the
   time. It picks the phase with the most halting vehicles per stop-line lane (99.8% of switches agree with that
   rule), with phase-order tie-breaks. Queues clear inside the 12 s green, so the delay comes from waiting at red.
2. **Its phase choice is already locally optimal (as far as the dev set can tell).** I rescaled how strongly the
   network sees the four left-pocket inputs by α = 0.75 and by α = 1.3 (top 32 W1 weights per column, 256 parameters).
   That shifts choices between through and left phases in either direction. Both were worse on dev (45.15 and 44.87
   against 44.51). The recorded data agree: the actuated controller serves more through traffic, lets the left
   pockets spill back, and ends up with more halting vehicles.
3. **The real defect: stranded vehicles deadlock.** When the junction is otherwise empty, the network holds the
   current green and ignores a few halting cars on some lanes. Examples: a lone car on N1 while phase 1, 2 or 3 is
   green; E0, W0, W1 or S1 under phase 3; E2 or W2 under phase 2.
   - Synthetic decision chains (forward passes only) show the original fails to clear 22–23% of random stranded
     groups of 1–6 cars within 30 min, and 18% of groups of 1–14 cars that spill upstream.
   - In the simulator the original left 1 vehicle at the 30-minute drain cap in dev episode nominal_1103.
   - The α = 0.75 variant got stuck in high_1101 and lost about 3.5 veh-h in that one episode.

   A max-green rule is the standard fix. With it, every probed group clears (worst case 95–130 s).

## Which units and weights do what (110 parameters changed: W1 4, b1 2, W2 40, b2 64)

- **Gate units (layer 1).** Unit 31 is the phase-2 gate and unit 0 is the phase-3 gate. I added A = 21 to
  `W1[j, phase_p]` and B = 60 to `W1[j, t_green]`, and subtracted C = 66 from `b1[j]`.
  - Pre-activation = old pre-activation (range −1.9 to 0.8) + 21·phase_p + 60·t_green − 66.
  - It is −1 whenever phase p is not green, or green for 37 s or less. On all recorded states the pre-activation
    is at most −7.9.
  - It turns +1 once phase p has been green for about 47–50 s (t_green is clipped at 60 s).
- **Old roles of units 31 and 0** are replaced by their mean activation, through `b2[i] += W2_old[i,j]·mean(h1_j) +
  W2_new[i,j]`. These two units were the least influential in layer 1. Replacing them by their means is the only
  change to normal behaviour: 0.29% of honoured decisions over all recorded states, 0.53% on the network's own.
- **Carriers (layer 2, nothing destroyed).** Twenty existing layer-2 units per gate get `W2[k, gate] = ±10`. In
  normal states the gate is exactly −1, and the `b2` term above cancels the −(±10) exactly, so these units are
  unchanged. When the gate fires, each carrier saturates to its sign s_k.
  - Carriers and signs were chosen by a closed-form score in the gated situation (empty junction, phase p green
    60 s): how much driving unit k to s_k lowers logit p compared with the mean of the other three logits, minus the
    spread it creates among those three.
  - The point is to suppress p evenly, so the network's own queue-based preference still picks the next phase.
    A first version scored only on lowering logit p, without the evenness term. It favoured phase 1 and looped 1 ↔ 2
    on a lone N1 car.
  - Carriers for phase 2: 37, 51, 23, 1, 11, 4, 28, 36, 49, 30, 18, 24, 62, 38, 17, 54, 33, 41, 25, 42.
  - Carriers for phase 3: 63, 46, 58, 55, 48, 18, 15, 38, 36, 39, 20, 30, 54, 41, 42, 23, 28, 4, 7, 3.
- **Phases 0 and 1 have no gate.** Their deadlocks always pass through phase 2 or 3 (checked in the chain probes),
  and leaving them out keeps the normal-state change at two units.
- `W3` and `b3` are unchanged.

Reproduce with `python work/build_final.py`. It uses `maxgreen3.py` and `maxgreen4.py`.

## Checks run

- **Recorded-state statistics.** I used `analyze*.py`: phase-duration and transition tables, red-gap distribution,
  queues at decision time, gradient importance of each input, per-unit mean-ablation impact, and the green-age
  distribution (maximum 37 s in the network's own runs).
- **Synthetic probes.** `probe1-3.py` give pairwise phase-preference tables and upstream-queue sensitivity.
  `probe_lone.py` and `probe_chain*.py` run deadlock chains (forward passes on synthetic observations, not a traffic
  simulator).

| stranded-group probe | original never cleared | final never cleared |
|---|---|---|
| 1–3 cars | 1460 / 6496 | 0 (worst 115 s) |
| 1–6 cars plus moving cars, 3 seeds | 343 / 1500 | 0 |
| 1–14 cars with upstream spill | 361 / 2000 | 0 |

- **Simulator calls.** 5 of 10 used (dev mean TTS; original 44.51):

  | call | weights | mean | low | nominal | high | vehicles left at cap |
  |---|---|---|---|---|---|---|
  | 1 | left pockets × 0.75 | 45.15 | 34.48 | 45.15 | 55.82 | 1 (high_1101) |
  | 2 | left pockets × 1.3 | 44.87 | 34.53 | 45.09 | 54.99 | 0 |
  | 3 | 4-gate max-green, dedicated layer-2 units (1.0% decisions changed) | 44.78 | 34.52 | 44.70 | 55.10 | 0 |
  | 4 | unmodified copy of the original | 44.51 | 34.51 | 44.61 | 54.42 | 1 (nominal_1103) |
  | 5 | **final**: 2-gate, non-destructive carriers | **44.63** | 34.43 | 44.75 | 54.71 | 0 |

- **Final weights.** `tools.edit_size` gives 110 changed. `trainability` is unchanged except that the two gate units
  are saturated (sat_l1 0.031, dead_l1 2).

## What I expect to go wrong

- **Chaotic noise is larger than the gain.**
  - Changing even one decision per episode sends a high-demand episode down a different trajectory, with a spread
    of about ±1 veh-h per episode.
  - The deadlock fix is worth roughly 0.5–3.5 veh-h per affected episode. The original showed it in 1 of 9 dev
    episodes, so the expected gain on the 30-episode mean is only about 0.05–0.15 veh-h. A 30-episode test can
    therefore still come out slightly worse than the original by chance.
- **The original looks lucky on dev.**
  - Almost every nominal/high dev episode came out 0.2–0.7 veh-h worse than the original under every variant,
    including the final one, which changes fewer than 1 honoured decision per episode.
  - That suggests the original's dev runs are a favourable draw (perhaps the checkpoint was chosen on them) rather
    than a systematic cost of the edit. I can't prove that on 9 episodes.
- **Gate side effects.**
  - The gate forces a switch after about 47 s of green on phase 2 or 3 even if nobody is waiting. That only happens
    in an empty junction, where it costs nothing, or in a very long main-road platoon, which I never saw.
  - A deadlocked car now waits up to about 50 s plus a possible second gate cycle (worst case about 2 minutes),
    instead of until the cap.
  - Deadlocks held on phase 0 or 1 that never pass through 2 or 3 aren't covered. None showed up in about
    10,000 probed groups.
- **Replacing units 31 and 0 by their means** changes about 0.5% of the network's own honoured decisions, which
  might systematically cost a little in rare states.
