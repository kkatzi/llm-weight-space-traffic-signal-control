# Repair of the network for the failed W2 detector

## What I found first

The recorded states already have the fault (x[11] and x[23] are 0 in all 14,340 rows). Two of the six
start-network episodes (episodes 0 and 5), and one actuated episode, run to the 5,400 s cap. In their tails
(t > 3,640 s) the observation never changes: phase 2 is green, `t_green` = 1, and every visible lane is empty
(episode 0) or only W1u holds 6 halting cars (episode 5). The network picks phase 2 at all 678 of those
decisions. The cars left in the W2 pocket are invisible, E2 is empty, so phase 3 (EW_left) is never served
again. The W2 cars, and in episode 5 the six cars stuck behind them on W1u, wait until the cap. That is a few
vehicle-hours per affected episode, and it should be worst in the high scenario.

During normal running the fault also shows: W1u has halting cars in 25% of start-network states, against 13%
for the mirror lane E1u. On this junction a queue on upstream lane 1 almost always means the left pocket is
full and spilling back (on the E side, E[q_E2 | q_E1u > 0] is about 0.8 and q_E1 is almost never full then).
So W left-turners are under-served, and their spillback blocks the W straight traffic that shares lane 1u.

## What I built

I feed the network a hand-built estimate in place of the dead inputs:

    q_W2_hat = q_E2 + 0.5 * t_green - 0.1 * phase3
    d_W2_hat = d_E2 + 0.5 * t_green - 0.1 * phase3

Layer 1 is linear in x before the tanh, so feeding x[11] = q_W2_hat is exactly the same as adding
`coef * W1[:, 11]` (and `coef * W1[:, 23]` for d) to the W1 column of each input used. The check reproduces
the original network on imputed inputs to 3e-15.

Changed parameters: 256 of 7,428, all in W1, four whole columns:

| W1 column | added | role |
|---|---|---|
| 5 `q_E2` | `+1.0 * W1[:,11]` | mirror: E2 stands in for W2 (halting) |
| 17 `d_E2` | `+1.0 * W1[:,23]` | mirror: E2 stands in for W2 (vehicles) |
| 45 `t_green` | `+0.5 * (W1[:,11] + W1[:,23])` | "W2 has been red at least this long": breaks the deadlock |
| 43 `phase3` | `-0.1 * (W1[:,11] + W1[:,23])` | removes the time term in phase 3 at its first honoured decision (12 s, t_green = 0.2) |

Why each term:
- **Mirror (E2 → W2).** Phase 3 serves both left pockets, and E and W have the same demand, so both
  pockets fill and empty on the same red-time clock. In training, E2 and W2 moved together, so this
  restores the joint input pattern the network learned. Coefficient 1, no fitting.
- **Time term, A = 0.5.** Outside phase 3, W2 has been red for at least the current green. Left-turn
  arrivals (~225 veh/h nominal) fill the 3-car pocket in about 48 s, which is about 1.25 per unit of
  t_green. I use 0.5, well below that, because the mirror already counts the same fill. Forward passes on
  the empty junction show the network switches 2 → 3 once the W2 input reaches about 0.4 (q = d). With
  A = 0.5 that happens after about 47 s of empty phase 2, or 32 s when 6 cars are halting on W1u.
- **Phase-3 cancel, G = 0.2·A.** In phase 3 the pocket is being served, and the network's W2 response is
  nearly the same in phase 3 as in phase 2 (W2 ≥ 0.3 keeps it in phase 3). So an uncancelled time term would
  hold phase 3. G = 0.1 makes the estimate equal the plain mirror at the 12 s decision, where the network
  decides nearly every phase-3 exit (it practically never extends phase 3 in the data). A larger G
  would make the estimate negative there, which is outside the training range, and it changed the next
  phase in 6% of phase-3 exits.
- **Not used:** a q_W1u term. It is a strong "pocket full" sign outside phase 3. In phase 3, though,
  W1u stays queued while the pocket discharges (E side: E[q_E2 | q_E1u > 0, phase 3] ≈ 0.2), so it would
  hold phase 3. The network already reacts to q_W1u directly.

## Checks I ran (all forward passes; `work/checks.py`, output in `work/checks_output.txt`)

- `tools.edit_size`: 256 changed (limit 371). Only W1 columns 5, 17, 43, 45 differ.
- Equivalence: edited network on raw x = original network on imputed x (max error 3e-15).
- Stuck tails (t > 3,700 s, 1,019 states, start-network eps 0 and 5 plus actuated ep 5): the faulty net
  picks phase 2 at 678/678 start-network tail states. The repaired net picks phase 3 at 998 of 1,019; the
  rest are early-green states that switch a few decisions later.
- Synthetic empty junction: phases 0, 1 and 3 hand over to phase 2 at 12 s. Phase 2 hands over to phase 3 at
  47 s (32 s with 6 halting on W1u). So an empty junction cycles 2 (~47 s) → 3 (12 s) → 2, and W2 is served
  about once a minute.
- Phase-3 hold check (phase 3, otherwise empty): with E2 empty it leaves at t_green ≤ 0.6. It holds only if
  phase 3 has already lasted ≥ 48 s, which never happens in the recorded data (the share of phase-3 states
  where the network stays is 0.00–0.01).
- Recorded states, excluding the tails: 19–29% of honoured decisions change, mostly to phase 3. The
  phase-3 share roughly doubles (start-network states: 302 → 645 of 1,523), as expected now that W2
  pushes for phase 3 again.
- N/S surrogate (hide S2, rebuild it the same way from N2, compare with the full-information decisions, tails
  excluded): agreement for zero / mirror-only / mirror+time (A = 0.5) is 0.871 / 0.880 / 0.863 on
  start-network states, 0.949 / 0.942 / 0.928 on fixed states and 0.792 / 0.883 / 0.864 on actuated states.
  So the mirror helps or is neutral, and the time term costs about 2% of normal decisions. I accepted that
  cost because the term is what breaks the deadlock.
- `tools.trainability`: about unchanged (median max-prob 0.983, no dead units).

## What I expect to go wrong

- Normal-operation decisions are only approximately right. W2 ≠ E2 from moment to moment: a W2 queue
  with an empty E2 waits until E2 fills, the green runs long or W1u spills. A full E2 with an empty W2 may
  still get phase 3 (two-sided push from one side's cars).
- The time term overestimates W2 after long greens and costs ~2% of decisions. When E2 is full and phase 2 has
  run 60 s, the estimate reaches 1.5, beyond the training range, which pushes phase 3 hard.
- If phase 3 is ever held 48 s or more and the junction then empties, the network may hold phase 3 until
  traffic appears (a mild cost; I see no route to that in the data).
- The predictions are guesses from the reference table, with no simulation: most of the gain should come
  from removing the end-of-episode deadlock, mainly in the high scenario. The rest of the gap to the
  full-information network (34.7 / 44.9 / 54.4) will likely remain.

## Files

- `build.py`: builds `weights.npz` (config in `build_config.json`).
- `checks.py` / `checks_output.txt`: all checks above. `surrogate_check.py`: the first mirror-vs-zero
  probe.
- `predictions.json`: low 35.1, nominal 45.7, high 57.0 veh-h.
