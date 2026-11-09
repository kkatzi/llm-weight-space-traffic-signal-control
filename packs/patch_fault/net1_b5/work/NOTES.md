# Repairing the network for the dead W2 detector

Final file: `work/weights.npz` (identical to `work/v2.npz`, built by `work/build_v1.py` then `work/build_v2.py`).
**292 of 7428 parameters changed** (W1 216, b1 2, W2 2, b2 64, W3 4, b3 4). Limit: 371.

## What was wrong

The W2 pocket detector (x[11] q_W2, x[23] d_W2) always reads 0. In the recorded start-network episodes, high demand
got stuck on phase 2 (EW_through) for over an hour. W2 filled up unseen and the queue spilled back into upstream
lane W1u, which also feeds the W through lane W1. The left-turners waiting at the head of W1u blocked every
through vehicle behind them. W1 then emptied and stopped moving, and the network kept phase 2 green because it never
saw the W2 queue. Setting q_W2 = d_W2 = 1 in those states makes the original network pick phase 3 (EW_left)
straight away. So the trained policy is fine; it just lacks the W2 reading.

## What I built

1. **Use E2 in place of W2 (128 params, `build_v1.py`).** The E pocket E2 has the same demand as W2 (30 % of 750 veh/h),
   is served by the same phase (3), and its detector works. I added `W1[:,11]` to `W1[:,5]` (q_E2) and `W1[:,23]` to
   `W1[:,17]` (d_E2). Every first-layer unit now behaves as if q_W2 = q_E2 and d_W2 = d_E2. On its own this
   gives a dev TTS of 46.0 (start: 214.3). One problem remained: 2 and 5 vehicles were stranded at the drain cap in two
   episodes, i.e. W2 left-turners left over when E2 was empty. On an empty junction the network just holds the current
   phase forever.

2. **Two trigger units and a relay that force phase 3 (164 more params, `build_v2.py`).**
   - `h1[0]`, **long-green trigger**: pre = 100·(t_green − 0.9) − 200·phase3. It is +1 when a phase other than 3 has
     been green for at least 57 s on the decision grid; otherwise −1. In the recorded normal operation, phases 0–2
     never last that long, so in practice it fires only when the junction looks empty. That serves stranded W2
     vehicles.
   - `h1[31]`, **W2-spillback trigger**: pre = 8·Q − 280·Dw + 272·phase2 − 276. Q is the number of halting cars on W1u
     (x31·34.13) and Dw the number of cars on W1 (x22·2.987). It is +1 when phase 2 is green, at least one car is
     halting on W1u and W1 is empty. On the E side, where every detector works, the same pattern ("phase 2, E1u
     halting, E1 empty") has E2 full in 99 % of the 246 recorded cases. The margins hold for any Q from 0 to 34.
   - `h2[25]`, **relay**: W2[25,0] = W2[25,31] = b2[25] = 20. Its other inputs sum to at most 11.3 in absolute value, so
     the relay is exactly −1 normally and +1 when either trigger fires.
   - `W3[:,25] = (0,0,0,5)`, so a firing trigger adds +10 to the phase-3 logit.
   - **Compensation**: the trigger units sit at −1 in normal operation instead of their old values. b2 absorbs the
     mean shift (`b2 += W2[:,j]·(mean_old(h1_j)+1)`), and b3 absorbs the old relay unit's mean output. Normal decisions
     are therefore those of v1 with these three units mean-ablated. I picked them as the least important units: each
     changes 0.2–0.3 % of decisions when mean-ablated.

## Checks I ran (all forward passes on recorded or synthetic states; no fitting)

- `analyse.py`, `stats.py`: start network locked on phase 2. E-side statistics for the spillback pattern. Phases other
  than 3 never last ≥ 0.9·60 s in non-gridlock recorded operation.
- `units.py`, `pick_units.py`: ablation importance of the hidden units, to choose which ones to repurpose.
- `check_v2.py`:
  - When no trigger fires, honoured decisions differ from v1 in 0.4–0.8 % of recorded states.
  - Trigger units and relay are fully saturated (|h| = 1).
  - On an idle junction, phases 0/1/2 switch to 3 at 57 s and phase 3 is held.
  - With phase 2 green, 1–5 cars halting on W1u and W1 empty, the network picks 3 (v1 picked 2).
- `debug_mirror.py`, `target_mirror.py`: E-side spillback states rotated onto the W side (so E2 now reads 0).
  - v2 picks phase 3 in 136/246 of them; v1 picked it in none.
  - The original network, shown W2 full, picks 3 in 94/246 with E2 = 0 and 149/246 with E2 = 1 car. v2 lies between
    the two, so +10 is a moderate push.
- Simulator (dev, mean TTS veh-h; 3 of 5 calls used):

| call | file | low | nominal | high | all | stranded at cap |
|---|---|---|---|---|---|---|
| 1 | v1 (E2 substitute only) | 34.8 | 45.6 | 57.5 | 46.0 | 2 and 5 vehicles |
| 2 | **v2 (final)** | 34.6 | 45.1 | 55.5 | **45.1** | none |
| 3 | v2 with push S=8 (+16, always phase 3) | 34.6 | 45.2 | 55.5 | 45.1 | none |

  Reference, network with every detector working: 34.5 / 44.6 / 54.4 / 44.5. The S=8 run was no better, so I kept S=5,
  whose choices stay closer to the trained policy.

## What I expect to go wrong

- W2 is still only guessed from E2. When W2 holds cars and E2 is empty, they wait until E2 fills, the spillback
  trigger fires, or the junction goes idle for 57 s. When E2 has cars and W2 is empty, phase 3 is sometimes served
  for nothing. Most of the remaining gap to the full-detector network (about 1 veh-h at high demand) is probably this.
- The spillback trigger needs W1 to be completely empty, so it reacts a few seconds late. A car halting on W1u for
  another reason during phase 2 would trigger an unneeded phase 3 (about 13 s lost).
- The long-green trigger acts as a 57 s maximum green for phases 0–2. If a test episode has a very long, heavy phase-2
  platoon, cutting it off could cost a little.
- Under strong E/W asymmetry, E2 is a worse stand-in for W2. The test scenarios have none.
- Units h1[0], h1[31] and h2[25] lost their old roles (mean-ablated), which shifts roughly 0.5 % of ordinary decisions.
- Test seeds differ, and dev TTS varies about ±1.5 veh-h between episodes, so I padded my predictions slightly
  above the dev values.
