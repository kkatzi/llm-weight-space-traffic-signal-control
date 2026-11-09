# L2 -> L2W transfer (net2)

## What I built and why

The L2W network is the L2 network with a new first layer: `W1_new = W1_L2 @ M`, `b1_new = b1_L2`.
`W2, b2, W3, b3` are copied unchanged and are bit-identical (float32) to `source_weights_L2.npz`; `build.py` asserts this.

`M` (46 x 54) turns an L2W observation into an equivalent L2 observation, so the trained L2 policy sees
the wider junction as an L2 junction. I chose the mapping on a per-lane basis. Per lane, L2W at nominal demand is almost L2 at nominal:
- EW through: 245 veh/h per lane (L2: 262);
- NS: identical;
- the only real difference is the EW left pocket, at 315 veh/h instead of 225.

That is inside L2's training envelope, which used direction bias up to e^0.3 and level up to 1.25. Averaging per lane keeps the inputs on L2's scale, and the stop-line discharge per lane is the same in both junctions.

| L2 input (L2 net) | built from L2W inputs | reason |
|---|---|---|
| N0..N2, S0..S2 stop-line q/d | 0.93 x same lane | N/S lanes are 19.2 m in L2W, so 1 car reads 0.391 instead of 0.335. 0.857 would match 1-2 cars exactly but make a full lane read 0.857; 1.0 keeps "full = 1" but over-reads 1-2 cars by 17 %. 0.93 splits the error. |
| E0/W0 (right+straight) | E0/W0 | same lane type |
| E1/W1 (straight) | mean of E1, E2 (resp. W1, W2) | the two straight lanes; the mean keeps the per-lane scale |
| E2/W2 (left pocket) | E3/W3 | the left pocket |
| N/S upstream | same lanes | identical geometry |
| E0u/W0u upstream | mean of E0u, E1u | the extra straight-only lane is averaged into the through-only neighbour, as at the stop line |
| E1u/W1u upstream (feeds straight + pocket) | E2u/W2u, full weight | the same topological role (it feeds the inner straight lane and the pocket). Left-pocket spillback shows up here at full strength. |
| phase, ok, t_green | same | identical semantics |

`b1` needs no offset because every mapping is linear with zero intercept.

## Which units or weights do what

The hidden units are the L2 net's units, unchanged. Only their input wiring changed:
- every W1 column of an L2 E1/W1 lane is split 50/50 over L2W lanes E1/E2 (W1/W2);
- the E0u/W0u columns are split 50/50 over E0u/E1u (W0u/W1u);
- N/S stop-line columns are scaled by 0.93;
- all other columns are copied.

The inherited policy behaves like this. When the current green reaches 10 s it nearly always asks to switch: 96 % of honoured decisions on L2W states, 97 % on L2. In its own L2 episodes every switch happened at 12 s of green. So it runs fixed 12 s greens plus 3 s of yellow, and adaptively chooses which phase comes next.

## Checks run (forward passes only; `checks.py`, `probes.py`)

1. **Copied layers:** W2, b2, W3 and b3 are bit-identical to the source after `tools.save`.
2. **Exact embedding:** I rewrote the 4,432 L2 states in L2W layout (extra lanes copy their neighbour) with the N/S scale isolated. Max |logit difference| between the new net and the L2 net was 1.8e-15 for all three upstream variants. The E/W mapping is exact.
3. **Embedding with the N/S lane-length conversion** (car counts to 19.2 m normalisation): decision agreement with the L2 net was 96.6 % at s = 0.93 and 97.4 % at s = 0.857.
4. **Behaviour on the 8,909 recorded L2W states:**
   - phase picks [0.16, 0.17, 0.23, 0.45], compared with [0.15, 0.24, 0.24, 0.37] for the L2 net on L2. EW-left is picked more, matching the heavier left demand.
   - When it switches, it picks the busiest other phase (by mean per-lane q+d) in 76 % of cases and the 2nd busiest in 21 %. On L2 the figures are 82 % and 16 %.
5. **Synthetic probes:**
   - Only one phase's lanes are full: it picks that phase from every current phase (16/16).
   - E2u/W2u spillback rising from 0 to 1: the EW-left pick share rises from 32 % to 97 %.
   - E0u/E1u spillback: the EW-through share rises from 25 % to 46 %.
   - N1u spillback: the NS-left share rises from 11 % to 41 %.
   - N0u spillback: only a weak response (NS-through 24 % to 33 %). L2 never saw N/S upstream values above 0.26.
6. **Sensitivity of the mapping choices** (agreement with the delivered net on L2W states):
   - s = 0.857: 95.6 %
   - s = 1.0: 96.4 %
   - upstream variant B (E0u, mean(E1u, E2u)): 98.1 %
   - upstream variant C (E0u, E2u): 99.2 %

   No single choice matters much.

7. **`tools.trainability`:**

| | median max_prob | share > 0.99 | median logit gap | max abs logit | sat_l1 | sat_l2 | dead_l1/l2 |
|---|---|---|---|---|---|---|---|
| delivered, L2W recorded states | 0.9976 | 0.607 | 6.11 | 12.32 | 0.00002 | 0.149 | 0 / 0 |
| source, its L2 states | 0.9944 | 0.553 | 5.31 | 12.00 | 0.000 | 0.078 | 0 / 0 |

The delivered net computes exactly the source function of the mapped input, and its W1 column norms are equal to or smaller than the source's, because averaging shrinks them. The slightly higher sharpness and sat_l2 come from the input distribution: the recorded L2W states come from fixed and actuated control and are much more congested, with upstream values up to 1. They do not come from larger weights. The network is not sharper than the source.

## What I expect to go wrong

- **Fixed 12 s greens.** These cost 3 s of yellow per 15 s slot. In the high scenario's peak 15 minutes, the 1.4x heavier EW left pocket (≈ 416 veh/h, 3-car pocket) pushes utilisation close to capacity. Queues may build and the high TTS may approach or pass fixed-time (77.3 dev).
- **Mean of two straight lanes.** This hides one full lane next to an empty one. For example, E2 empties when left spillback blocks E2u. The spillback itself is still seen through E2u at full weight.
- **Extrapolation of upstream inputs.** L2 never saw upstream values above about 0.3. The responses to E/W spillback are sensible, but the N/S upstream response is weak.
- **Short N/S lanes.** No linear scale maps them perfectly, so about 3 % of decisions differ from a perfect car-count conversion.
- **Trainability.** The policy is near-deterministic (inherited), so PPO's early updates will explore little. The critic warm-up helps.

## Predictions

Predicted mean test TTS (veh-h): low 44.0, nominal 58.5, high 75.0. Dev references: fixed-time 48.3 / 63.0 / 77.3, actuated 44.6 / 60.9 / 86.9.

I assume about 48 s free-flow time per vehicle. On that basis, the L2 net's 44.8 on L2 means about 24 s delay per vehicle. I expect about 22 / 26 / 33 s delay per vehicle on L2W for low / nominal / high.

## Files

- `build.py`: builds `weights.npz` (`python work/build.py [s_ns] [up_mode]`; defaults 0.93, A).
- `checks.py`, `probes.py`: the checks above.
