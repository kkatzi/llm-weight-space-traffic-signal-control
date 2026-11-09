# Repair of the PPO signal network after the W2 detector failure

## What went wrong in the original network

With q_W2 = d_W2 = 0 the network never sees the west left-turn pocket fill up. That pocket holds only 3 cars,
so it overflows. Waiting left-turners then block the shared upstream lane W1u (and E1u fills the same way on
the east side). The network reads the halted cars on W1u/E1u as straight-ahead demand and keeps EW_through
(phase 2) green while the E/W through lanes are empty. N/S traffic piles up until every N/S upstream lane is
full. In 4 of the 6 recorded episodes it then holds phase 2 until the 5400 s cap (`explore1.py`). Those
gridlocked states are far outside the training data, and there the original network never picks phase 0 or 1
from any phase, even with the W2 input restored.

The original network also strands single vehicles. With one car on certain lanes (N1, S1, E0, W0, W1, E2) it
holds a wrong phase for ever, and with the fault every W2 car is invisible (`lone_vehicle.py`: 11 of 36 cases
stuck).

## What I built (304 of 7428 parameters changed; budget 371)

1. **Mirror proxy for the dead detector (112 params).** E2 and W2 have the same demand and are served by the
   same phase (EW_left), so the east pocket is the best stand-in for the west one. The network now sees q_E2
   where it expects q_W2, and d_E2 where it expects d_W2:
   `W1[i,5] += W1[i,11]`, `W1[i,17] += W1[i,23]` on the 56 layer-1 units that were not repurposed. On the
   recorded gridlock states this switches the choice from "hold phase 2" to "phase 3" (EW_left), which
   clears the blockage. I also tried adding the W1u queue to the proxy. It changed less than 1% of decisions,
   so I left it out.
2. **Max-green round robin (layer-1 units 31, 0, 59, 54 → layer-2 gates 54, 0, 44, 43).** Unit a_p fires
   when the current phase is p and `t_green + 0.0875*P >= 0.975`, where P is the sum of the 8 upstream
   queues. With P ≈ 0 that means a green of at least ~60 s; at gridlock (P ≈ 6) it means ~27–32 s. Its
   gate adds +30 to logit (p+1) mod 4. Any phase held that long therefore moves on in the order
   0→1→2→3→0. This ends every hold-for-ever lock, and it reaches an invisible W2 car within at most ~2 min.
3. **Congestion sequence lock (layer-1 units 27, 34, 5, 28 → layer-2 gates 35, 26, 59, 25).** Unit s_p
   fires when the current phase is p and P >= 1.25. In the recorded normal runs (fixed and actuated) P
   never went above 0.94; in the gridlock runs it is about 6. Its gate subtracts 30 from logits (p+2) and
   (p+3). Under heavy upstream queueing the network can then only hold or advance in the cycle. This stops
   the 2↔3 alternation that would starve N/S, so the controller degrades to a fixed cycle with greens of
   10–32 s.
4. **Compensation.** The repurposed units sit at −1 when inactive. I add their old mean activation to b2
   (64 params), and add the gates' old mean contributions plus their offsets to b3 (4 params). Each trigger
   row gets only its own weights (phase input, t_green, the 8 upstream queues, bias). The old weights stay
   but are swamped (gains of 40–1000).

I picked the units to repurpose greedily, choosing those whose mean-ablation changes the fewest recorded
decisions (`select_units.py`, recomputed inside `build.py`). No weights were fitted. Every value is set by
construction or comes from a quantile/mean of the recorded data. Run `build.py` to rebuild the network and
`verify.py` to check it.

## Checks run (forward passes only)

- `tools.edit_size`: 304 changed (4.1%). Checks run on the float32 file as saved.
- When no trigger is active, the delivered net agrees with the ideal mirror (original weights fed with
  x11 := x5, x23 := x17) on 98.5% / 98.5% / 98.0% of honoured decisions (start / fixed / actuated states).
  The disagreements are near-ties: the median logit gap there is 0.24, against 5.05 overall. On 128k random
  synthetic trigger-free states agreement is 96.5%.
- When the max-green trigger is active, the action is (p+1)%4 in 100% of 2986 recorded and 36,959
  synthetic states. When only the congestion trigger is active, the action is p or p+1 in 100% of 21,802
  synthetic states. The gate units are saturated except on 0.05% of synthetic states that sit right on a
  threshold.
- Lone-vehicle check (traffic frozen, only the signal logic stepped): before, 11 of 36 cases were stuck for
  ever; after, 0 of 36 are stuck. The longest wait is 135 s, which is the invisible W2 car (served by the
  round robin).
- Frozen gridlock snapshots: the network now cycles 3→0→1→2 with 12–32 s greens instead of holding phase 2.
- On fixed-time states, the share of honoured decisions that pick phase 3 rises from 12% to 29%. That fits
  W2 demand that is now counted again.

## What I expect to go wrong

- The mirror is noisy. The paired left pockets only correlate at about 0.3–0.5. When W2 holds more than E2,
  W2 can still overflow and briefly block W1u. This should resolve itself, because E2 keeps filling and then
  draws phase 3, but it costs delay, mostly in the high scenario. When E2 holds more than W2, phase 3 gets a
  little too much green.
- Once the junction is badly congested, the sequence lock and short max green turn it into a fixed-time-like
  controller, which is worse than the trained policy at its best. Each such episode probably costs a few
  veh-h, but it can no longer gridlock.
- At the end of an episode, invisible W2 cars and other lone cars can still wait up to ~2 min. That is small.
- Repurposing 16 units changes about 1.5% of near-tie decisions. The network is also more confident than
  before (trainability: median max-prob 0.995, 5 constant units per layer), so continued PPO training would
  start from a stiffer network.
