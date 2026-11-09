# What the trained controller (start_weights.npz) does at junction J0

## In one paragraph

The network is a **minimum-green, queue-based phase picker**. As soon as the current green has lasted 10 s
(the first decision at which a switch is honoured, i.e. a green age of 12 s), it almost always ends the green and
gives the next green to the phase with the **largest weighted queue**. Vehicles in the **left-turn pockets count
more** than vehicles in the through lanes. Two through lanes (N1 and W0) count only about half, and vehicles
approaching on the E/W upstream lanes add a little to the E/W phases. A small preference that depends on the
phase now green breaks near-ties, and it roughly follows the ring EW-through → EW-left → NS-through → NS-left. The
network extends the current green only when the rest of the junction is (nearly) empty. On its own runs, 97 % of
greens end at the first opportunity (green age 12 s), and only 4.7 % of honoured decisions extend.

## When it switches

- **Almost always, at the first honoured decision.** The green therefore lasts 12 s, followed by 3 s of yellow.
- **Extensions** (the network keeps the current phase) happen only when there is very little demand elsewhere:
  - after **NS-through**: never;
  - after **NS-left**: only when the whole junction is empty;
  - after **EW-through**: when the best other phase scores less than about 1–1.4 "vehicles" (roughly one waiting
    car elsewhere). This is the most common extension. The network leans slightly towards keeping the main road
    green, more so when E/W vehicles are still on the stop-line lanes or approaching upstream;
  - after **EW-left**: when the others score less than about 0.75–1.2 vehicles.
- Time in green (`t_green`) has almost no influence. An extended green is not more likely to be ended later.

## To which phase

Score of each phase, in "vehicles" (stop-line lane: a halted vehicle = 1, a moving one = 0.5; each lane holds at
most 3 vehicles, so a lane contributes at most 3):

| phase | own lanes that count (weight per vehicle) | extra credit |
|---|---|---|
| 0 NS-through | N0 ×1, S0 ×1, S1 ×0.7–1, N1 ×0.5 | W0 ×0.5; ~0.2 per vehicle waiting on the left pockets |
| 1 NS-left | N2 ×1.4, S2 ×1.8 (≈1.6 on average) | none |
| 2 EW-through | E0 ×0.65, E1 ×1.05, W1 ×0.8 (≈1 on average), W0 ×0.5 | N1 ×0.5; ~0.3 per NS-through and EW-left vehicle; +0.1 per E/W upstream vehicle |
| 3 EW-left | E2 ×1.3, W2 ×1.3 | ~0.2 per vehicle elsewhere; +0.1 per E/W upstream vehicle |

Then a preference for the next phase, which depends on the phase that is green now (in vehicles, from
`rules_full.py`; `rules_small.py` uses slightly larger values for its simpler scores):

| green now | NS-through | NS-left | EW-through | EW-left |
|---|---|---|---|---|
| NS-through | (never extends) | +0.5 | 0 | −0.45 |
| NS-left | +0.5 | +0.55 (extend) | +0.1 | +0.1 |
| EW-through | 0 | −0.4 | +1.1 (extend) | −0.1 |
| EW-left | 0 | −0.7 | −0.55 | +0.75 (extend) |

These preferences are small (under one vehicle). The choice is driven by the queues. A left-turn pocket vehicle
weighs about 1.3–1.6 through vehicles. Ten vehicles approaching on the E/W upstream lanes weigh about as much as
one waiting vehicle. On the recorded states the network's typical sequences are EW-through → EW-left or
NS-through, EW-left → NS-through or EW-through, NS-through → EW-through, NS-left or EW-left, and NS-left →
EW-through or EW-left. Phases with little demand are skipped.

## What it ignores (or nearly ignores)

- **Lane N1 (north straight) and lane W0 (west right + straight).** They count about half, and the same for both
  through phases. So they do not help choose between NS-through and EW-through. A queue there is served
  mostly because the other lanes of the same movement trigger the phase.
- **Upstream queues** (`q_*u`): no effect beyond what upstream density already says. **Upstream N/S density**: very
  small effect (+0.03 per vehicle for NS-through).
- **Time in green**: almost no effect.
- **Before the minimum green** (`ok = 0`) its outputs are arbitrary, but they are ignored by the signal anyway.
- It cannot see a left-turn queue longer than the 3-vehicle pocket. Left-turners waiting further back appear only
  as upstream density, and the network credits that to the E/W phases in general, not to the left phase.

## Where it may perform poorly

- **Short greens and high lost time.** Every green is 12 s plus 3 s of yellow, so about 20 % of the time is
  yellow. Under heavy demand a policy with longer greens on the main road would lose less capacity. The network
  still beats the tuned actuated and fixed-time controllers at the tested demand levels.
- **Saturated left turns.** The pocket observation saturates at 3 vehicles per pocket, so a left phase scores at
  most about 8–10, while a through phase can reach 12 or more (more with upstream traffic). In very heavy traffic, NS-left in particular can be postponed for
  several cycles. On synthetic "everything full" states the network picks NS-left very rarely.
- **Lanes N1 and W0** are under-weighted (see above). Unusually heavy straight traffic from the north, or heavy
  right/straight traffic from the west, may wait longer than necessary.
- **Asymmetries without an obvious traffic reason** (S2 weighed more than N2, E1 more than E0, W1 used but W0
  not). These are learned from the training demand. With a different turning mix or a different directional
  bias they may not be appropriate.
- **Near-ties.** About 12 % of the network's honoured decisions have a logit gap under 1. In those states the
  network is nearly indifferent, and the simple rules disagree with it about half the time. Most of the
  remaining rule–network disagreements are of this kind, so they should cost little delay.

## The two rule files

- `rules_small.py`: movement sums (NS-through = N0+S0+S1, NS-left = N2+S2, EW-through = E0+E1+W1, EW-left =
  E2+W2; N1 and W0 half for both through phases), left pockets ×1.6 (NS) / ×1.3 (EW), E/W upstream ×0.1, and the
  4×4 preference table. 3 comparisons (a 4-way maximum).
- `rules_full.py`: the same structure with lane-by-lane weights, the spill-over credits and N/S upstream.
  3 comparisons.

Agreement with the network on the 5,095 recorded states where the minimum green has elapsed: small 90.3 %,
full 91.0 % (on the network's own runs: 89.4 % and 89.1 %). Both extend the green on 2.9 % of these states,
as the network does.
