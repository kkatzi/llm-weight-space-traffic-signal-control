# L2 → L2W transfer: notes

## What I built

The L2W network is the L2 network running on an L2-shaped version of the L2W observation. A fixed affine map turns
the 54 L2W inputs into the 46 inputs the L2 network was trained on, `x_L2 = M x_L2W + c`. That map is folded into
layer 1:

    W1 = W1_L2 @ M        (64 x 54)
    b1 = b1_L2 + W1_L2 @ c
    W2, b2, W3, b3 = source values, copied unchanged (bit-identical float32, checked after saving)

So `net(x) == source(M x + c)` exactly, up to float rounding of 1.5e-7 in the logits. No weights were fitted. M and c
were set by hand from the lane layout, and two scalars (NS_SCALE, KAPPA) were chosen by reasoning plus forward-pass
probes. Code: `build_weights.py` (construction), `common.py` (lane map), `checks.py` (output in `checks_output.txt`).
`explore*.py` are the design-time probes.

### The lane map M (why)

The L2 network reads **per-lane** occupancies. In L2W the per-lane demand on E/W through lanes is about the same as in
L2: 1050 veh/h × 70 % over 3 lanes is 245 veh/h per lane, against 750 × 70 % over 2 lanes, 262 veh/h per lane in L2.
Lanes in one phase discharge in parallel, so how often a phase needs green depends on per-lane load, not the
approach total. So the map keeps per-lane meaning and keeps every input inside [0, 1]:

| L2 input (q and d) | built from L2W | reason |
|---|---|---|
| N0..N2, S0..S2 | same lane × 0.93 | the L2W N/S stop-line lanes are 19.2 m long, so 1 car = 0.391 instead of 0.335; 0.857 would match 1–2 cars exactly, 1.0 would match a full lane (3 cars, clipped to 1). 0.93 sits between them, because a full side lane must still look full. |
| E0 / W0 (right+straight) | E0 / W0 | same lane type |
| E1 / W1 (straight) | ½(E1 + E2) | the two L2W straight lanes; their mean is a per-lane occupancy |
| E2 / W2 (left pocket) | E3 / W3 | same lane type. L2W left demand is 40 % higher per pocket, and the net sees that directly. |
| E0u / W0u | E0u / W0u | feeds the right+straight lane in both junctions |
| E1u / W1u (straight+left feeder) | ½(E1u + E2u) | mean level matches L2's (about 0.05 under fixed time); E2u alone is about 2× higher |
| N/S upstream, phase, ok, t_green | unchanged | same meaning |

I tried other variants (E2 only, E1 only, E2u only, sums). Most changed under 5 % of honoured decisions on the
recorded states. Using sums pushes inputs up to 2, outside the trained range, so I didn't use them.

### The unstick term (KAPPA = 0.75)

I found a real pathology in the source network. When EW-through (phase 2) is green and no other traffic is
around, it keeps that green forever while 1–2 cars wait on N/S through, a N/S pocket or a left pocket. It treats
phase 2 as the default for a near-empty junction. In the recorded L2 data, one of 6 episodes ended this way: 3 cars
sat for the full 30-minute cap. On frozen lone-car snapshots in L2W, the plain map leaves 44 of 289 cases never
served.

Fix, inside W1/b1 only: the q and d inputs of every stop-line lane that phase 2 does **not** serve (N0 N1 N2 S0 S1
S2, E3 W3) get an extra `0.75 × (t_green/60 − 0.2)`. This is exactly 0 at t_green = 12 s, where the source makes
almost all of its honoured decisions (it switches at 12 s in 99 % of cases). It only grows when a green is held
longer: +0.25 at 32 s, +0.6 at 60 s, as if a car had appeared on those lanes. In weight terms, the t_green column of
W1 gets 0.75 × the sum of those 16 source columns, and b1 gets −0.15 × the same sum.

## Which units/weights do what

- Hidden layers are the source's. On recorded states no layer-1 or layer-2 unit is dead, and layer-1 saturation is
  0.03 %.
- W1 columns for N/S lanes are 0.93 × source columns. The E1/E2 columns (and E1u/E2u) are each ½ × the source's E1
  (E1u) column. The E3 column is the source's E2 (left pocket) column. The t_green column carries the unstick
  direction, which is why max|W1| is 2.18 against 0.79 in the source. Only the t_green column is that large, and
  t_green ≤ 1.

## Checks run (see `checks_output.txt`)

1. Copied layers are bit-identical to the source. Shapes are right; saved with `tools.save`.
2. `net(x) == source(Mx + c)` on all 8909 recorded states, max logit diff 1.5e-7, all decisions equal.
3. The unstick term is inert at t_green = 12 s: 100 % of those honoured decisions equal the plain map. On recorded
   honoured states with t_green > 12 s (from the fixed and actuated controllers' long greens), 92 % are equal.
4. **Trainability** (`tools.trainability`, recorded L2W states):
   median_max_prob 0.988, share_max_prob>0.99 0.479, median_logit_gap 4.60, max_abs_logit 9.55, sat_l1 0.0003,
   sat_l2 0.125, dead_l1 0, dead_l2 0.
   For comparison, the plain map without the unstick term gives 0.987 / 0.469 / 4.53 / 9.55 / 0.0002 / 0.128 / 0 / 0.
   The source on its own L2 states gives 0.963 / 0.363 / 3.46 / 9.40 / 0.0 / 0.070 / 0 / 0.
   The slightly higher confidence comes from the input states, not the weights: the recorded L2W states are more
   congested, from worse controllers, and the plain map gives the same numbers. Layer 1 is not saturated.
5. Mapped inputs stay in [0, 1] except at long greens, where the unstick term adds up to 0.6 (max 1.53).
6. Single-movement probes: demand on only NS through / NS left / EW through / EW left (or one approach of it) gives
   the right phase from every current phase, at levels 0.67 and 1.0. An empty junction defaults to phase 2.
7. Queue-served proxy (waiting vehicles on the chosen phase ÷ best phase, honoured states): 0.865 on fixed-time states
   (fixed-time itself: 0.494), 0.783 on actuated states (actuated itself: 0.756). The source on its own L2 states
   scores 0.853. This is a myopic proxy only.
8. Frozen-snapshot switching check (`unstick_check.py`: vehicles frozen, only the 5 s / 10 s min-green / 3 s yellow
   rule iterated). Never-served lone-car cases fall from 44/289 to 0/289, max wait 30 s. On 600 random sparse
   snapshots with upstream traffic: 73 → 0 stuck.

## What I expect to go wrong

- **High demand.** The source always ends a green at the 12 s minimum. Every 15 s slot loses about 5 s to yellow and
  start-up. L2W's EW-left pocket (315 veh/h, ×1.32 at the high peak) and the 3-car pockets are tight. Expect queues
  and left-turn spillback into E2u/W2u, which also blocks straight traffic, during the high scenario's peak 15 minutes.
  The net responds only weakly to upstream spillback (probes: +2–6 % picks), so this is where it is most likely to
  lose to fixed time.
- **N/S lanes are shorter (19.2 m).** The 0.93 scale is a compromise, so side-street urgency is over- or
  under-stated by about 7 %.
- **Long-green behaviour changed.** If PPO starts holding greens past 12 s, the unstick term biases those decisions
  toward N/S and left phases. That's harmless in light traffic. Under congestion it slightly lowers the proxy
  (0.806 → 0.783 on long-green recorded states).
- **The net is near-deterministic,** like the source (median max prob ≈ 0.99), so early PPO updates may be jumpy.
  The task accepts this, because the trained layers are copied unchanged.
- The source has other untrained corners: its upstream-queue inputs were almost always 0 in L2, and their learned
  responses are noisy (for example, q_E1u pushes toward NS-left).

## Predicted test TTS (veh-h)

low 43.5, nominal 59, high 82 (in `predictions.json`). Reasoning: L2W low has the same total volume as L2 nominal,
with more E/W lanes, so it should be about the source's L2 level (44.7 dev). Nominal should be near actuated and fixed
time. High suffers from 12 s greens, so I put it between fixed (77) and actuated (87).
