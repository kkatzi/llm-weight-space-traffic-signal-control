# L2 -> L2W transfer (net1)

## What I built

The L2W network is the L2 network with a fixed linear "adapter" folded into layer 1:

    v  = M @ x_L2W            (46 x 54, a virtual L2 observation)
    W1 = W1_L2 @ M            (64 x 54)
    b1 = b1_L2                (no offsets needed: every mapping keeps 0 -> 0)
    W2, b2, W3, b3 = the L2 network's, copied bit for bit (checked after saving: float32 arrays identical)

So layer 1 receives exactly the pre-activations the L2 network would compute on `v`. Every entry of M is set by
the lane-role argument below. Nothing was fitted, searched or trained. Code: `work/build.py` (builds and saves),
`work/checks.py` (all checks; output in `work/checks_output.txt`).

## The mapping M (why each input goes where)

**Phase, ok, t_green** (L2W 48-53 -> L2 40-45): identity. They mean the same thing on both junctions.

**N and S approaches**: same lane structure as L2, so lane k -> lane k for queue and density. The upstream lanes
(256 m, same capacity) use identity. The stop-line lanes are 19.2 m on L2W and 22.4 m on L2, so the same vehicle
count reads higher on L2W. 0/1/2/3+ vehicles give 0/.391/.781/1 on L2W and 0/.335/.670/1 on L2. No single linear
factor matches all four levels: 0.857 is exact for 1-2 vehicles, and 1.0 is exact for a full lane. I used the
minimax factor **s = 0.937**, which makes the error at 2 vehicles equal and opposite to the error at 3+ vehicles. It
applies to the 12 N/S stop-line queue and density inputs.

**E and W approaches**: 4 stop-line lanes and 3 upstream lanes have to fit into L2's 3 and 2. I matched lanes by
role, i.e. by which upstream lane feeds them:

| L2 input (role) | L2W inputs | weight |
|---|---|---|
| E0: right+straight, own upstream lane | E0 (right+straight), E1 (straight); both have their own upstream lane | 0.5 + 0.5 (mean) |
| E1: straight, fed by the shared straight+left upstream lane | E2: straight, fed by the shared upstream lane | 1.0 |
| E2: left pocket | E3: left pocket | 1.0 |
| E0u: upstream right+straight | E0u, E1u (no left-turners) | 0.5 + 0.5 |
| E1u: upstream straight+left (left spill-back shows here) | E2u: straight+left | 1.0 |

W is the same. Queue and density use the same mapping.

Why use per-lane means and not sums? Per lane, L2W's EW through load (about 245 veh/h/lane) is close to L2's (about
262 veh/h/lane). The mean keeps "queue per lane", which is roughly the green time a movement needs, and keeps
the inputs in the source network's range. Summing the three through lanes would read EW as 1.5 times busier,
push inputs outside the trained range (layer-2 saturation rises from 0.097 to 0.135) and change 17% of decisions.
The role match (not a lane-position interpolation) keeps the L2 network's "shared-lane / left spill-back" inputs
(E1, E1u) tied to the physically equivalent L2W lanes (E2, E2u). This matters because EW-left is the movement that
is relatively heavier on L2W (315 vs 225 veh/h in one pocket), and E2u reaches the L2 network at full weight.
Per-lane loads on L2W fall inside the L2 training range (level 0.7-1.25 x profile x EW bias up to e^0.3), so the
source policy is not extrapolating.

## What the units do

The 64 layer-1 units are the L2 network's own units. Each now reads an L2W lane group through M, so their meaning
is unchanged. The input-to-logit Jacobian of the source network, averaged over states, shows what the policy uses.
Through-lane inputs push their own through phase. Left-pocket inputs push their left phase. Upstream EW densities
push phases 2 and 3 strongly (about 3-4 logits per unit). The source policy is almost "switch at the first decision
after the 10 s minimum green" (it stays only 3% of the time at 12 s). Its skill is in choosing the next phase and in
extending a green while that green's lanes are still full.

## Checks run (forward passes only)

1. **Frozen layers**: W2, b2, W3 and b3 in `weights.npz` are byte-identical to `source_weights_L2.npz`.
2. **Plumbing**: L2 states lifted into L2W layout and pushed through the new net reproduce the L2 net's logits to
   1.8e-15 when the N/S factor is 1. With the N/S rescale and realistic count-based N/S lifting, agreement with
   the L2 network's own decisions is 95.6% (95.1% on honoured decisions). The other factors give 96.6% at 0.857 and
   93.2% at 1.0, so the choice of factor barely matters.
3. **Feature statistics** of mapped L2W recorded states vs L2 states. EW through per-lane queues match (q_E0 0.293
   vs 0.297). EW-left reads heavier (q_E2 0.42 vs 0.26, d_E1u 0.092 vs 0.057), as expected from the heavier left
   demand and from the recorded L2W controllers being worse than the L2 network.
4. **Variants**: the role mapping agrees with a lane-position interpolation on 95% of states and with "drop the
   extra lane" on 96%. Sums disagree on 17%. So the result is not sensitive to the exact split, only to mean vs sum.
5. **Synthetic probes**: any one loaded movement gets its own phase with p of about 1, from every current phase. An
   empty junction holds the current phase. All lanes full goes to EW through. A held EW-through green gives way to NS
   once NS upstream queues reach about 0.2 (about 7 vehicles per lane), even when everything is full, so no
   starvation was seen.
6. **Long greens on recorded L2W states**: stay rate 3% at 10-15 s, 9% at 25-40 s, 23% at 40-60 s (these states
   came from the fixed-time controller's long greens).

### Trainability (`tools.trainability`)

| | median max_prob | share > 0.99 | median logit gap | max abs logit | sat_l1 | sat_l2 | dead l1/l2 |
|---|---|---|---|---|---|---|---|
| L2W net on L2W states | 0.992 | 0.522 | 4.97 | 12.24 | 0.0003 | 0.097 | 0 / 0 |
| source net on L2 states | 0.982 | 0.439 | 4.17 | 12.00 | 0.0001 | 0.058 | 0 / 0 |

W1 is no larger than the source's (max |W1| 0.877 for both; Frobenius norm 11.40 vs 12.12). Layer-1
pre-activation std is 0.65 vs 0.55, and layer 1 is essentially unsaturated. The slightly higher confidence comes
from the recorded L2W states being more extreme (worse controllers, heavier traffic), not from sharper weights. The
network is as near-deterministic as the source, as TASK.md warns. PPO starts from a confident policy. Its entropy
bonus and the 50k-step critic warm-up should keep the first updates from destroying it. The duplicated columns
(E0/E1 -> L2 E0, E0u/E1u -> L2 E0u) receive different inputs, so PPO can separate them.

## Predictions and what I expect to go wrong

Predicted mean test TTS: low 45.5, nominal 59.5, high 76.5 veh-h. That is about fixed-time level at high demand and
a little better at low and nominal. This is a rough delay-per-vehicle estimate, scaled from the source's 44.5 on L2
and from the dev references; no simulation was run.

Risks:
- **Short greens under heavy load**: the policy almost always switches after about 12 s. Lost time (3 s yellow per
  switch) is costly in the high scenario, where EW-left and EW-through are near capacity at the 1.15 x 1.15 peak.
  It does extend greens while the current lanes stay full, but that may not be enough.
- **EW-left underservice**: the pocket holds 3 cars and saturates at 1.0 quickly. The extra left demand shows up
  mainly through E2u/W2u. If the policy does not repeat phase 3 often enough, queues spill into E2u and block the
  straight vehicles in that lane too.
- **Per-lane reading undervalues EW through**: EW through discharges 1.5 times as many vehicles per second of
  green as on L2. A delay-optimal (c-mu) policy would weight it more than the per-lane mean does. This is the first
  thing I expect PPO to correct.
- **N/S rescale**: full N/S lanes read 0.937 instead of 1, and partial ones read 6-9% high. These are small errors.
- The recorded L2W states come from fixed and actuated control, not from this network, so the decision statistics
  above are only a proxy for its closed-loop behaviour.
