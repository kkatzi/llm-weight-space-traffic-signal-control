# Repair of the network after the W2 detector failed

## What I built and why

The W2 detector (west left-turn pocket) now always reads 0, so x[11] = q_W2 and x[23] = d_W2 are 0. The network
learned to rely on W2 heavily. If you sweep W2 from 0 to 1 on recorded states, its pick of phase 3 (EW_left) rises
from 21 % to 71 %. With the detector stuck at 0 it under-serves the west left turners. The pocket holds only about
3 cars, so it fills and spills back into upstream lane W1u, which also carries straight traffic. That fits the
largest loss being in the high-demand scenario.

The fix is to substitute the partner pocket. E2 (east left-turn pocket) has the same demand as W2, since the E and
W demands and the directional bias are always equal, and it is served by the same phase 3. Given the signal
history, q_E2 and q_W2 are two copies of the same random variable. The patch makes the network read E2 wherever it
used to read W2:

    W1[:, 5]  += W1[:, 11]    # q_E2 also stands in for q_W2
    W1[:, 17] += W1[:, 23]    # d_E2 also stands in for d_W2

Because x[11] = x[23] = 0, the patched network computes exactly what the original network would compute if W2 had
the same readings as E2. Everything downstream (W2, b2, W3, b3) is untouched, so the policy is the trained one,
with a mirrored estimate of the missing input.

**Parameters changed: 128 of 7428** (two columns of W1). Nothing else changed.

## Which units/weights do what

- `W1[:, 5]` now carries the original q_E2 weights plus the original q_W2 weights.
- `W1[:, 17]` now carries the original d_E2 weights plus the original d_W2 weights.
- `W1[:, 11]` and `W1[:, 23]` are unchanged. They only ever multiply 0 now.

The first layer is distributed: the E2 and W2 columns have low cosine similarity, so no single unit is "the W2
unit". That is why I patched at the input level instead of editing units.

## How I chose it (no fitting, forward passes only)

The recorded states already have W2 = 0, so W2 cannot be compared with anything directly. I used the N/S pair as an
analogue (`work/analog_check.py`): I faked a second failure on S2 (and, separately, on N2), then compared a few
hand-built imputations by how closely the network's decisions matched its decisions with the true S2/N2.

| S2 imputation (analogue) | agreement on honoured decisions | NS-left pick rate (true 0.233) |
|---|---|---|
| fault (0) | 0.851 | 0.117 |
| constant mean | 0.802 | 0.244 |
| mirror S2 := N2 (chosen) | **0.881** | 0.250 |
| shrunk mirror rho*N2 + (1-rho)*mean | 0.870 | 0.222 |
| mirror + upstream-density difference | 0.838 to 0.872 | about 0.249 |
| mirror + k * q_S1u (linear, unclipped) | 0.874 to 0.881 | 0.254 to 0.265 |

The reverse direction (N2 failed, imputed from S2) ranks them the same way: fault 0.877, mirror 0.890, half
mirror 0.884. The mirror was best both times, and it restores the left-phase pick rate that the fault halves.

The upstream lane does carry real information: when the lane feeding a pocket has halting vehicles, the pocket is
almost always full. But the effect is a step, and a linear term without clipping did not help on the analogue, so I
left it out.

## Checks run (`work/checks.py`)

1. `edit_size` = 128 (W1 only, columns 5 and 17), after a save/reload round-trip through float32.
2. Exact equivalence: patched(x) equals original(x with W2 := E2) on all 14,337 recorded states (max logit
   difference 7e-8; identical decisions).
3. Honoured-decision shares on the recorded states: phase 3 picks go from 0.209 to 0.374, and 17 % of decisions
   change. These states were produced by the under-serving faulty network, so E2 is fuller in them than it will be
   in closed loop. The real phase-3 share should come out lower.
4. When E2 is empty (2,428 honoured states), decisions are identical to the faulty network.
5. On a synthetic median phase-2 state, the patched network switches to EW_left once E2 holds 1 car (0.33). The
   faulty network needs 2 cars.
6. `trainability` (for information only): no dead or saturated first-layer units. The policy is somewhat more
   decisive than before.

## What I expect to go wrong

- When E2 and W2 differ (for example E2 empty while W2 is full), the network still misjudges the west left turners.
  This matters most at low and nominal demand, where the pockets are not always full. In the N/S analogue the
  mirror recovered only part of the decision agreement (0.851 to 0.881). Restoring the left-phase frequency is the
  main expected gain.
- W2 spillback that E2 does not share (halting on W1u) is still invisible to the network.
- The mirror can over-call phase 3 when E2 is full and W2 is not, which costs a little green time for the other
  phases.
- I expect results between the faulty network (35.9 / 46.9 / 60.9 on dev) and the full-detector network
  (34.5 / 44.9 / 55.0), with most of the gain in the high scenario. Predictions: low 35.3, nominal 46.0,
  high 57.8. Test-seed noise of about ±1 veh-h per scenario is likely.
