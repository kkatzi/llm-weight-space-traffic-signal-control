# Repairing the network for the failed W2 detector

## What I built and why

I didn't retrain anything or change any decision logic. The network is the original PPO network. I added a small
**imputation front-end** that rebuilds the two dead inputs (q_W2 = x[11], d_W2 = x[23]) from inputs that still work,
then folded it into the first layer. Swapping an input x_i for `sum_j a_j x_j + b` is equivalent to
`W1[:, j] += a_j * W1_orig[:, i]` and `b1 += b * W1_orig[:, i]`. The rest of the network sees the kind of input it was
trained on, so its learned behaviour carries over.

Imputed inputs (deployed values: k = 6, c3 = 1.0, cN = 0.5):

    q_W2_hat = q_E2 + k*q_W1u + c3*(t_green - 0.2)
    d_W2_hat = d_E2 + k*q_W1u + c3*(t_green - 0.2)
    q_N0, q_N1, q_S0, q_S1, d_N0, d_N1, d_S0, d_S1  each  += cN*(t_green - 0.2)

The three ingredients:

1. **Mirror lane (q_E2, d_E2).** E2 is the opposite left pocket. It has the same demand (the directional bias
   scales E and W together) and the same phase (3, EW_left) serves it. It's the natural stand-in for W2 and needs no
   fitting. On its own this took dev TTS from 48.0 to 45.6.
2. **Upstream spill-back evidence (k * q_W1u).** W1u feeds W1 and W2. Under the network-like controllers in
   `states.npz`, I checked the mirror sides. Whenever the upstream lane 1 holds any halted car, the left pocket is
   almost always full: E[q_E2 | q_E1u > 0] ≈ 0.8 and P(q_E2 > 0.9) ≈ 0.7–0.8, against E[q_E2] ≈ 0.3 when
   q_E1u = 0. The N side gives the same picture. With the detector dead, the network was seeing an impossible
   combination: halted W1u but an empty W2 pocket. I chose k by hand. A typical non-zero q_W1u is 0.03–0.12, so
   k ≈ 3–6 lifts the imputation to about 0.8 in those states. I then checked it in the simulator: k = 3 gave 45.3,
   k = 6 gave 45.0 and k = 10 gave 45.1, so k = 6 sits on the plateau.
3. **Time phantom (t_green term), the drain-out fix.** On an empty junction the network holds phase 2 forever. A
   car in the dead W2 pocket was never served. Separately, the *original* network (even with all detectors)
   strands a lone halted car on N0/N1/S0/S1, because it needs about 3 cars there before leaving phases 1, 2 or 3.
   Both kinds of stranded car stay until the 30-minute drain cap, costing about 0.47 veh-h each. In call 1, 5
   vehicles were left in 4 of 9 dev episodes. In the recorded start-network episode 5, about 9 cars (W2 full plus
   6 halted in W1u) sat from t=3640 s to the cap.
   - The phantom is anchored at t_green = 0.2, which is 12 s, the first honoured decision. The start network nearly
     always leaves phases 0, 1 and 3 there, so most decisions see a phantom of exactly 0. It only grows during long
     holds.
   - c3 and cN were set offline, with forward passes on synthetic near-empty states (`work/explore3b.py`,
     `work/drain_test.py`), so that:
     - an empty junction goes 2 → 3 after about 40 s (serving an unseen W2 car);
     - a W1u spill-back deadlock goes to 3 after about 35 s;
     - a lone N/S through car goes to 0 after 30–45 s.

   Neither phantom alone does this: c3 alone never serves N/S cars, and cN too large makes the empty junction cycle
   2↔0 and strand W2 again.

## Which weights changed (320 of 7428; only layer 1)

| tensor / column | role |
|---|---|
| `W1[:, 5]` (q_E2) | += `W1o[:, 11]`: q_E2 also feeds every unit through q_W2's original weights |
| `W1[:, 17]` (d_E2) | += `W1o[:, 23]`: same for d_W2 |
| `W1[:, 31]` (q_W1u) | += 6·(`W1o[:, 11]` + `W1o[:, 23]`): spill-back evidence into both W2 inputs |
| `W1[:, 45]` (t_green) | += 1.0·(`W1o[:,11]`+`W1o[:,23]`) + 0.5·Σ `W1o[:, NS]`: the time phantom |
| `b1` | −0.2 × (the t_green increment): anchors the phantom at 12 s |

`W1[:, 11]` and `W1[:, 23]` are left untouched; they multiply inputs that are always 0. W2, b2, W3 and b3 are
unchanged. No hidden unit was repurposed, because none is dead (checked by ablation in `work/units.py`).

## Checks I ran

- `tools.edit_size`: 320 changed (limit 371), measured after the float32 save. `work/weights.npz` is bit-identical
  to `work/v5_k6.npz`, the file evaluated in call 5.
- Offline toy drain test (`work/drain_test.py`). Lone halted cars sit at stop lines and leave once their phase has
  been green for 7 s. I tried all singles, all pairs and 150 random triples from each start phase:
  - start network: 15 / 133 / 371 stranded cases;
  - final network: 0 / 0 / 0.
- Decision changes on recorded in-hour honoured states are small: the phantoms change about 0.7% and the k-term
  about 2–3%, mostly toward phase 3.
- Simulator (6 of 10 calls; dev mean TTS, veh-h):

| call | variant | low | nominal | high | all | left at cap |
|---|---|---|---|---|---|---|
| ref | start network, failed detector | 35.5 | 46.8 | 61.8 | 48.0 | – |
| ref | network, all detectors | 34.7 | 44.9 | 54.4 | 44.7 | – |
| 1 | mirror only | 35.0 | 45.7 | 56.1 | 45.6 | 5 |
| 2 | mirror + W2 phantom 0.75 | 35.2 | 45.8 | 56.2 | 45.7 | 4 |
| 3 | mirror + phantoms (1.0 / 0.5) | 35.0 | 45.8 | 55.9 | 45.6 | 0 |
| 4 | + k = 3 | 34.9 | 45.4 | 55.5 | 45.3 | 0 |
| **5** | **+ k = 6 (delivered)** | **34.9** | **45.3** | **54.8** | **45.0** | **0** |
| 6 | + k = 10 | 35.0 | 45.1 | 55.1 | 45.1 | 0 |

- One idea was rejected offline: shrinking the mirror toward its mean (alpha = 0.5). The constant it adds breaks
  the drain logic (stranding returns in the toy test), so I never sent it to the simulator.

## What I expect to go wrong

- The E2 mirror is noisy: pocket-to-pocket correlation is only about 0.3–0.5. When W2 and E2 differ, phase 3 is
  timed for the wrong pocket. This is the main remaining gap to the all-detector network, which is larger in
  high demand.
- k·q_W1u is linear. With a long W1u queue, the imputed W2 goes above 1, which is outside the training range: in
  about 3–7% of recorded states it exceeds 1.3. That can over-serve phase 3 when the W1u queue is really due to a
  full W1 rather than W2 spill-back (more likely under unusual, fixed-time-like queue patterns).
- The phantom also acts during long in-hour phase-2 holds after 12 s, as a soft "max green" toward phases 3 and 0.
  On dev this was neutral, but on other seeds it can cut a useful main-road hold short.
- Only the 9 dev episodes back the choices (k = 6 was the best of 3, 6 and 10), so dev is slightly optimistic.
  Episode-to-episode noise between variants is about ±0.5 veh-h.
- The policy is still very confident (median max-prob 0.99), so it isn't a smooth starting point for further PPO
  training.

## Files

- Builders: `build.py`, `build2.py`, `build3.py`, `build4.py` (history) and `build5.py` (final family; the
  defaults are the delivered net).
- Final weights: `make_final.py` writes `weights.npz`.
- Analysis: `analyze.py`, `sym.py`, `empty.py`, `single_car.py`, `tails.py`, `holds.py`, `phantom_probe.py`,
  `lone.py`, `explore3*.py`, `single_car3.py`, `drain_test.py`, `corr.py`, `condmeans.py`, `mirror_ns.py`,
  `units.py`, `check*.py`.
- Evaluated variants: `v1_proxy.npz` … `v6_k10.npz`.
