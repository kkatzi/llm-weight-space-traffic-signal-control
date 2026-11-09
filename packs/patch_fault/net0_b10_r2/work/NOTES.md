# Repair of the network with the dead W2 detector

## What I built and why

The W2 detector (west left-turn pocket) always reads 0, so the first-layer columns W1[:,11] (q_W2) and
W1[:,23] (d_W2) multiply 0 and contribute nothing. With W2 reading as empty, the network under-serves phase 3
(EW_left): in the recorded start-network episodes it chose phase 3 in only 21 % of honoured decisions. Left-turners
then fill the 3-car pocket and spill back into W1u, where they block straight traffic. In two of the six recorded
start-network episodes vehicles were stranded until the 30-minute drain cap. In one of them, 6 left-turners were
halting on W1u under a W-through green, and the network sat in phase 2 from t = 3650 s to 5400 s.

The fix is **input imputation inside the first layer**. I estimate the missing inputs as linear functions of live
inputs and fold the estimate into the weights. Pre-activations are linear in x, so
`W1[:, j] += c_q[j] * W1_orig[:, 11] + c_d[j] * W1_orig[:, 23]` makes the network behave exactly like the
original network fed the estimate. I checked this numerically: the largest logit difference is 1e-7, which is
float32 rounding. Every number was chosen by construction, guided by statistics of the recorded states. Nothing
was fitted.

Estimate fed to the original W2 weights (`ok` = 1 in every decision that can take effect, so it acts as an intercept):

    q_W2_hat = q_E2 + 8 * q_W1u + 1.0 * (t_green - 0.2 * ok)
    d_W2_hat = d_E2 + 8 * q_W1u + 1.0 * (t_green - 0.2 * ok)

plus a "phantom" added to the four NS-through inputs (q and d of N0, N1, S0, S1), each:

    + 0.5 * (t_green - 0.2 * ok)

1. **Mirror term (q_E2, d_E2 -> W2).** The E2 pocket mirrors W2: it has the same demand, the same directional
   bias and the same serving phase (3). This term alone (candidate A) brought the dev mean from 48.0 to 45.6.
2. **Spillback term (8 * q_W1u).** W1u feeds both W1 and W2. On the mirror side, whenever E1u holds any halting
   vehicle and phase 3 is not green, the E2 pocket is full about 90 % of the time (87–94 % in phases 0–2).
   So a halting queue on W1u is strong evidence that W2 is full, even when E2 is empty. This term is the one that
   releases the stranded W1u left-turners. One halting car (0.029) adds 0.23, and four cars add about 0.9. I
   checked that the original network is monotone in the W2 input even above 1: more W2 always means a larger
   phase-3 margin. So an estimate above 1 only pushes harder toward phase 3.
3. **Long-green priors, zero at t_green = 0.2.** 12 s of green is the first decision that can switch, and it
   accounts for 81 % of honoured demand-hour decisions. Anchoring both priors there leaves that decision
   untouched. The priors grow only when a green lasts long, which during the demand hour almost never happens
   (11 of 4939 honoured states have t_green > 0.5). It happens at the end of an episode, when the junction looks
   empty:
   - the W2 prior makes the network visit phase 3 after about 48 s of an apparently empty phase-2 rest, so hidden
     W2 cars get served;
   - the NS prior fixes a flaw inherited from the trained network: a single car on any NS-through lane is
     *never* served, because the network rests in phase 2 forever. With the prior, a lone NS car is served after
     36–48 s.
   Without these priors, candidates A and C left 5 vehicles at the drain cap across the 9 dev episodes, and B
   left 1. With them, D and D8 left 0.

## Which weights changed

Only five columns of W1 changed: q_E2 (col 5), d_E2 (col 17), q_W1u (col 31), ok (col 44) and t_green (col 45).
That is 5 x 64 = **320 parameters** (limit 371). b1, W2, b2, W3 and b3 are untouched, and so are W1 columns
11 and 23, which multiply 0 anyway. No unit is repurposed: every hidden unit keeps its trained meaning and simply
receives the estimated W2 signal.

## Checks run

- `tools.edit_size`: 320 changed. The weights pass `tools.save`'s shape and finiteness checks.
- Exact linearity equivalence with "original net + estimated inputs" (1e-7).
- Probe tables of synthetic states. The cases covered are: empty junction, a single car on each lane, two cars
  on NS lanes, and W1u queues, each from every current phase at t_green = 0.2 … 1. The final network serves every
  single-car case and visits phase 3 from an empty phase-2 rest.
- Decision changes on recorded states: 1156 of 4939 honoured demand-hour decisions differ from the start network,
  almost all of them toward phase 3. The phase-3 share rose from 0.21 to 0.42, and the probes show the
  mirror-only network (A) already picks phase 3 for a single visible E2 car.
- Simulator, 5 of 10 calls (dev mean TTS, veh-h; low / nominal / high; vehicles left at the drain cap):

| call | candidate | mean | low | nominal | high | left |
|---|---|---|---|---|---|---|
| 1 | A: mirror only (128 params) | 45.6 | 35.0 | 45.7 | 56.1 | 5 |
| 2 | B: + spillback (5) + W2 prior (anchor 0.25) | 45.1 | 34.8 | 45.3 | 55.1 | 1 |
| 3 | C: as B, calibrated mirror 0.15 + 0.6*E2 | 45.2 | 35.2 | 45.5 | 54.8 | 5 |
| 4 | D: B-style, priors anchored at 0.2 + NS prior | 45.1 | 35.0 | 45.5 | 54.8 | 0 |
| 5 | **D8 = final**: D with spillback weight 8 | **45.0** | 35.0 | 45.1 | 55.0 | 0 |

  References: start network with the fault 48.0 (35.5 / 46.8 / 61.8); trained network with every detector
  working 44.7 (34.7 / 44.9 / 54.4).
- Paired per-episode differences between candidates are about ±0.5 veh-h, so B, C, D and D8 are statistically
  tied on the demand hour. I chose D8 for robustness: it strands no vehicles, its rest behaviour is clean in the
  probes, and its spillback evidence is the strongest. The calibrated mirror (C-style) was rejected because in
  the probes its intercept makes the phantom W2 outweigh a lone NS car, which can then lock phase 3.
- `tools.trainability`: median max-prob 0.988, no dead units, sat_l2 0.14 (the start network had 0.07).

## What I expect to go wrong

- The W2 estimate is still a guess. When W2 is full but E2 is empty and W1u has no queue yet, the network is
  late with phase 3. When E2 is full but W2 is empty, it gives phase 3 a little too much. This is the main reason
  it remains slightly worse than the trained network with every detector working.
- The linear priors can reinforce themselves. A phase-3 green held for 36 s or more, or a phase-0 green held for
  48 s or more, raises the phantom demand for the same phase and can hold it even with a lone car waiting
  elsewhere. Greens that long never appeared in the dev runs (switch counts stayed at 241–246 per episode), but
  an unusual test episode could trigger it.
- The spillback term also fires when through traffic queues on W1u during the phase-3 green (about 50 of 1128
  recorded phase-3 states). It may then extend phase 3 slightly.
- Tail strandings are now much rarer but not impossible, for example two hidden W2 cars that arrive in an awkward
  order. Each stranded vehicle costs about 0.47 veh-h for its episode.
- The 9 dev episodes cannot separate candidates closer than about 0.3 veh-h. Expect the test means to sit
  slightly above the dev numbers.

## Files

- `impute.py`: helper that folds a linear estimate of (q_W2, d_W2) into W1.
- `build_a.py`, `build_b.py`, `build_c.py`, `build_d.py`, `build_d8.py`: the evaluated candidates (`cand_*.npz`).
- `build_final.py`: writes `weights.npz` (identical to `cand_d8.npz`) and runs the checks above.
