# Hand-built PPO starting network for junction J0

## What I built and why

The network implements a demand-based controller with a short hold bias. It is close to the tuned actuated
controller but also uses upstream information.

- **Phase demand.** Every phase p gets a demand score D_p in vehicle-equivalents (exact weights in `rule.py`):
  - halting vehicle on one of its stop-line lanes: 1.0;
  - moving vehicle on one of its stop-line lanes: 0.3;
  - halting vehicle on an upstream lane that feeds it: 0.8;
  - moving vehicle upstream: 0.3. These arrive during the next green.
  - The shared upstream lane 1 (straight + left) is split between through and left. Halting vehicles are split
    35 / 65, because the data show they are mostly left-turners spilling back from a full 3-car pocket. Moving
    vehicles are split 40 / 60, the expected left share of that lane.
- **Decision.** Pick the phase with the largest D_p. The current phase gets a bonus of about 1 vehicle (about 1.5 at
  very low demand). So it holds only when nothing else is really waiting, or when its own queue is still large.

Why this design:
- In the recorded data, queues clear within about 7 s of green, so the 12 s minimum green is nearly always enough.
- The tuned actuated controller almost always switches at 12 s to the phase with most halting vehicles. It agrees
  with "argmax of stop-line halting" on 92 % of its switches.
- My controller keeps that behaviour and adds:
  - upstream queues, which break the ties that come from stop-line lanes saturating at 3 cars;
  - approaching platoons;
  - not switching when nothing else is waiting.

When ok = 0 the pick is ignored by the junction. There the network moderately prefers the current phase (gap of
about 3.6 logits). This gives PPO less gradient noise from these states and does not change any honoured decision.

## Which units do what

Units are numbered from 0. All other units keep their random PPO initialisation, including their tiny output
weights, so they are alive and trainable.

| layer | units | what they do |
|---|---|---|
| L1 | 0-7 | `F[p,j] = tanh(0.1*(D_p - c_j))`, c = 4 and 14 vehicles, two per phase (0-1 = NS_through, 2-3 = NS_left, 4-5 = EW_through, 6-7 = EW_left). The pair sums to a nearly linear demand signal over 0-20 vehicles that compresses gently above that. |
| L1 | 8-11 | `P_p = tanh(2*phase_p - 1)` = ±0.76: current-phase flag |
| L1 | 12-15 | `G_p = tanh(3*phase_p - 3*ok - 1.5)`: about +0.9 only when p is current and ok = 0 |
| L2 | 0-7 | `L[p,j] = tanh(0.8*(F[p,0] + F[p,1] + 0.103*P_p - 0.3) ± 0.25)`: demand plus hold bonus, two per phase |
| L2 | 8-11 | `H_p = tanh(G_p)`: min-green hold |
| out | | `logit_p = 5*(L[p,0] + L[p,1]) + 2.5*H_p`, b3 = 0 |

Every phase passes through the same monotone function. So the argmax equals argmax_p [G(D_p) + β·phase_p], which
is the rule. Weights are set by construction (`build.py`, using `rule.score_matrix()`). The only numbers chosen by
looking at the data are:
- the shared-lane split;
- the logit scale A = 10 and the ok = 0 weight K0 = 2.5. I picked these from a small hand grid of forward passes to
  meet the trainability targets.

No optimiser, regression or imitation fit was used.

## Checks I ran

Scripts: `checks.py`, `sanity.py`, `check_rule.py`, `explore*.py`.

**`tools.trainability`** on `states.npz`:

| median_max_prob | share_max_prob_above_0.99 | median_logit_gap | max_abs_logit | sat_l1 | sat_l2 | dead_l1 | dead_l2 |
|---|---|---|---|---|---|---|---|
| 0.887 | 0.155 | 2.51 | 10.43 | 0.019 | 0.000 | 0 | 0 |

- All required limits are met (≤ 0.95, ≤ 0.30, ≤ 15) and so are the recommended ones.
- sat_l1 comes from the four min-green gate units: (other phase, ok = 1) has pre-activation −4.5.
- On ok = 1 states only, the median max probability is 0.78. Low confidence is concentrated at near-ties.

**Fidelity to `rule.py`**: 97.9 % on honoured (ok = 1) states. All 74 disagreements are low-demand near-ties (rule
gap < 0.5 vehicles) where the network holds and the rule switches.

**Agreement with the recorded controllers** on ok = 1 states:
- actuated: 84 %;
- fixed-time: 31 %.

On actuated states the network picks NS_left less often (9 % vs 16 %) and EW_through more often.

**Synthetic states** (`sanity.py`): 8 of 8 expected decisions, plus one ok = 0 case with no expected answer. The
cases cover:
- empty network → hold;
- full opposing queues → switch;
- full EW pockets with spillback → EW_left;
- own queue still discharging → hold;
- approaching platoon with nothing waiting → go to it.

**Perturbation robustness**: Gaussian noise on every weight, ok = 1 states.

| noise σ | decisions flipped | flips where the rule gap is > 2 vehicles |
|---|---|---|
| 0.003 | 1.7 % | 0 % |
| 0.01 | 5.4 % | 0.01 % |
| 0.03 | 16 % | 1.7 % |

Small updates therefore only move near-tie decisions.

## What I expect to go wrong

- **It has never run in closed loop.** Agreement with actuated is only 84 %. The differences come from:
  - the hold bonus;
  - the 0.3 weight on approaching vehicles;
  - the shared-lane split.

  These are reasoned guesses, not tuned. The result could be 1-3 % worse than actuated as easily as better.
- **NS_left is served less often.** That can lengthen left-turn waits and cause pocket spillback that blocks
  straight vehicles in upstream lane 1, mainly in the high scenario and with an E/W bias.
- **No maximum green.** Under a strong directional bias, a busy phase could be held for a long time while light
  phases wait. The halting weight of 1.0 vs 0.3 for moving vehicles should cap this, but it is not guaranteed.
- **PPO starts from a fairly exploratory policy at ok = 1** (median max prob 0.78). Training-time behaviour is
  noisier than the greedy policy, and early updates may first sharpen near-tie decisions.

## Predictions

Mean test TTS in veh-h: low 34.6, nominal 45.2, high 56.8. These are close to the tuned actuated dev references
(34.4 / 45.0 / 56.6).
