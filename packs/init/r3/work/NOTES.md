# Starting network for PPO: queue-score ("max-pressure") controller with a small hold bonus

Files: `rule.py` (the controller as plain numpy), `build.py` (writes `weights.npz` from it), `check.py` (all
checks below), `weights.npz`, `predictions.json`. All weights were set by construction. No optimiser,
regression or search over weights was used, and no simulator was run.

## What I built and why

**What the data showed.**
- The tuned actuated controller (45.3 veh-h, better than fixed-time's 48.3) switches at the first honoured
  decision, at 12 s of green, in 97% of cases.
- When it switches, it goes to the phase with the most halting vehicles. This holds in 96.6% of switches when
  stop-line and upstream queues are counted, with the shared upstream lane split half/half.
- A simple fluid (Webster-style) estimate gives the same picture. At these demands the optimal cycle (about
  40–65 s) is at or below the 60 s that four 12 s minimum greens already force. So extending a green past
  12 s only pays while that phase still has a queue to discharge, not for free-flow arrivals.

**The rule** (`rule.py`). Every 5 s it scores each phase p in vehicles:

    V_p = Σ stop-line lanes of p   3·(0.5·q + 0.5·d)              halting car = 1, moving car = 0.5
        + Σ upstream lanes of p    share·34.1·(0.9·q + 0.1·d)     halting car = 1, moving car = 0.1
        + 0.6 if p is the current phase                            (cost of a switch: 3 s yellow + start-up)
    phase = argmax_p V_p

- **Upstream lane shares.** X0u feeds only through traffic (share 1). X1u feeds both the straight lane and
  the left pocket. In the data, a queue on X1u comes with a full left pocket 70–83% of the time under
  actuated control, so most of that queue is left-turn spillback. X1u is therefore split 0.35 to through
  and 0.65 to left.
- **The bonus** is below 1 on purpose. A single halting car (V = 1.0) must always beat an idle green.
  With 1.0, a lone car tied the current phase, and during the drain period it could have waited forever.
- **What the rule does.**
  - It switches at 12 s to the fullest phase.
  - It holds the green only while the current phase still has queued or discharging vehicles worth more than
    the best alternative.
  - It holds an idle green when nothing waits anywhere.
  - It skips phases that have nobody waiting.
- **Agreement with the recorded controllers** (decisions where the minimum green has passed):
  - actuated: 89.5%. Most disagreements are near-ties between two red phases. Both hold the green in about
    3% of these decisions (rule 3.0%, actuated 3.3%).
  - fixed-time: 31%. Fixed-time holds long, fixed greens.

## Which units and weights do what

`V_p = A[p] · x` is a fixed linear map of the 46 inputs (`rule.coef_matrix()`). Coefficients in vehicles
per unit of input:
- stop-line q and d: 1.49 each;
- full-share upstream lane: q 30.7, d 3.4;
- X1u through share: q 10.8, d 1.2;
- X1u left share: q 20.0, d 2.2;
- current phase one-hot: 0.6.

| layer | units | weights |
|---|---|---|
| L1 | 0–11: unit 3p+j = tanh(a_j·V_p − c_j) | (a,c) = (0.30, 0.0), (0.30, 1.2), (0.08, 1.2). Two steep units resolve 0–8 vehicles. A shallow "backbone" unit keeps large queues ordered up to about 40 vehicles. |
| L1 | 12–63 | random initial weights, unchanged |
| L2 | 0–11: unit u = tanh(1.2·h1[u]) | one weight per row. Every other W2 entry in these rows is 0 but still trainable. |
| L2 | 12–63 | random initial weights, unchanged |
| out | logit_p = 4.25·Σ_j h2[3p+j] − 2.08 | b3 is the same for all phases and only centres the logits. W3 columns 12–63 keep their random initial values (about 1e-3). |

- **Argmax is exact.** All four phases go through the same increasing map from V to logit, so
  argmax(logits) = argmax(V).
- **Shape of the map.** Logits rise about 1.7 per vehicle for V from 0 to 5, then about 0.4 per vehicle up
  to 20. Small-queue decisions are crisp. Choices between two large queues stay soft, which leaves PPO room
  to explore.
- **How much changed.** 1380 of 7428 parameters differ from `start_weights.npz`.

## Checks (`python work/check.py`)

`tools.trainability(w)` on states.npz:

| | value | requirement |
|---|---|---|
| median_max_prob | **0.909** | ≤ 0.95 |
| share_max_prob_above_0.99 | **0.221** | ≤ 0.30 |
| max_abs_logit | **8.56** | ≤ 15 |
| sat_l1 / sat_l2 | 0.004 / 0.000 | ≤ 0.05 recommended |
| dead_l1 / dead_l2 | 0 / 0 | |
| median_logit_gap | 2.51 | |

The same metrics on subsets of the states:

| states | median_max_prob | share > 0.99 | note |
|---|---|---|---|
| ok=1 only (decisions that change the signal) | 0.924 | 0.227 | |
| actuated-recorded | 0.826 | 0.111 | closest to the traffic this controller will see in PPO rollouts |
| fixed-recorded | 0.965 | 0.330 | longer queues, so larger gaps |

Other checks:
- **Network vs rule.py.** They agree on 99.93% of the recorded states. All 6 disagreements are exact ties
  (rule gap ≤ 0.03 vehicles) broken by the tiny random W3 entries. `tools.fidelity("work/rule.py", w)`
  gives honoured 0.9994.
- **Synthetic stress test.** On 20,000 random states with upstream queues up to half a lane (V up to 50),
  the network agrees with the rule on 99.88%. Only one disagreement has a gap above 0.05 vehicles
  (top score 23). On these states saturation is sat_l1 0.04 and sat_l2 0.
- **Hand-made situations.** All 8 give the intended phase:
  - empty network: hold the current phase;
  - 1 car on a red left pocket: switch to it (p = 0.54);
  - drain with 1 car left: switch to it;
  - current phase still discharging an upstream queue: hold (p = 0.999);
  - full NS pockets with spillback: NS left (0.999);
  - 25-vs-20-vehicle heavy queues: the larger one (0.71).
- **Only `work/` was written.** The pack's `__pycache__` was not modified.

## Prediction

The controller is essentially the tuned actuated logic: same 12 s switching, same "fullest phase" choice.
It adds a hold when the current queue is still discharging, counts approaching vehicles a little, and
handles left-turn spillback. I expect actuated-level TTS, about 34.5 / 45.0 / 56.6 veh-h (low / nominal /
high). Uncertainty is about ±2 veh-h, since no simulation was run.

## What I expect to go wrong

- **Untested weights.** The weights for moving vehicles (0.5 and 0.1), the 0.6 bonus and the 0.35/0.65 split
  are reasoned, not measured. Each could be a little off, in either direction, compared with actuated.
- **No waiting-time memory.** The rule only sees counts. A lone left-turner can wait two or more cycles at
  low demand while through phases keep refilling, just as with the actuated controller. PPO cannot fix this
  either, since the observation carries no age.
- **High demand.** Queues may grow beyond what the upstream observation shows, and vehicles waiting to enter
  are not observed at all. Choices between two big queues use soft probabilities; the argmax is still right.
- **Sampling cost during training.** At low demand the probabilities are deliberately soft (0.5–0.6 for a
  one-car decision). PPO's sampled rollouts will therefore make more needless switches than the greedy
  policy, and early training returns will look worse than the deterministic test. When sampling, the expected share of
  non-argmax picks is 20% on the actuated-like decision states, 15% on all ok=1 states and 17% on all
  states.
- **Exact ties.** Ties (common, since counts are discrete) are broken by the random initial W3 entries. This
  is harmless, but tie-breaks will shift as soon as PPO updates W3.
