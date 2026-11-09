# Task: improve a trained signal-control network by editing a few of its weights

## Goal

`start_weights.npz` is a network trained with PPO on this junction. Improve it so that the edited network has a
lower test TTS than the original, by changing **at most 371 of its 7428 parameters (5%)**. Any value that differs
from the original counts as changed. No further training happens: your edited network is tested exactly as
delivered.

How the network was trained: Stable-Baselines3 PPO:
- configuration: clip 0.2, learning rate 0.0003, entropy coefficient 0.01, discount 0.99, GAE lambda 0.95, 4
  parallel episodes per update (720 decisions each), minibatch 720, 10 epochs, normalised rewards;
- reward per decision: -(delay over the 5 s)/10, where the delay counts, every second, 1 - speed / speed limit for
  each vehicle in the network plus 1 for each vehicle waiting to enter, averaged over the 5 s;
- training demand: overall level 0.7-1.25 x nominal, a random 15-minute profile (multipliers 0.6-1.25) and a random
  directional bias (E and W entries x e^b, N and S entries x e^-b, b between -0.2 and +0.3);
- training episodes run like the test episodes, until the network is empty;
- training length: 3,000,000 environment steps (one step = one decision).

## The traffic system (SUMO microsimulation, 1 s steps)

One signalised junction J0 with four approaches (N, E, S, W), each 300 m long, speed limit 50 km/h. Each approach is
a road that widens 40 m before the junction, where a left-turn pocket opens:
- All approaches: at the stop line 3 lanes, lane 0 = right + straight, lane 1 = straight, lane 2 = left-turn pocket;
  these lanes are 22.4 m long (about 3 cars each). Upstream 2 lanes, 256 m long: upstream lane 0 = right + straight,
  continues into lane 0; upstream lane 1 = straight + left, continues into lanes 1 and 2. Exits have 2 lanes.

Demand at nominal level: 750 veh/h per E and W approach (the main road) and 450 veh/h per N and S approach. From
every approach, 55 % go straight, 30 % turn left, 15 % turn right.

**Signals.** The junction has 4 protected green phases:

| phase | green movements |
|---|---|
| 0 NS_through | from N and S: straight + right |
| 1 NS_left | from N and S: left |
| 2 EW_through | from E and W: straight + right |
| 3 EW_left | from E and W: left |

Every 5 s the controller picks a phase. If the pick differs from the current green and the current green has lasted
at least 10 s, the junction shows 3 s of yellow and then the new green (so at the next decision the new green is 2 s
old). A different pick before 10 s of green is ignored. Any phase may follow any phase; there is no maximum green.

**Demand over time.** One hour of traffic in four 15-minute intervals with nominal multipliers 0.75, 1.0, 1.15, 0.9,
then no new traffic until the network is empty (capped at 30 more minutes). Arrivals are random.

**KPI.** Total time in system, TTS (vehicle-hours): every vehicle's time in the network plus its time waiting to
enter, summed over the episode. Lower is better. Teleporting is disabled, so vehicles that are not served stay in
the network and keep adding to TTS.

## The network

An SB3 PPO actor (MlpPolicy): 46 inputs -> 64 tanh -> 64 tanh -> 4 logits (7428 parameters). The decision is
`argmax(logits)`, deterministic. Forward pass, exactly as evaluated:

    h1 = tanh(W1 @ x + b1);  h2 = tanh(W2 @ h1 + b2);  logits = W3 @ h2 + b3

Weight shapes (npz keys, torch convention [out, in]): W1 (64, 46), b1 (64,), W2 (64, 64), b2 (64,), W3 (4, 64), b3
(4,).

## Observation x (46 values, all in [0, 1])

- `queue` 0-11: halting vehicles (speed < 0.1 m/s) on each stop-line lane / lane capacity;
- `density` 12-23: vehicles on each stop-line lane / lane capacity;
- `up_queue` 24-31 and `up_density` 32-39: the same for the upstream lanes;
- `phase` 40-43: one-hot of the current green phase;
- `ok` 44: 1 if the current green has lasted ≥ 10 s (a switch request would be honoured);
- `t_green` 45: time in the current green / 60 s, clipped to 1.

Lane capacity = lane length / 7.5 m; every value is clipped to 1.
- Stop-line lanes, in order: N0 (right + straight), N1 (straight), N2 (left pocket), E0 (right + straight), E1
  (straight), E2 (left pocket), S0 (right + straight), S1 (straight), S2 (left pocket), W0 (right + straight), W1
  (straight), W2 (left pocket).
- Upstream lanes, in order: N0u (feeds N0), N1u (feeds N1 and N2), E0u (feeds E0), E1u (feeds E1 and E2), S0u (feeds
  S0), S1u (feeds S1 and S2), W0u (feeds W0), W1u (feeds W1 and W2).

Lane names: approach letter + lane index (lane 0 is the rightmost); `u` marks an upstream lane. `tools.names()`
lists all entries; `tools.block(x)` shows one observation by name.

## Reference controllers on the 9 dev episodes (mean TTS, veh-h)

| controller | low | nominal | high | all 9 |
|---|---|---|---|---|
| fixed-time (tuned) | 37.7 | 48.4 | 58.7 | 48.3 |
| actuated (tuned) | 34.4 | 45.0 | 56.6 | 45.3 |
| the trained network (start_weights.npz) | 34.5 | 44.6 | 54.4 | 44.5 |

- **Fixed-time:** a cycle 0 -> 1 -> 2 -> 3 with green times tuned on the dev episodes.
- **Actuated:** after the minimum green, extends the current green while vehicles still move on its lanes, up to a
  maximum green; otherwise it switches to the phase with the most halting vehicles. Its thresholds and detector
  lanes are tuned on the dev episodes.

## Files

- `TASK.md`: this file.
- `tools.py`: load, save, forward, hidden, greedy, probs, states, names, block, edit_size, trainability, fidelity
  and count_comparisons helpers (read its docstring).
- `spec.json`: the observation layout, the lane names and the dev references.
- `start_weights.npz`: the trained network.
- `states.npz`: recorded observations of THIS junction from 6 training-like episodes per controller (start_network:
  4433 decisions, fixed: 4445 decisions, actuated: 4429 decisions). Keys: obs, act (the decision taken), episode,
  time_s, controller (index into controller_names).
- `eval.py`: runs the simulator on your weights (budgeted, see rule 3).

## How your result will be tested (after you finish; you will not see these runs)

Your network runs on 30 held-out episodes with seeds you have no access to, 10 each of
- **low**: nominal x 0.8;
- **nominal**: the demand described above;
- **high**: nominal x 1.15.

Score: mean TTS over the 30 episodes, also per scenario.

## Python

Use `/home/traffic-ntua/anaconda3/envs/dlr/bin/python`. It has numpy, which is all `tools.py` needs.

## Rules

1. Work only inside this folder. Read only files in this folder; write your files to `work/`. Do not read any other
   file on this machine: no project code, models, results or other experiments.
2. Do not run SUMO, TraCI or libsumo, any training, or any simulator other than what rule 3 allows.
3. You may run the simulator on YOUR weights through `python eval.py work/<file>.npz`, at most 10 times in total.
   Each call runs the 9 dev episodes (3 seeds each of low, nominal and high demand; seeds differ from the test),
   prints TTS per episode and per scenario, leftover vehicles and switch counts, and is logged.
4. **No fitting of weights to data.** No gradient steps on any loss (SGD, Adam, ...), no least-squares or other
   regression fit of weights, no random or evolutionary search over weights, and no imitation fit of a controller's
   actions. Set the values by construction. You may compute statistics of the recorded data (means, quantiles,
   correlations, activations) to choose numbers such as thresholds, and you may run forward passes freely, on
   recorded or synthetic inputs.
5. At most 371 parameters may differ from `start_weights.npz` (check with `tools.edit_size`). More than that
   disqualifies the edit.

## Deliverables (in `work/`)

- `weights.npz`: written with `tools.save(w, "work/weights.npz")`, which checks the shapes.
- `NOTES.md`:
  - what you built and why, in plain words;
  - which units or weights do what;
  - the checks you ran;
  - what you expect to go wrong.
- `predictions.json`: your predicted mean test TTS (veh-h) per scenario for YOUR network as delivered: `{"low": x,
  "nominal": y, "high": z}`.
- Keep your scripts in `work/` too.
