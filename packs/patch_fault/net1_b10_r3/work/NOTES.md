# Repair of the PPO signal controller after the W2 detector failure

Delivered: `work/weights.npz` (built by `work/make_weights.py`, identical to the evaluated candidate `v3_full.npz`).
It changes **368 of 7428 parameters**: 308 in W1 and 60 in b1. W2, b2, W3 and b3 are untouched.
Dev TTS (eval call 3): **45.2 veh-h** (low 34.8, nominal 45.5, high 55.4). No vehicles were left at the drain cap.
For comparison: start network with the failed detector 214.3, fixed-time 48.3, actuated 49.5, and the network with
every detector working 44.5.

## What went wrong

The network relies heavily on the W2 pocket inputs (q_W2 = x[11], d_W2 = x[23]). On the recorded states, setting
q_W2 = d_W2 = 0.33 (one car) raises the share of EW_left (phase 3) decisions from 14% to 40%. With the detector stuck
at 0, the network almost never picks phase 3 under heavy load. In 3 of the 6 recorded start-network episodes, the
W and E left pockets fill up and left-turners queue back into the upstream lanes W1u and E1u. Those left-turners
block the straight traffic behind them, while the network keeps EW_through green forever (logits about
[-0.1, -2.5, 1.2, 0.4]). That is the gridlock behind the 556 veh-h high-demand mean. There is a second problem: when
the junction is otherwise empty, the network just holds its current phase, so a W left-turner it cannot see waits
until the 30-minute drain cap. The actuated controller's recorded episode 5 shows the same stranding.

## What I built

The two dead input columns `a11 = W1[:,11]` and `a23 = W1[:,23]` are kept as they are. I make them useful again by
feeding them an estimate of the W2 pocket: I add them, scaled, onto the columns of the live inputs the estimate is
built from. This is a fixed linear "virtual sensor":

    W2hat = E2 + 0.6*(t_green - 0.2) - 0.25*phase3 + 3*q_W1u     (the q and d channels each use their own E2 input)

| term | why | weights changed |
|---|---|---|
| `q_E2` -> q_W2, `d_E2` -> d_W2 | The opposite left pocket is served by the same phase (EW_left) and has the same demand, including the E/W directional bias. When q_W1u = 0 the proxy averages 0.35, against 0.32 on the E side. | `W1[:,5] += a11`, `W1[:,17] += a23` (128) |
| `0.6*(t_green - 0.2)` | W left-turners keep arriving while another phase is green. The term is zero at the usual decision point (12 s of green) and reaches +0.48 after 60 s, which pushes an idle junction to serve EW_left and frees invisible W2 cars. | `W1[:,45] += 0.6a`, `b1 -= 0.12a` (120) |
| `-0.25*phase3` | During EW_left the W2 pocket is being emptied, so the timer must not keep phase 3 green. Without this term the network held phase 3 when idle and stranded single cars elsewhere. | `W1[:,43] -= 0.25a` (60) |
| `3*q_W1u` | Halted cars on W1u, the upstream lane that feeds W1 and W2, usually mean the W2 pocket has spilled back. On the E side, q_E2 averages 0.70-0.76 when q_E1u > 0, against 0.32 when it is 0. When q_W1u > 0, the E2 proxy only averages about 0.5, short by about 0.2-0.3 at q_W1u = 0.03-0.12, which is a gain of about 3. | `W1[:,31] += 3a` (60) |

Here `a = a11 + a23`, so the timer, phase and spillback terms enter both the q and the d channel. Those three terms
are written only on the 60 units with the largest |a|; the other 4 units have |a| <= 0.011 and carry 0.003% of the
energy. All numbers come from physical reasoning and simple conditional means of the recorded states. Nothing was
fitted, regressed or searched.

## Checks run

- Edit size: 368 changed parameters (`tools.edit_size`), within the 371 limit.
- Counterfactual on the gridlock state: with W2hat = 1, the network picks phase 3 (logit 1.77 vs 1.25).
- The timer's offset leaves decisions at 12 s of green unchanged. On recorded honoured states, terms 2-4 change
  3.0% (start network) and 2.5% (fixed-time) of the decisions made with the E2 proxy alone. For the actuated
  controller the figure is 22%, almost all of it in its long idle phase-0 holds, where moving to phase 3 is the
  intended effect.
- Frozen-queue phase-machine check (`idle_check.py`, `sparse_check.py`): one to three halted cars are placed on
  random lanes, W2 stays invisible, and the network's own decisions are stepped with the 10 s minimum green and 3 s
  yellow. The share of states where a car is never served drops from 27% (start) and 24% (E2 proxy only) to 6.7%.
  Every remaining case is a lone car on N1, a flaw the original network already has.
- Simulator calls: 7 of 10. Dev mean TTS per call:

| call | variant | mean | low / nominal / high | left at drain cap |
|---|---|---|---|---|
| 1 | E2 proxy only | 46.0 | 34.8 / 45.6 / 57.5 | 2 and 5 cars in two episodes |
| 2 | + timer 0.75, no phase-3 correction | 45.7 | 34.8 / 45.4 / 56.7 | 1 car |
| **3** | **+ timer 0.6, phase-3 correction 0.25, spillback 3 (delivered)** | **45.2** | **34.8 / 45.5 / 55.4** | **0** |
| 4 | same, spillback 5 | 45.2 | 34.8 / 45.4 / 55.6 | 0 |
| 5 | same, no spillback | 45.5 | 35.0 / 45.4 / 56.0 | 0 |
| 6 | call 3 + N1 fix (0.5), terms on 44 units | 45.3 | 34.9 / 45.3 / 55.8 | 0 |
| 7 | call 3 + N1 fix (0.6), terms on 44 units | 45.5 | 34.9 / 45.2 / 56.3 | 0 |

The spillback term helps under high demand (episode high_1101: 58.9 without it, 56.9 with it). Gains of 3 and 5
perform the same. The "N1 fix" (q_N1 also drives the q_N0/d_N0 columns) removes the lone-N1 stranding in the frozen
check, but it gave the main road less green under high demand and did not improve dev TTS, so I did not use it.

## What I expect to go wrong

- The estimate is right on average but wrong in individual cases: E and W arrivals are independent, so phase 3 will
  sometimes run for E2 alone, or start late when only W2 is busy. The remaining gap to the full-detector network is
  about 0.3-1 veh-h per scenario, and largest under high demand.
- High demand is chaotic: one changed decision can move an episode by about 2 veh-h. With 3 dev episodes per
  scenario, differences under about 0.5 veh-h between my variants are noise, and choosing the best of them is
  slightly optimistic.
- Strandings at the drain cap can still happen: a lone last car on N1, inherited from the original network. If it
  happens, it costs up to about 0.5 veh-h in that episode. An invisible W2 car is served within about 50 s of idle.
- The spillback term overestimates W2 when W1u holds a long queue of straight traffic (W1 full during a long red).
  This leads to an extra EW_left phase, which is not costly in that state.
- The network now changes more abruptly: its median top-action probability is 0.985 against 0.965 for the start
  network, and 4.0% vs 1.4% of first-layer pre-activations are saturated. That would matter only if it were trained
  further.

## Files

`build.py` (the construction, all candidates), `make_weights.py` (delivered weights), `make_v1.py`, `make_v6.py`,
`make_v7.py` (rejected candidates), `diag.py`, `stuck.py`, `idle_check.py`, `sparse_check.py` (checks),
`v*.npz` (evaluated candidates), `predictions.json`.
