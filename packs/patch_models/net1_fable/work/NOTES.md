# NOTES: hand edit of start_weights.npz

## What was delivered

`work/weights.npz` = `start_weights.npz` with **2 parameters changed**: `b3[0] += 0.5` and `b3[1] += 0.5`
(the output biases of phase 0 NS_through and phase 1 NS_left). Everything else is untouched.
Built by `work/build3.py 0.5 work/weights.npz`.

## Why this edit

The network was trained with a random directional bias: E/W entries x e^b, N/S entries x e^-b, b uniform in
[-0.2, 0.3]. Averaged over training, EW demand was x1.06 and NS demand x0.96 of nominal, i.e. the net learned
to value EW phases for a world where the EW/NS ratio is about 10 % higher than in the unbiased test demand.
The edit shifts the preference back toward the NS phases.

Size of the shift, set by a statistic of the recorded states (no fitting): on the honoured decisions of the
start network, scaling all NS queue/density inputs by e^{+0.05} and all EW inputs by e^{-0.05} (the same 10 %
ratio) moves the net's own (NS logits - EW logits)/2 by +0.53 on average. So +0.5 on the two NS logits is the
size of preference change the net itself associates with a 10 % demand-ratio change. It flips about 2 % of the
honoured decisions on the recorded states; the phase mix moves from (0.238, 0.179, 0.329, 0.253) to
(0.252, 0.187, 0.316, 0.245).

## Which units/weights do what

Only `b3`: a constant added to the logits of phases 0 and 1. No hidden unit was changed, so the network's
internal features and its in-distribution behaviour are intact; the edit only tips close calls between an NS
phase and an EW phase toward NS.

## What else was tried (and why it was dropped)

Diagnosis of the start network on the recorded states: it switches at 95 % of the honoured decisions, so it
runs 12 s greens + 3 s yellow continuously (about 243 switches per episode) and the job is purely "which phase
next". The junction is never over-saturated (upstream halting share never above 0.3) and never idle during the
hour; the stop-line lanes of EW_through hold a median of 1 car at the 12 s decision. The net's picks agree
with "max halting queue" in 68 % of cases and beat the tuned actuated controller by 0.8 veh-h on dev.

1. **Constant stay bonus** (`work/build.py`, 317 params): layer-1 unit 31 rewritten to encode the current
   phase as 4 tanh levels, its old W2 column severed, layer-2 units 25/35/54 rewritten as threshold decoders
   so that the current phase's logit gets +beta. Verified exactly on all recorded states (max deviation 0.004).
   Simulator: beta=3 -> 142.8 veh-h (high scenario locks up: ~500 vehicles left), beta=6 -> 1107 veh-h. Once
   the net stays past 12 s the queues grow into states it never saw and it keeps staying. Dropped.
2. **Time-gated bonus** (`work/build2.py`, 316 params): same decoders, but the level unit also reads t_green so
   the bonus exists only at the 12 s decision and vanishes from 17 s on (no lock-in possible). Verified exactly.
   Simulator: beta=3 -> 44.55, beta=6 -> 44.64 vs baseline 44.51. Neutral; dropped as 300 parameters for nothing.
3. **NS bias** (`work/build3.py`): +0.5 -> 44.48 (delivered); +1.0 -> 44.61.

## Checks run

- `tools.edit_size`: 2 of 7428 parameters changed; shapes/finiteness checked by `tools.save`.
- Forward-pass checks of every constructed mechanism on all 13,307 recorded states (`work/check.py`,
  `work/check2.py`).
- Simulator (7 of 10 calls used; same 9 dev episodes, so comparisons are paired):

| call | file | dev mean | low | nominal | high | note |
|---|---|---|---|---|---|---|
| 1 | stay_b3 | 142.8 | 34.8 | 45.1 | 348.6 | lock-in at high demand |
| 2 | stay_b6 | 1106.6 | 735.5 | 1168.1 | 1416.2 | lock-in everywhere |
| 3 | start_weights | 44.51 | 34.51 | 44.61 | 54.42 | baseline |
| 4 | gated_b3 | 44.55 | 34.71 | 44.42 | 54.52 | neutral |
| 5 | gated_b6 | 44.64 | 34.67 | 45.08 | 54.18 | neutral |
| 6 | **ns_05 (delivered)** | **44.48** | 34.54 | 44.90 | 54.00 | 0 vehicles left, 244 switches/episode |
| 7 | ns_10 | 44.61 | 34.59 | 44.69 | 54.53 | neutral |

Paired per-episode differences of the delivered weights vs the baseline: -0.14, +0.10, +0.14, +0.92, +0.09,
-0.16, -1.41, -0.35, +0.50 veh-h (mean -0.03). This is within the dev noise: a handful of flipped decisions
moves a single episode by about +-1 veh-h in either direction, so 9 episodes cannot resolve effects below
roughly 0.3 veh-h. The edit is therefore justified by the training/test demand argument, not by a measured gain.

## What I expect to go wrong

- Most likely outcome: test TTS within +-0.3 veh-h of the original network (prediction: low 34.5, nominal 44.8,
  high 54.2, mean about 44.5). The improvement, if any, is small.
- The bias correction assumes the test demand is directionally unbiased (it is, per TASK.md) and that the net's
  sensitivity to observed queues is a fair proxy for its sensitivity to expected arrivals; if the net already
  internalised the bias differently, the +0.5 could cost up to a few tenths of a veh-h.
- No out-of-distribution risk: the hidden layers are unchanged and the shift is small (median logit gap 4.7).
