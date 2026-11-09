# NOTES: net1_haiku patch attempt

## Result
No edit I built beat the start network on the dev episodes. The delivered `weights.npz` is therefore the
**unchanged start network**: 0 of 7428 parameters differ (checked with `tools.edit_size`).

## What I tried
Idea: the trained net switches phase almost every time the 10 s minimum green allows it (97% of
honoured decisions in the recorded states, `states.npz`, controller 0). Each switch costs 3 s of yellow, so
an actuated-style "hold green while the current approach still has demand" rule might cut lost time.

1. **Unconditional stay bias** (`build_stay.py`): add beta * s_p to row p of W3, where s_p is a correlation-
   weighted direction in layer-2 space that tracks "current phase = p" (67 weights changed for beta=1).
   Sim call 1 (beta=1): low 36.0 / nominal 46.3 / high 695.3, mean 259.2 veh-h. High demand gridlocked
   (1144, 786, 574 vehicles left at the drain cap in three episodes). Rejected. The bias has no demand gate,
   so the net holds a green while the other approaches starve.
2. **Demand-gated stay bias** (`build_stay2.py`): the same construction, but the direction is chosen by
   correlation with (current-phase indicator x mean density of that phase's green lanes). 31 weights changed,
   beta=1. Offline, switch rate on honoured states falls from 0.97 to 0.95, and from 0.94 to 0.90 when the
   current green is busy, so the gate is weak. Sim call 2: low 34.5 / nominal 44.9 / high 54.6, mean 44.7 veh-h,
   against 44.5 for the start network. Not an improvement (within noise, slightly worse). Rejected.

Simulator calls used: 2 of 10.

## Units and weights
- W3 (4x64) rows = logits of the four phases; b3 = logit offsets.
- Layer-2 units were used only through correlations with the current phase and with demand. No unit was
  identified by hand as a clean "queue of phase p" detector. Dead units: none (layer 1 and 2 both).

## Checks run
- `analyze*.py`: switch rates, net vs max-queue rule (about 70% agreement on honoured states), correlations.
- `tools.trainability` on the start network: median max-prob 0.989, 49% of states above 0.99, saturation low.
- Sim check of both candidates (2 calls).

## What I expect to go wrong
- Any hand edit that changes the hysteresis of the switching policy can starve a phase under high demand. The
  first candidate showed this. Hysteresis needs a real demand gate, which needs new hidden units (about 110
  weights per feature), and I did not have the budget or the evidence to build that.
- The delivered network equals the start network, so its test TTS is the start network's: about 34.5 / 44.6 / 54.4.
