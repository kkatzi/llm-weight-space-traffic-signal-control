# Patch notes: net1, budget 371 parameters

Delivered: `work/weights.npz`, a copy of `wG.npz`. It changes **112 of 7428** parameters
(W1 24, b1 4, W2 4, b2 60, W3 16, b3 4). It is built by `build_gate.py` (`build(W0)` with defaults).

## What I built and why

**What the original network does.** It switches almost as soon as the 10 s minimum green allows: about 95% of honoured
decisions (ok = 1) switch, and greens last about 12 s. The phase order is adaptive. Queues are short, and the served
phase is cleared within one 12 s green (0 halting vehicles left on its stop-line lanes at the 90th percentile). So
the controller's only real job is to choose the next phase. I found no sign that longer greens would help. The tuned
fixed-time and actuated controllers point the same way: the actuated one never extends.

**The defect.** In near-empty states the network prefers to hold the current green, and its response to a single
waiting car is too weak to overcome that preference. Example: empty junction, phase 2 green, logits
[-0.3, -4.3, 3.9, -0.7]. One waiting car adds only about +1 to its phase's logit. In synthetic states with one to
three cars on a single lane whose phase is red, the original network keeps the wrong green in **296** combinations
of lane, phase, green age and halting/moving. The car then waits until another vehicle shows up. During the final
drain no new vehicles arrive, so the car can wait until the 30-minute cap. That costs up to about 0.5 veh-h per car.
Simulator call 1 (a different edit with the same weakness) left exactly one car at the drain cap. The recorded drains
also show the original holding phase 3 for 10 s while the only car waited on phase 0 (episode 3, t = 3665–3670).

**The fix: an "empty-phase guard".** Never pick a phase that has no vehicle on any of its lanes while some other phase
has one. Each phase p gets a gate:

- `gate_p` = +1 when phase p's lanes (its stop-line lanes plus the upstream lanes that feed them) hold no vehicle,
  otherwise -1.
- When `gate_p` = +1, logit p drops by 30, more than the largest logit spread in the data. When it is -1 the logits
  are unchanged.

If every phase is empty, all four are penalised equally and the original preference stands. When at least one phase
has a vehicle, the choice among the non-empty phases is left to the original network.

## Which units and weights do what

| phase | lanes counted as "phase p has a vehicle" (density inputs) | layer-1 gate unit | layer-2 relay unit |
|---|---|---|---|
| 0 NS_through | d_N0, d_N1, d_S0, d_S1, d_N0u, d_N1u, d_S0u, d_S1u | 31 | 25 |
| 1 NS_left | d_N2, d_S2, d_N1u, d_S1u | 0 | 35 |
| 2 EW_through | d_E0, d_E1, d_W0, d_W1, d_E0u, d_E1u, d_W0u, d_W1u | 59 | 39 |
| 3 EW_left | d_E2, d_W2, d_E1u, d_W1u | 54 | 15 |

- **Layer 1 (W1 24, b1 4).** Each gate unit gets weight -K·capacity on its phase's density inputs (capacity is 2.99
  for a stop-line lane and 34.1 for an upstream lane), so each input counts vehicles. Its bias is B. The unit's other
  old input weights stay, and their worst case over x in [0,1] is bounded by B = bound + 4 and K = 2·bound + 8 per
  vehicle. Its pre-activation is therefore ≥ +4 when the phase is empty and ≤ -4 when it has a vehicle, so the unit
  sits saturated at ±1 (checked: |h| > 0.99 on all recorded states). The four host units are the least important
  ones: replacing any one by a constant changes 0.2–0.7% of decisions.
- **Layer 2 (W2 4).** W2[relay, gate] = 40, so each relay copies its gate's sign (saturated).
- **b2 (60).** Every other layer-2 unit gets the mean change of the four replaced layer-1 units' outputs, so its average
  input stays the same.
- **W3/b3 (16 + 4).** Each relay's old output column becomes (0, …, -15 at phase p, …, 0). b3 absorbs the relays' old
  mean contribution and the constant +15 that the relay adds when it is -1 (inactive).

## Checks I ran

- Edit size 112 (`tools.edit_size`), within the 371 limit. Shapes and finiteness pass `tools.save`.
- Synthetic single-lane states (`deadlock.py`): 296 stuck cases for the original, **0** for the delivered weights.
- Recorded states (`test_gate.py`, `fixes.py`): the guard removes every "pick an empty phase while another phase has
  cars" decision (4, 4 and 5 such decisions in the network, fixed-time and actuated logs, all at the episode start or
  in the drain). As a side effect of reusing four layer-1 and four layer-2 units, about 1.5% of other honoured
  decisions change (2.0% on the network's own states, 0.9–2.3% overall). Those changes have no consistent direction.
  Overall agreement with the original on all recorded states is 97.7%.
- Simulator, 4 of 5 calls used (dev mean TTS; the original scores 44.51, with low/nominal/high 34.5/44.6/54.4):

| call | weights | params | mean | low | nominal | high | cars left at cap |
|---|---|---|---|---|---|---|---|
| 1 | wA: symmetry repair only | 192 | 44.49 | 34.37 | 44.47 | 54.64 | 1 |
| 2 | **wG: empty-phase guard (delivered)** | 112 | **44.26** | 34.41 | 44.53 | 53.85 | 0 |
| 3 | wAG: repair + guard | 303 | 44.32 | 34.53 | 44.66 | 53.76 | 0 |
| 4 | wZ: guard with the gates' other outgoing weights cut | 352 | 44.32 | 34.43 | 44.41 | 54.12 | 0 |

  All three versions with the guard sit at 44.26–44.32. Both versions without it sit at 44.49–44.51. The gain is
  mostly in the high scenario.

- I also tried a symmetry repair (`repairA.py`). The junction and the test demand are symmetric under a 180° rotation,
  yet a car waiting on N1 pushes the network toward EW_through and a car waiting on W0 toward NS_through. The repair
  copies the W1 columns of S1 and E0 into those of N1 and W0. It makes the lane responses sensible but showed no
  simulator gain (calls 1 and 3), so I left it out.

## What I expect to go wrong

- The gain is small: about 0.25 veh-h on dev, about 0.5%. Per-episode chaos in SUMO is about ±0.5 veh-h, so on the 30
  test episodes the improvement could be smaller than on dev, mostly in low/nominal. The guard matters most in test
  episodes where the original would have stranded a car in the drain. There it saves up to about 0.5 veh-h per car.
- The roughly 1.5% side-effect changes to normal decisions are not targeted, and they could cost a little.
- The guard counts upstream vehicles, so a phase with only one car far upstream is "non-empty" and may keep the green
  for about 18 s while that car arrives. This is bounded, but not optimal when another phase has halting cars.
- Large gate weights (up to about 850 on upstream density inputs) make the network brittle for further PPO training:
  `tools.trainability` shows max |logit| of 40 and more saturated units. This does not affect the test as delivered.

## Files

- `common.py`: shared loading and lane-count helpers.
- `analyze*.py`, `residual.py`, `lanecost.py`, `qstats.py`, `upq.py`, `drain.py`, `stays.py`, `smallpick.py`:
  behaviour analysis.
- `mirror.py`, `fd.py`, `repairA.py`, `changesA.py`: the symmetry diagnosis and repair (not delivered).
- `deadlock.py`: the synthetic stuck-car test.
- `importance.py`, `choose_units.py`: choosing the host units by ablation impact.
- `build_gate.py`, `test_gate.py`, `side.py`, `fixes.py`, `variant_zero.py`: building and checking the guard.
- `wA.npz`, `wG.npz`, `wAG.npz`, `wZ.npz`: the evaluated candidates. `weights.npz` = `wG.npz`.
