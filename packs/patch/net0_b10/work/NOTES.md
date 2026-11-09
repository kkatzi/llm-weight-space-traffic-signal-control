# Edit notes: "empty-phase gates" on the PPO signal network

## What I found

- In normal traffic the trained network is already a good gap-out controller. At almost every decision where a
  switch would be honoured (ok = 1) it switches right after the minimum green. At that moment the current phase
  never has a halted vehicle left at its stop line (0 of 1842 recorded ok-states), and the next phase it picks has
  real queues: only 12 of 1458 switches went to a weak phase while a long queue waited. I found nothing in its
  normal timing or phase order that I could improve with confidence.
- **The defect: it over-prefers phase 2 (E-W through) even when nothing is on its lanes.** In recorded
  start-network episode 2 the network drained to 3 waiting vehicles (N0, N1 and the W2 left pocket) while phase 2
  was green with empty E-W lanes. It then held phase 2 from t = 3625 s until the 30-min drain cap (logits
  `[-1.7, -4.0, 2.1, 2.0]`: phase 2 beats phase 3 by 0.1). That costs about 3 vehicles x 0.5 h = 1.5 veh-h in
  that one episode.
- Probes (`probe_deadlock.py`, `probe2.py`) on synthetic sparse drain-time states found 2960 of 36,711 states
  where the original picks a phase with nothing to serve while another phase has vehicles. 2946 of these pick
  phase 2 (753 of them holding it). On recorded network states the same thing happens 355 times, all in that
  lock-up.

## What I built

A rule written into the network: **a phase with no vehicle on any lane it serves gets its logit lowered by 20.**
When some phases have vehicles, an empty phase can then never be chosen or held. When all four phases are empty,
all four logits drop by the same amount, so the original choice stands.

"Lanes a phase serves" means its stop-line lanes (all vehicles, density `d`) plus the upstream lanes that feed
them. On upstream lanes only **moving** vehicles count (`d - q`). A halted upstream vehicle is always blocked behind
a stop-line lane that already holds a vehicle, so that phase's demand is visible there anyway. Not counting the
halted ones stops a left-turner, blocked in a shared upstream lane behind a full pocket, from keeping the through
phase "busy".

| phase | stop-line lanes (`d`) | upstream lanes (`d - q`) |
|---|---|---|
| 0 NS through | N0 N1 S0 S1 | N0u N1u S0u S1u |
| 1 NS left | N2 S2 | N1u S1u |
| 2 EW through | E0 E1 W0 W1 | E0u E1u W0u W1u |
| 3 EW left | E2 W2 | E1u W1u |

### Units

| phase | layer-1 gate unit | layer-2 relay unit | penalty |
|---|---|---|---|
| 0 | h1[32] | h2[37] (`W2[37,32] = -20`) | `W3[:,37] = [-10,0,0,0]` |
| 1 | h1[60] | h2[0]  (`W2[0,60] = -20`)  | `W3[:,0]  = [0,-10,0,0]` |
| 2 | h1[57] | h2[5]  (`W2[5,57] = -20`)  | `W3[:,5]  = [0,0,-10,0]` |
| 3 | h1[59] | h2[49] (`W2[49,59] = +20`) | `W3[:,49] = [0,0,0,-10]` |

- **Gate (layer 1).** Each row is set to zero except the lane inputs above: stop-line `d` x 36 (one car = 12),
  upstream `d` x 410 and `q` x -410 (one moving car = 12), with bias +-4. The unit sits at its "normal" value
  (+1 for 32, 57, 60; -1 for 59) whenever any vehicle is present, and flips to the opposite sign (|tanh(4)| =
  0.9993) when the phase is empty.
- **Relay (layer 2).** The relay gets a weight of +-20 from its gate. All the relay's other inputs together are
  bounded by |pre-activation| <= 13.7, so the relay is pinned at -1 normally and +1 when the gate fires.
- **Penalty (layer 3).** The relay's W3 column is replaced by -10 on its own phase's logit. In normal states every
  relay gives +10 to its own logit, which is the same on all four logits, so the ranking is unchanged. A fired
  relay moves its logit by -20, more than the largest logit spread seen on recorded states (14.8).
- **b3.** The four relays' old average contribution (`W3_old[:, j] * mean(h2_j)`) is added to b3.
- I chose the sacrificed units (`analyze_units.py`, `choose_units.py`) as those whose removal changes the fewest
  recorded decisions. This was a choice among units, not a fit of any value.

**Parameters changed: 212 of 7428** (W1 4 rows = 184, b1 4, W2 4, W3 16, b3 4; b2 untouched).

All values are set by construction. No optimiser, regression or search was used.

## Checks

- `check.py` (recorded states, ok = 1 only):
  - the empty-phase choices fall from 369 to 0;
  - agreement with the original is 99.1 % on fixed-time states and 98.6 % on actuated states;
  - outside the empty-phase rule, 37 decisions changed, a side effect of the 8 sacrificed units.
- `probe_deadlock.py`: synthetic "current phase empty, 1-3 lanes waiting elsewhere" states. The original holds
  the empty phase in 193 of 6408; the edit holds it in 0.
- `probe2.py`: 40,000 random sparse states with stop-line, upstream, halted and moving vehicles. Empty-phase
  choices fall from 2960 to 0.
- Recorded lock-up (episode 2, t >= 3625 s): the original picks 2, 2, 2, ...; the edit picks 3, which serves
  W2. N0/N1 then get served by phase 0.
- The saved `weights.npz` is byte-identical to the file evaluated in call 3. `tools.edit_size` gives 212.

### Simulator calls (3 of 10 used)

| call | weights | low | nominal | high | mean | left at cap |
|---|---|---|---|---|---|---|
| 1 | original (`start_copy.npz`) | 34.7 | 44.9 | 54.4 | 44.7 | 0 |
| 2 | gates v2 (all upstream vehicles count) | 34.8 | 44.6 | 54.0 | 44.5 | 0 |
| 3 | gates v3 = final (moving upstream only) | 34.8 | 44.6 | 54.0 | 44.5 | 0 |

- Per episode, original -> final: 35.4->35.7, 33.5->33.4, 35.1->35.1, 46.0->45.5, 44.6->44.2, 44.2->44.2,
  55.5->55.5, 53.4->53.0, 54.4->53.4.
- None of the 9 dev seeds triggers the lock-up, so the 0.2 veh-h gain there is mostly noise from the few changed
  decisions.
- The real benefit is in episodes where the original locks up: about 0.5 veh-h per stranded vehicle. The
  recorded episodes show that in 1 of 6.
- I stopped after 3 calls on purpose. More variants would only be fitted to dev noise.

## What could go wrong

- **No lock-up in the test seeds.** If no test episode triggers it, the edit gains about nothing (around 0.1-0.3
  veh-h of noise either way from the ~0.7 % of borderline decisions the sacrificed units changed).
- **Rare demand patterns.** These sacrificed units (h1 32, 57, 59, 60; h2 0, 5, 37, 49) could have mattered in
  demand patterns the recorded states don't cover.
- **Empty network.** All logits drop equally, so the original choice stands. Partly empty states differ: say only
  one far, moving upstream car on phase 2 and nothing elsewhere. Then the other phases are penalised and phase 2 is
  held. That is intended, but it is new behaviour.
- **What the gate cannot see.** It only sees lane occupancy. A vehicle stopped inside the junction, or a
  queue-entry backlog, is invisible to it, though neither should occur at these demand levels.
- **Trainability is worse.** The gate units saturate by design (sat_l1 rises from 0 % to 6 %) and logits reach
  about 19. The task scores TTS only.

## Predictions (test, mean TTS veh-h)

`{"low": 34.8, "nominal": 44.7, "high": 54.2}`. These are the edited network's dev values, pulled slightly
towards the original's dev values, because the dev gap is mostly noise and the test seeds differ.
