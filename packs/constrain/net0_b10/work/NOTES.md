# Supervisor for the PPO network under the operating rules

Final file: `work/supervisor.py` (same decisions as `work/sup_v3_pen4_k4.py`; evaluations 4 and 5 are identical).
Dev TTS: **46.86 veh-h** (low 36.04, nominal 47.01, high 57.55). The naive shield gets 46.92 (35.86 / 47.05 / 57.85).
The difference is within noise (see "Checks").

## What the network does under the rules (from `states.npz`, 6 episodes, 4451 decisions)

- The network is a pure phase picker. It asks to switch at every chance: it never holds a left green at 12 s, and
  holds a through green at 17 s only 1.5 % of the time. Every green is a minimum green.
- Legal sets are almost always either a single phase (minimum green running, or an overdue phase) or all four
  phases. When the network's request is illegal there is no real choice. The shield's fallback ("most halting") and
  any other fallback give the same phase. So a supervisor that only "respects" the rules equals the naive shield
  (confirmed: evaluation 2 reproduced the naive reference exactly, episode by episode).
- **Rule 1 (minimum green)** is the main cost. 645 of 659 through greens were held from 12 s to 17 s against the
  network's request. At 12 s the through stop-line lanes are already empty (median 0 vehicles, ~1 moving), even at
  the highest loads. So ~5 s per through green (~105 per episode) idle while every other phase waits.
  Left greens are unaffected (they could already end at 12 s).
- **Rule 3 (max wait)** forces ~12 services per episode against the network's wish. Most are NS-left (phase 1) with
  1–2 halting vehicles. They come at the deadline, i.e. as late as possible, which is already the cheapest timing for
  the served phase.
- **Rule 2 (max green 45 s)** never binds (longest green seen: 37 s).
- The network ignores the `t_green` input (setting it from 17 to 12 s changes 8 of 659 decisions), so its
  17 s decisions are not distorted by an unfamiliar input.

## What the supervisor does and why

1. **Respect.** It always returns a legal phase. With one legal phase it returns that phase. Otherwise it starts
   from the network's best legal phase by logits, not the shield's most-halting rule. This only matters in rare
   states (max green with several legal phases, two overdue phases).
2. **Anticipate the minimum green.** Starting a through phase now costs 20 s of junction time instead of the 15 s
   the network was trained with, and the extra 5 s mostly idle. Meanwhile the 3-car left pockets keep filling and
   spill into the shared straight+left upstream lane. So when the network wants to **start** a through phase,
   the supervisor serves a left phase first, if all of these hold:
   - the left phase is legal and is not the current green;
   - it has **≥ 4 halting vehicles** (~2 per pocket: it will use its 12 s, and its pockets are near overflow);
   - the network ranks it within **4 logits** of the through phase;
   - the through phase is not overdue.

   Otherwise the network's choice stands.

## When it overrides the network

- Wherever the rules force a phase (the minimum-green holds and overdue services, ~67 % of all decisions,
  identical to what the shield does).
- At real decisions (more than one legal phase) it changes the network's choice in about 7 % of cases (78/1130 in
  the recorded states), always from a through start to a left phase. Most of these are at higher loads, because the
  ≥ 4 halting condition rarely holds at low demand.

## Checks run

- Offline (`work/explore*.py`, `work/check_offline.py`, `work/compare_variants.py`): legality of every output on the
  recorded states, number and kind of changed decisions, queue sizes at switch decisions and at 12/17 s,
  sensitivity of the logits to queue size and to `t_green`, overdue statistics, pocket spill-back frequency. No
  fitting, no searches.
- Simulator, 5 of 10 calls. Paired per-episode differences are against evaluation 2:

  | call | variant | mean TTS | low / nominal / high | mean paired diff |
  |---|---|---|---|---|
  | 1 | penalty 2 on every through start | 46.91 | 36.12 / 46.84 / 57.76 | −0.01 |
  | 2 | respect only (equals naive) | 46.92 | 35.86 / 47.05 / 57.85 | 0 |
  | 3 | penalty 4 on every through start | 46.79 | 36.17 / 47.16 / 57.06 | −0.13 |
  | 4 | penalty 4, only if the left phase has ≥ 4 halting (final) | 46.86 | 36.04 / 47.01 / 57.55 | −0.06 |
  | 5 | final file, confirmation (identical to 4) | 46.86 | same | −0.06 |

  Per-episode paired differences have SD ≈ 0.8–1.1 veh-h, so a 9-episode mean has SE ≈ 0.3. None of the variants
  is distinguishable from the naive shield. Calls 1 and 3 were slightly worse at low demand in both runs. That fits
  a queue/service-time (cμ) calculation: at near-ties the through phase usually has about twice the left's queue,
  so moving it later does not pay when queues are short. Hence the final variant restricts swaps to left phases
  with a real queue. I stopped at 5 calls: the paired noise is larger than any effect this family can have, so more
  calls would only have meant picking the luckiest variant.
- Also considered and rejected by reasoning or data:
  - Serving a nearly overdue phase early at near-ties. This means serving it more often; the forced service already
    comes as late as possible.
  - Serving the left first because of spill-back. The network already prefers the through phase strongly
    (gap 5–6) even then.
  - Holding productive greens. Through lanes are empty at 12–17 s, and the hold/switch trade-off is the same as in
    training.
  - Using the logit gap of the forced hold as a value estimate. The logits are near-deterministic (hold gap ≈ 8 at
    every stage), not calibrated to value.

## What I expect to go wrong

- The supervisor is essentially the network plus the shield. The rules' cost (+2.2 veh-h dev) is structural:
  5 idle seconds per through green, plus forced short services. Choosing which phase goes next barely touches it.
  Expect test TTS about equal to the naive shield's, perhaps 0–0.3 veh-h better. It may be slightly worse at low
  demand, where any reordering away from the network tends to cost a little.
- The test differences between this supervisor and naive will be dominated by chaotic trajectory divergence
  (±1 veh-h per episode), not by the policy change.
- Predictions (`predictions.json`): low 36.0, nominal 47.0, high 57.6 veh-h. High demand has the largest seed
  spread (dev episodes 55.4–59.1), so it is the least certain.
