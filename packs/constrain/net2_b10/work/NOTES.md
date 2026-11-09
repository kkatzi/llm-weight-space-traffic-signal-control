# Supervisor for net2 under the operating rules: notes

## What the supervisor does and why

`work/supervisor.py` is a thin layer: **the network stays in charge, and the rules are applied at the latest
legal moment.**

1. **The rules leave one legal phase.** This happens while a minimum green is running, or when a single overdue phase
   must come next. The outcome is fixed whatever is asked, so the supervisor passes the network's request through and
   the shield applies the rule. The junction then behaves exactly as if the legal phase had been returned.
2. **Several phases are legal and the network's choice is one of them.** Keep the network's choice. This is the
   normal case.
3. **Several phases are legal and the network's choice is not one of them.** Return the legal phase the network ranks
   highest, instead of the shield's "most halting vehicles". This happens with max green, or with two phases overdue
   at once. On dev it fired a handful of times in 2 of 9 episodes, with no measurable effect.

Why so little intervention: I analysed the recorded run (`analyze1-10.py`) and tried three ways of anticipating the
rules in the simulator. None beat the network plus the shield (see the checks below).

### How the network behaves and what the rules cost (recorded states, 6 episodes)
- The network is a **"minimum green, then pick the next phase" controller.** At 99 % of its effective decisions it
  asks to switch at the first chance: at 12 s for every phase when there are no rules. Under the rules, through
  greens are held to 17 s (pedestrian rule) and left greens still end at 12 s.
- **Max green (45 s) never binds**, because the network only holds a green when nothing else is waiting.
- **Max wait (90 s) binds about 13 times per episode.** About 70 % of those are the NS-left phase, which the network
  tends to neglect. The forced phase usually has 2–6 halting vehicles.
- The network's next-phase choices are decisive (median top-1/top-2 logit gap 5.6, roughly 2–3 vehicles' worth).
  They pick through phases with about 8 halting vehicles and left phases with about 4. Upstream queues are nearly
  always empty.
- So the rule cost (47.0 vs 44.8 veh-h) is mostly structural: about 110 through greens per episode run 5 s longer,
  plus forced NS-left service. The shield already applies both at the latest legal moment.

## When it overrides the network
- Only in case 3 above: the network's choice is illegal while more than one phase is legal. On the recorded states
  this never happens: the choice differs from the network's argmax in 0 of 4433 states. On dev it changed a handful
  of decisions in 2 of 9 episodes (switch counts +1 in those episodes).
- Every other rule-driven replacement is left to the shield. These are the through-phase requests at 12 s (about 108
  per episode) and the forced overdue phases (about 13 per episode). The eval reports them as shield overrides,
  about 120 per episode, the same as naive.

## Checks run
- Statistics of `states.npz` (no fitting): legal-mask patterns, illegal requests by cause, green durations, overdue
  events and what the network wanted at each, logit gaps, logit sensitivity to one extra vehicle per lane, queue and
  spill-back picture, and the network's behaviour after forced insertions.
- `check_supervisor.py`: first-order offline check of each variant on the recorded states (changed decisions,
  legality).
- **Simulator (5 of 10 calls).** The dev runs are deterministic: pass-through reproduced the naive reference
  46.978 exactly. Differences are paired against it, per episode; the sd is over the 9 episodes.

| call | variant | dev mean | low / nom / high | paired Δ (mean, sd) |
|---|---|---|---|---|
| 1 | penalty 2.0 logits on starting a through phase (Smith's rule: through now takes 20 s, not 15 s) | 46.955 | 36.19 / 47.15 / 57.53 | −0.02 (0.45) |
| 2 | penalty 4.0, dropped when the through phase spills upstream | 47.103 | 36.10 / 47.02 / 58.19 | +0.13 (0.24) |
| 3 | anticipate max wait: serve a phase that would become overdue during the chosen green first | 47.399 | 36.23 / 47.20 / 58.77 | +0.42 (1.04); one high episode +3.0 |
| 4 | pass-through (network + shield) | 46.978 | 36.07 / 47.01 / 57.85 | 0 |
| 5 | **final** `supervisor.py` | **46.979** | **36.07 / 47.01 / 57.85** | +0.001 |

What the calls showed:
- **Through penalty (calls 1–2).** It changed 1.4 % and 10 % of the network's free decisions and did not help.
  The queueing argument assumes the 5 extra seconds of a through green are wasted. They are not entirely: they serve
  arrivals, and through traffic is 70 % of demand.
- **Early service of near-overdue phases (call 3).** It is harmful. It serves a phase before the rule forces it, at
  the expense of the phase the network rates more urgent. Under high demand it broke the EW left/through alternation
  and caused a spill-back episode (+3 veh-h). The shield's "serve exactly when overdue" is better.
- Paired per-episode noise is about 0.25–0.45 veh-h, so only effects of roughly 0.3 or more on the 9-episode mean
  are detectable. I adopted no variant whose gain was within that noise.

## What I expect to go wrong / caveats
- **No real gain is expected over the naive shield deployment.** The final supervisor is essentially the network plus
  the shield, so I predict test TTS close to the naive dev values: low 36.1, nominal 47.0, high 57.9. The dev means
  rest on 3 episodes per scenario; between-episode sd is about 1.2 (low) and 1.7 (high), so each scenario prediction
  is uncertain by about ±0.7–1.0 veh-h.
- **High-demand tail risk.** Under the rules, the high scenario is close to capacity: a 70 s minimum cycle with
  3-car left pockets. A test episode can break down with spill-back (+2–3 veh-h), as one dev episode did under
  variant 3. The naive network is exposed to this too.
- **The case-3 fallback is barely exercised.** If test episodes produce many max-green or double-overdue states, the
  network ranking replaces the shield's most-halting choice. I could not measure which is better.
- I did not try green extensions. The network never wants them, and nothing suggested they would help.

## Files
- `supervisor.py`: the final supervisor.
- `sup_v0_pass.py`, `sup_v1_pen2.py`, `sup_v2_pen4.py`, `sup_v3_antic.py`: the tested variants. `sup_v1_pen2.py`
  and `sup_v2_pen4.py` are copies of `supervisor.py` as evaluated in calls 1 and 2.
- `analyze1.py` … `analyze10.py`: statistics of the recorded run. `check_supervisor.py`: offline check.
- `predictions.json`: predicted test TTS per scenario.
