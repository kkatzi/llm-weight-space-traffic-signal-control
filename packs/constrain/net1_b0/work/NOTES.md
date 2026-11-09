# Supervisor for the PPO network under the operating rules

## What the data say (states.npz, 6 episodes, 4437 decisions; scripts `explore1-6.py`)

- The network switches at the first chance almost every time. At decisions where every phase is legal it keeps the
  current green only 2 % of the time. Without the rules, every green would end at 12 s, so its skill is choosing
  the **next phase**, not choosing durations.
- Where the rules block a request that the junction would otherwise honour (727 decisions):
  - 600 are **minimum-green holds**, through phases held from 12 s to 17 s;
  - 75 are **forced overdue switches**, 65 of them to phase 1 (NS left, the lowest-demand phase);
  - 51 have an overdue phase pending during a minimum green;
  - 1 is a **max-green** case.
- The 12 s to 17 s hold is almost pure waste. In all 645 held through greens, the held phase had **0 halting
  vehicles** at 12 s, about 0.7 vehicles on its stop-line lanes and about 7 vehicles spread over its four 256 m
  upstream lanes. That is most of the 2.5 veh-h rule loss. No supervisor can remove it: the hold is the rule.
- The network does not react oddly to the unusual t_green = 17 s. Setting that input to 12 s changes its argmax in
  only 0.8 % of those states. Its request at 12 s matches its choice at 17 s in 88 % of cases. So the decision at
  17 s needs no correcting.
- The network's chosen phase usually has a real queue: 4-12 halting for through phases, 3-6 for left phases. Only
  about 6 % of its through choices have 2 or fewer halting. A "skip weak through phases" rule therefore has little
  to act on.
- The top-1 minus top-2 logit gap at free decisions has a median of 4.5. It is at most 0.8 in 10 % of them.

## What the supervisor does (`supervisor.py`)

`decide(x, logits, state)` works only from the logits it receives and the rule state.

1. **Only one legal phase** (minimum green, or a single overdue phase): return it. The rules leave no choice.
2. **Otherwise**, take `c` = the legal phase with the highest logit. When the network's own choice is legal, `c`
   is that choice. When it is not (max green, or several overdue phases), `c` is the network's best legal phase,
   not the shield's "most halting" phase. If `c` is the current green (the network wants to extend and may),
   return it.
3. **Near-tie rules.** These apply only to other legal switch targets within `DELTA = 1.0` logits of `c`:
   - **b1, anticipate max-wait.** Pick a phase `p` that has halting vehicles and would reach 90 s before the next
     switch opportunity. That is `since[p] + horizon(c) >= 90`, where the horizon is 20 s if `c` is a through
     phase (3 s yellow + 17 s) and 15 s if `c` is a left phase. If the network picked `c`, `p` would be forced in
     at the next decision and pre-empt whatever the network wants then. Serving it now, while the network is
     nearly indifferent, is the cheaper slot. b1 does not fire if `c` itself is about to become overdue.
   - **b2, prefer the shorter commitment.** If `c` is a through phase and a left phase with at least 1 halting
     vehicle is within `DELTA`, pick the left phase. Under the rules a through green costs 20 s and its last 5 s
     are almost always empty. A left green still costs 15 s, as in training. In a true tie, serving the left phase
     first delays the through phase by 15 s, while the network's order delays the left phase by 20 s. Serving
     left phases sooner also keeps the 3-car pockets from spilling into the shared upstream lane. b2 does not fire
     if one left green of delay would make `c` overdue.
4. Any malformed input falls back to the network's argmax. The shield still enforces the rules in that case.

`DELTA = 1.0` was chosen by reasoning, not tuned. It means the network gives the alternative at least about
e^-1 (37 %) of the odds of its top choice, so it is not confident. That is well below the median gap of 4.5.

### When it overrides the network

On the recorded trajectory:

| case | share of decisions |
|---|---|
| forced: only one legal phase (almost all minimum-green holds) | 64.9 % |
| same as the network | 34.0 % |
| b2: through to left near-tie | 1.0 % |
| b1: soon-overdue near-tie | 0.1 % |
| network's choice illegal, best legal by logits | 1 decision |

- It differs from the network's argmax 66 % of the time overall. Nearly all of that is rule-forced, and any
  rule-respecting controller has it.
- At decisions with more than one legal phase, it differs 4.4 % of the time.
- Its executed choice differs from the naive shield's at **48 / 4437 decisions (1.1 %)**.

These counts come from the recorded trajectory. Once trajectories diverge they will shift a little.

### Larger interventions I considered and rejected

- **Penalising through phases in general**, to serve them less often with fuller greens. Upstream lane 1 is shared
  by straight and left traffic. Longer through queues would block left-turners from reaching the pocket, and that
  coupling cannot be judged without a simulator. Only the near-tie version (b2) is kept.
- **Serving soon-overdue phases early regardless of the logits.** Deferring a forced service to the latest legal
  slot keeps its frequency lowest and lets more vehicles build up per green. The shield already does that. Early
  service is used only when the network is nearly indifferent (b1).
- **Extending greens.** At every hold the held phase had no halting vehicles, so extending would only add waste.
  The network itself almost never extends.

## Checks run (`check_supervisor.py`, recorded states only; no simulator, no search)

- On all 4437 recorded decisions, every output is a legal phase.
- Wherever the network's choice is legal, the supervisor agrees with it except at the 48 near-tie overrides. Each
  of those was listed and inspected by hand. They look like true ties: for example logits 4.32 vs 4.28, or
  2.20 vs 2.07.
- Results are identical with list or numpy state entries and with float32 or float64 logits.
- With `state` empty or `None`, or with `rules` missing, it returns the network's argmax or uses the default rule
  values.
- Hand-made edge cases all behave as intended:
  - max green: best legal phase by logits;
  - two overdue phases: the network's preference between them;
  - minimum-green hold;
  - a b2 tie;
  - a b2 tie blocked because the through phase would become overdue;
  - a b1 tie.
- The rule details used were confirmed in the data: overdue fires at since_served = 90 (all 147 onsets), through
  greens end at 17 s and left greens at 12 s, and an overdue phase makes the current green illegal once its
  minimum green has passed.

## What I expect to go wrong

- The supervisor is essentially the naive controller plus about 7-8 near-tie flips per episode. I expect a small
  gain: roughly 0.1-0.3 veh-h, larger at high demand where the left pockets overflow more. Neither rule's effect
  on TTS could be verified, since there was no simulator. Either could be neutral or slightly harmful. The
  plausible range is about -0.3 to +0.2 veh-h relative to naive.
- The 15 s through minimum green remains the main loss, and the supervisor cannot recover it. Test TTS should
  stay close to the naive 47.0 rather than near the rule-free 44.5.
- If the test harness's `state` uses different key names or types, the supervisor falls back to the network's
  argmax plus the shield, which is exactly the naive baseline.
- If `since_served` were measured from a different reference point (for example the end of yellow), the b1
  horizon would be off by a few seconds. b1 would fire slightly too early or too late, with negligible effect.
- Test seeds differ from dev, and the dev means rest on only 3 episodes per scenario. Per-scenario predictions
  carry roughly ±1 veh-h of sampling uncertainty.

## Files

- `supervisor.py`: the deliverable.
- `check_supervisor.py`: the checks above.
- `explore1.py` - `explore6.py`: statistics of the recorded data and network-sensitivity checks. The network was
  only run for analysis; no fitting was done.
- `predictions.json`: predicted mean test TTS per scenario.
