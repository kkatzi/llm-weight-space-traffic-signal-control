# Supervisor for net0 under the operating rules

## What the recorded run shows (work/analyze1-5.py; counting only, no fitting)

In 4451 recorded decisions (network under the rules, shield only):

- **The network switches at the first chance it gets.** Every left green ends at 12 s. Through greens end at
  17 s in 649 of 659 cases. In 645 decisions the network asked to leave a through phase at 12 s, and the 15 s
  pedestrian minimum held it. Requests at green ages 2 and 7 s are ignored by the junction, so 2924 of the 2995
  "illegal" requests change nothing.
- **Max green never binds.** The longest green was 37 s. The network stays put in only 21 of 1130 free
  decisions.
- **Max wait binds about 12 times per episode** (71 forced decisions). In 53 of them the forced phase is 1
  (NS left), which the network tends to starve. At the forced decision, the network ranks the forced phase 2nd
  in 44 cases.
- **The legal set is always either one phase or all four.** When all four are legal, the network's pick is
  always legal. So the shield's "most halting" fallback is never used for a real choice: the shield either
  passes the request through or has only one option.

The 2.2 veh-h cost of the rules therefore comes mostly from through greens growing from 12 s to 17 s, plus the
forced service of starved phases. A supervisor can't touch the first: the minimum green is absolute and it
binds right after any through phase is chosen. On the second, the forced timing is close to the least
disruptive one available. The pick c usually has a longer queue than the phase p that is forced after it (99
vs 20 cases out of 138). By Smith's rule (load / time occupied), the forced order c then p is usually already
the better order. Serving p earlier would also raise how often p is served.

## What decide() does

1. **Keeps the network's argmax whenever it is legal**, apart from the narrow rule in 3.
2. **If the argmax is illegal:**
   - With exactly one legal phase (minimum green running, or a single overdue phase), it returns the argmax
     unchanged. The shield turns any request into that single phase, so the outcome is the same, and no
     override gets recorded for a decision the rules have already fixed.
   - With several legal phases (max green reached, or several phases overdue at once), it returns the legal
     phase with the highest logit, with halting vehicles as the tie-break. This keeps the network's own
     preference rather than the shield's "most halting".
3. **Anticipates max wait at a free switch decision.** Say the network picks c ≠ green, and another legal
   phase p has at least 2 halting vehicles and `since_served[p] + T(c) >= 90`. Here T = 3 s yellow + the age
   of the first decision at which the new green may end, so T = 20 s for through and 15 s for left. Then p
   will be forced right after c anyway, and the only real choice is the order. The supervisor serves p first
   only when `halting(p)/T(p) > 1.5 * load(c)/T(c)`. load(c) counts the vehicles on c's stop-line lanes, moving
   or halting, never fewer than its halting count. The 1.5 margin covers what these counts miss: queues
   capped at 3 per lane, upstream platoons, and the network's own knowledge of c.

The rule parameters default to the TASK.md values (15/10 s minimum, 45 s maximum, 90 s wait). They are taken
from `state["rules"]` only when it uses one of a few guessed key names. Only numpy and math are used, and the
network is used only through `logits`.

## When it overrides the network

- In the recorded states: **never**. The output equals the argmax in all 4451 decisions, and the applied phase
  equals the naive shield's in all 4451. Rule 3 fired once in an earlier version, on a borderline case where
  clipping counted 3 pocket vehicles as 2.99; whole-vehicle counting removed it.
- At test, it overrides only in situations this network did not produce in the recorded states:
  - several legal phases with an illegal argmax (max green or several overdue phases);
  - a phase about to become overdue that has a clearly larger queue than the network's pick.

## Checks (work/check_supervisor.py)

- **Replay on all 4451 recorded states**, with logits from the network and the state rebuilt from rule_state:
  - outputs are always in 0-3;
  - 0 differences from the argmax;
  - 0 differences in the applied phase compared with the naive shield;
  - illegal outputs occur only where exactly one phase is legal.
- **Synthetic edge cases, all passing:**
  - a legal pick is kept;
  - minimum green;
  - max green (best-logit legal phase is chosen);
  - two overdue phases (best logit among them);
  - one overdue phase;
  - the anticipation rule fires and doesn't fire, including the "not overdue in time" case;
  - lists instead of arrays, with no `rules` key.
- **Limit:** the replay is open loop, on states produced by the naive controller. Since the supervisor
  matches it on every one of them, closed-loop runs should follow the same trajectories until an unseen
  situation comes up.

## What I expect to go wrong

- **It recovers essentially nothing.** Expected test TTS is about the naive level: dev 35.9 / 47.1 / 57.9.
  That's still better than tuned actuated and fixed-time, but well above the network without the rules (44.7
  on dev).
- **The structural loss stays.** I didn't find a way to cut the cost of 17 s through greens without a model to
  check it against. Tilting choices toward left phases, or batching through traffic, could help or hurt, so I
  left them out.
- **The best-logit fallback is untested.** The network's second choice when max green or several overdue
  phases bind never occurred in the data. In those states it could do worse than the shield's "most halting".
- **The Smith-rule swap is crude.** It counts only stop-line vehicles, capped at 3 per lane, ignores upstream
  platoons and assumes the rest of the schedule is unchanged. It is rare, but could misfire.
- **The rule keys are guesses.** If `state["rules"]` uses other names, the TASK.md defaults apply. That is
  only a problem if the test rules differ from TASK.md.
- **The predictions are uncertain.** They are the dev naive means, from only 3 episodes per scenario. The
  test means over 10 new seeds could differ by about ±1-2 % from sampling alone.

## Files

- `supervisor.py`: the decision layer.
- `check_supervisor.py`: replay and edge-case checks.
- `analyze1.py` … `analyze6.py`: counting statistics of the recorded run. analyze6 counts how often the swap
  rule would trigger at margins 1.0 / 1.5 / 2.0; it is a count only, not a search, since the margin of 1.5
  was set by reasoning.
- `predictions.json`: predicted mean test TTS per scenario.
