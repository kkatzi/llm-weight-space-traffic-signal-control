# Supervisor for the PPO network under the operating rules

## What the recorded data says about this network (states.npz, 6 episodes, 4433 decisions)

- **It switches as soon as it can.** At free decisions (every phase legal, nothing overdue), it asks for a different
  phase 97 % of the time. In training (no rules) every green ended at 12 s. Under the rules, through greens end at
  17 s (pedestrian minimum) and left greens at 12 s. Its longest green in the data was 37 s, so the 45 s maximum
  green never bound.
- **The network's switch requests fail mostly because of the minimum green.** 2568 requests asked to switch while
  only the current phase was legal. In those cases the shield's replacement is the only legal phase, so no other
  controller could do differently. Its choice at a through green's 12 s (blocked) matches its choice at 17 s in 87 %
  of cases, so the 5 s delay does not confuse it.
- **Max wait:** a phase is overdue when `since_served >= 90` and it has halting vehicles (no exceptions in the data).
  That gives about 23 overdue events per episode, mostly NS-left (phase 1). In 46 % of them the network was going to
  pick the overdue phase anyway. In the other 54 % (about 13 per episode) the rule changes the order of phases: the
  phase the network wanted waits one 15 s slot.
- **Its logits are decisive and track queues.** The median top-2 gap at free decisions is 5.6; it is below 0.5 in
  only 5 % of them. Each phase's centred logit correlates 0.7-0.8 with that phase's halting count.

## What the supervisor does and why

`decide(x, logits, state)`:

1. **Respect the rules.** It never returns an illegal phase. If the network's argmax is illegal, it returns the legal
   phase with the highest logit, where the shield would take the legal phase with the most halting vehicles. This
   only changes the outcome when two or more phases are legal and the network's pick is not one of them: max green,
   or several phases overdue at once. Neither happened in the recorded data, but they might at high demand.
2. **Anticipate max wait, only in near-ties.** At a free decision, let r be the network's pick. Suppose a phase q
   would become overdue by the first decision at which r's green may end (`since[q] + 20 s` for a through r,
   `+15 s` for a left r, `+5 s` if r means staying). Then q is forced next anyway, so the real choice is the order:
   r then q, or q then r. The supervisor serves q now only if both hold:
   - the network nearly ties them: `logit[r] - logit[q] <= 0.75`;
   - Smith's rule (serve the phase that releases the most halting vehicles per second of junction time first)
     strictly favours q: `halt[q] * H(r) > halt[r] * H(q)`, with H = 20 s for through and 15 s for left.

   It is not more aggressive, for two reasons. The network's pick usually has the larger queue, so serving it first
   is the cheaper order. And serving q early moves q's next deadline earlier, which means more forced services over
   the hour. Waiting for the forced service is near-optimal on average.
3. **Anticipate the pedestrian minimum, only in near-ties.** A through green now blocks the junction for 20 s rather
   than the 15 s the network learned with. When the network's pick is a switch to a through phase and the best left
   phase is within 0.5 logits, the supervisor takes the left phase if:
   - the left phase has halting vehicles;
   - it has more halting vehicles per second of blocked time (`halt[l]/15 > halt[r]/20`);
   - the through phase is not itself close to max wait.

The margins (0.75 and 0.5) were chosen by reasoning, not by search. They sit inside the band where the network is
close to indifferent (top-2 gap under 0.5 in 5 % of free decisions, under 1.1 in 10 %), so the network stays in
charge everywhere else. Everything else returns the network's argmax unchanged.

## When it overrides the network

- **Minimum green running, or one overdue phase:** only one phase is legal, so it returns that phase. This is the
  same outcome as the shield, and the network's request would have had no effect anyway. It accounts for almost all
  of the raw disagreement.
- **Network's pick illegal but several phases legal:** the best legal phase by logit.
- **The two near-tie anticipations above.**

Rates on the recorded states, replayed one step at a time:
- It differs from the network's argmax in 61.4 % of decisions. That figure is the reported metric, and it is almost
  entirely the shield's forced phases (minimum green, single overdue phase) made explicit.
- It differs from what naive (network + shield) would actually apply in only **4 of 4433 decisions (0.09 %, about
  0.7 per episode)**: 2 early max-wait services and 2 through-to-left near-ties.

## Checks I ran (all scripts in work/)

- `explore1-9.py`: these computed statistics of the recorded data and ran the network on the recorded states.
  - The recorded `act` equals the network's argmax on every state.
  - The rule_state layout (phase one-hot vs green, t_green) is consistent.
  - The overdue threshold is since ≥ 90 with halting > 0.
  - `phase_halting` = stop-line queue × lane capacity, with correlation 0.99999.
  - I also looked at green durations, transitions, where requests become illegal and why, the overdue events and
    their logit gaps, queue statistics and how often near-ties occur.
- `check_supervisor.py`:
  - On every recorded state the output is a legal phase.
  - It counts the rates above and lists each decision where the outcome differs from naive.
  - It runs hand-made edge cases: max green with several legal phases, minimum green, one and two overdue phases, no
    legal phase, max-wait anticipation with a small and a large gap, the through/left near-tie and its guard.
  - It confirms that supervisor.py imports only numpy and math.
- I did not use a simulator, fit any model or search over variants. The one change made after looking at the
  replay, adding the Smith condition to rule 3 and making both comparisons strict, came from reasoning about cases
  where the rule picked a left phase with fewer halting vehicles per second than the through phase (e.g. 4 vs 9).
  It was not a score-driven search.

## What I expect to go wrong / limitations

- **The gain over naive will be very small, probably within seed noise.** For this network most of the 2.2 veh-h
  rule cost is the 5 s added to every through green (about 100 through greens per hour). A supervisor cannot remove
  that. The only lever is choosing through phases less often, and I could not calibrate that without a simulator. It
  also risks through queues spilling past the 40 m flare and blocking left-turners on the shared upstream lane, so I
  limited it to near-ties.
- **Single-step replay cannot show knock-on effects.** A flipped decision changes all later states. With about 0.7
  flips per episode this should not matter much, but I could not measure it.
- **The near-tie rules use stop-line halting counts only.** These are capped at about 3 per lane and miss upstream
  queues and approaching platoons, which the network does see. In a near-tie the network may have had a reason the
  supervisor cannot see.
- **At high demand the network may hold greens longer.** Max green could then bind more often; there my choice
  (network's best legal phase) and the shield's (most halting) differ, by design. I expect this to help slightly but
  it is untested.
- **The rule constants (15/10 s minimum, 90 s max wait, 5 s steps, 3 s yellow) are hard-coded from TASK.md**, not
  read from `state["rules"]`, whose keys I could not see. The legal and overdue flags, which decide everything that
  matters, come from `state`.

## Predictions

Essentially the naive (shield-only) numbers, minus a negligible amount: low 36.1, nominal 47.0, high 57.85 veh-h.
The uncertainty is about ±0.7 veh-h per scenario from the held-out seeds alone.
