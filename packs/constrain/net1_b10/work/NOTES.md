# Supervisor for the network under the operating rules (pack constrain_net1_b10)

## Result

| supervisor | low | nominal | high | all 9 dev |
|---|---|---|---|---|
| naive, shield only (reference) | 36.1 | 47.1 | 58.0 | 47.048 |
| v1 respect only (eval call 1) | 36.1 | 47.1 | 58.0 | 47.048, identical to naive |
| **final `supervisor.py`** (eval calls 2 and 3) | **36.0** | **47.2** | **57.7** | **47.007** |

The final supervisor's dev TTS is 47.007 veh-h, against 47.048 for the naive shield. The 0.04 difference is
within chaos noise: per-episode changes are −0.21, −0.01, +0.02, +0.00, +0.55, −0.04, −0.19, −0.34, −0.16.
I am not claiming a real improvement. The supervisor is essentially the network made rule-compliant, with two
small rule-aware tie-breakers.

## What the network does under the rules (from states.npz, 4437 decisions)

- It is a minimum-green sequencer. It asks for a switch at every chance: of the 1268 decisions where a switch
  was allowed, it kept the current green in only 26. Its keep logit is about −4.5 when a switch is possible.
- Every green clears its own queue. At the last decision of every left green (t_green 12) and every through
  green (t_green 17), the halting count on that phase's lanes is 0 in all 1241 cases. Upstream queues are
  about 0. The junction is well undersaturated.
- So the rules cost efficiency in two ways:
  1. **Through minimum green.** Through greens last 17 s instead of 12 s. The 5 extra seconds only serve new
     arrivals while every other queue waits. A cycle through all four phases grows from 60 s to 70 s. This is
     most of the 2.5 veh-h loss and no supervisor can avoid it: any through green the controller starts must
     last 17 s.
  2. **Maximum wait.** About 12 forced services per episode, mostly NS-left (phase 1), which the network
     serves at about the 90 s limit anyway (mean interval 88 s).
- Maximum green never binds: the network never holds a green that long (the longest was 47 s, once).
- The network's choices are already close to rule-aware optima:
  - **Visit frequencies.** A polling model with fixed slot lengths (20 s through, 15 s left, including yellow)
    and nominal arrival rates gives an optimum of visit frequency ∝ √(λ/slot). That predicts about 50, 38, 65
    and 49 visits per hour for phases 0–3. The network makes 50, 41, 61 and 54.
  - **Order.** The network picks the phase with the most halting vehicles 81 % of the time. When it picks a
    through phase, that phase almost always has at least 1.33× the halting vehicles of any left phase, which
    is the break-even ratio once the 20 s vs 15 s slot difference is counted.
- The green-time input hardly matters to the network. Setting t_green to 12/60 instead of 17/60 changes 0.8 %
  of its decisions. So its decisions at 17 s (a state it rarely saw in training) are not out of distribution
  in any harmful way.

## What the supervisor does (`decide(x, logits, state)`)

1. **Respect.** It never asks for an illegal phase. If the network's argmax is illegal, it returns the legal
   phase with the highest *logit*. The shield would pick the legal phase with the most halting vehicles. In
   practice almost every illegal request happens when only one phase is legal (the current green during its
   minimum, or the single overdue phase), so this rarely changes anything. Shield overrides in the
   evaluation: 0.
2. **Anticipate the maximum wait.** Say the network's choice is q, and starting q would let another phase p
   (with halting vehicles) reach `max_wait` before q may end. Then the rules will force p right after q
   anyway. Both will be served next, and only their order is free. The supervisor picks the order with less
   delay for the two queues (pairwise interchange):
   - cost of q first = h_p·slot_q + λ_p·slot_q²/2
   - cost of p first = h_q·slot_p + λ_q·slot_p²/2
   - here slot = 3 s yellow + the earliest decision at which the green may end: 20 s through, 15 s left;
     λ = the nominal arrival rate from TASK.md.
3. **Anticipate the through minimum green, in close calls only.** Under the rules a through green takes a
   20 s slot, not 15 s. When the network's choice is a through phase and a legal left phase is within
   DELTA = 1.0 logit, the supervisor takes whichever clears more halting vehicles per second of slot. Outside
   such close calls the network keeps its choice.

The rule numbers (min greens, max wait) are read from `state["rules"]`. If they cannot be parsed, the
TASK.md values are used. Only numpy and math are used, and the network is used only through `logits`.

## When it overrides the network

- **During minimum greens.** The network asks for a switch while the rules forbid it, and the supervisor
  returns the current green. This is 65 % of all decisions, so the raw "differs from the network's choice"
  rate is about 65 %. In these cases the shield would do exactly the same: only one phase is legal.
- **When a phase is overdue or max green ends the current green.** The supervisor returns the best legal phase
  by logit, which is the forced phase when only one phase is legal.
- **Real choices.** On the recorded states, the supervisor departs from the shielded network in only 14 of the
  1121 decisions where more than one phase was legal (1.2 %): 11 from rule 2 and 3 from rule 3. In the
  simulator its switch counts per episode match the naive run within ±1.

## Checks I ran

- `analyze.py`, `analyze2.py`, `analyze3.py`: what kinds of illegal request occur, t_green values, transition
  counts, logit gaps, which phases become overdue and when, residual queues at the end of each green, and
  service-interval statistics. These are descriptive statistics only; nothing was fitted.
- `count_overrides.py <file>`: how often a candidate supervisor changes the network's or the shield's decision
  on the recorded states, and that it never outputs an illegal phase.
  - The through/left slot correction would change only 5 of 1121 free decisions at DELTA = 1. Even with no
    gate at all it changes only 18.
  - The forced-order rule changes 11.
  - So no version of these ideas could move TTS by much.
- Sensitivity of the network to its t_green input (described above).
- Simulator (3 of 10 calls):
  - Call 1, `sup_v1_respect.py`: 47.048, exactly the naive reference. This shows the simulator is
    deterministic and gives the per-episode baseline.
  - Call 2, `sup_v4_combined.py` (the final logic plus a one-time dump of the state format to
    `state_sample.json`): 47.007.
  - Call 3, the final `supervisor.py`: 47.007, identical. Offline it makes identical decisions to v4 for every
    rules format I tried.
- I stopped at 3 calls on purpose. The remaining ideas change too few decisions to show up against the chaos
  noise on 9 episodes, which is about ±0.2–0.5 veh-h per episode. Choosing among such variants on dev would
  only fit noise.

## Ideas considered and rejected

- **Holding a green past its minimum.** Every green already clears its own queue, and the network strongly
  prefers to switch.
- **A larger through penalty, or skipping through phases.** The network's visit frequencies are already near
  the √(λ/slot) optimum. Moving away from an optimum costs at second order, and I had no way to calibrate the
  logit scale without fitting.
- **Serving soon-overdue phases early, unconditionally.** The forced service already comes as late as the
  rules allow, which is the most efficient time (the most vehicles per slot). Serving earlier adds visits.
- **Making the sequence more regular.** The service intervals vary by about 25 % (CV ≈ 0.25), which in theory
  adds about 6 % delay. But the irregularity is how the network adapts to random arrivals. Fixed-time is
  perfectly regular and does worse (49.5).

## What I expect to go wrong

- **No real gain over the naive shield.** Expect test TTS close to the naive network under the rules
  (dev 47.0). The difference will be within ±0.2 in either direction, decided by chaotic divergence more than
  by the logic.
- **Seed variation dominates.** Dev scenario means come from only 3 seeds each, with per-seed spread of about
  1 veh-h. The 10-seed test means could differ from dev by roughly ±0.5–1 veh-h per scenario.
- **Rule 2 uses nominal arrival rates.** In the high scenario, real rates are about 15 % higher, so the cost
  comparison is slightly off. It only changes order in a handful of decisions per episode.
- **If test demand were much heavier** (saturated, with residual queues and pocket spill-back), the fixed
  slot lengths in rules 2 and 3 would underestimate the green actually used. That is not the case for the
  stated test scenarios.
