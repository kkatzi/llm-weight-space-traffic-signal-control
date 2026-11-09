# Notes: how the rules were derived

Python: `/home/traffic-ntua/anaconda3/envs/dlr/bin/python`. All scripts are in `work/`. `common.py` loads the
network, the recorded states and helpers. No simulator was run. Nothing was fitted to the network's decisions, and
no loop searched thresholds or weights to maximise fidelity. The few places where I compared hand-specified
alternatives are listed under "Choices compared" below.

## 1. Basic behaviour (explore1, explore5, explore12, facts.py)

* On honoured states (`ok = 1`, 5077 states) the network almost never picks the current phase. Across all
  controllers it extends only 97 times: 89 in EW-left and 8 in EW-through. In its own runs it extends 42 of 1503
  times. It never extends NS phases.
* In its own runs 1455/1503 honoured decisions happen at green age 12 s, so every green is the minimum one.
  The policy is a phase selector: "which phase next".
* `t_green` is irrelevant in practice: its gradient is about 0.02 logit per second.

## 2. What drives the choice (explore3, explore8, explore10, explore14, explore24–26)

* Mean input gradients of the logits per current phase (explore3) and pairwise logit-difference gradients
  (explore8, explore10 on near-boundary states) show that each phase's logit rises with its own stop-line lanes
  and falls with the others. Upstream lanes matter far less per vehicle, and `t_green` hardly at all.
* Finite differences, adding one halting or one moving car to a lane over the network's own states (explore14),
  gave the lane weights:
  * N0 is weak (0.38 relative effect), N1 1.34, S0 1.11, S1 0.96;
  * N2 2.4, S2 2.4; E0 1.1, E1 1.8, W0 0.9, W1 3.0; E2 2.5, W2 2.0;
  * a moving car is worth about half a halting car, so lane load = 3·(q + d), with a halting car = 2 and a
    moving car = 1.
* Synthetic grids with uniform load levels per movement group (explore20) showed the qualitative structure:
  * argmax over the other three phases, close to "largest per-lane load";
  * a strong EW-through preference after either left-turn phase;
  * fixed tie orders.
* **Linearisation test (explore24, explore25).** A first-order Taylor expansion of the network for each current
  phase (mean gradient over near-boundary states, anchored at the mean logits), followed by argmax over the other
  phases, agrees with the network on 94.3 % of states. This is a check of linearity, not a fit. Variants:
  * gradients shared across phases, stop-line lanes only: 92.4 %;
  * same, with q and d combined into one lane load: 91.7 %;
  * own-lane-only terms (no cross effects): 84 %.

  So the network is close to a linear scorer of lane loads, and the cross-group effects carry about 7 points.

## 3. Compressing to readable scores (explore28–30, explore33–34, explore37)

* I averaged the shared Taylor coefficients (per unit of lane load, centred over the four logits) within movement
  groups, weighting each lane by its own-phase coefficient, rounded:
  * NS-through = N1 + 0.75(S0 + S1) + 0.25 N0
  * NS-left = N2 + S2
  * EW-through = E1 + 1.5 W1 + 0.75(E0 + W0)
  * EW-left = E2 + W2

  This gives a 4×4 matrix of group effects on each logit.
* I shifted each column by a constant to make it sparse (argmax is unchanged by adding the same quantity to all
  four scores) and divided by 1.13 so that NS-through has coefficient 1. Rounding gives the four phase scores in
  `rules_small.py`. The group-level model agrees about 88.5 %.
* The residuals between the lane-level and group-level coefficients (explore33) were largest for the EW lanes:
  * E0 and W0 push towards EW-left and away from NS-left;
  * E1 and W1 push towards EW-through and away from EW-left.

  Splitting EW-through into straight (E1 + 1.5 W1) and right+straight (E0 + W0) gives 89.9 % (explore34).
* Upstream: the Taylor coefficients show that N1u and S1u favour NS-left and NS-through, and that the EW upstream
  lanes favour EW-left and EW-through (explore36). Two counts were added: `up_ns = N1u + S1u` and
  `up_ew = E0u + E1u + W0u + W1u`, in vehicles. This gives 90.6 % (explore37).
* Per-phase group matrices instead of one shared matrix gave no gain (explore42, 90.8 % vs 90.7 %), so the scores
  are shared and only the bonuses depend on the current phase.

## 4. Phase-transition bonuses (derive_bonuses.py)

For each current phase *p* and each pair of candidates (*i*, *j*), I took the recorded honoured states where *i*
and *j* are the network's top two logits and differ by less than 0.75, i.e. the network's tie points. The median
of score_i − score_j there is the boundary offset b_j − b_i. The three pair offsets per phase were combined with
count weights and centred, then rounded to about 0.05. This is a statistic of the network's switching decisions.
The pair offsets are not perfectly consistent, especially after NS-through, which shows the remaining
non-linearity.

## 5. The EW-left hold (explore32, explore35, explore40, stay_table.py)

The 89 EW-left holds have no crisp region, but they become likely when two things hold:

* a car is still in or entering a left pocket (`ew_left ≥ 1`);
* the best alternative's score is low.

For each rule file, `stay_table.py` tabulates P(hold) by the best alternative's score and by `ew_left`. I read off
where P(hold) crosses 50 %:

* small rules: about 8, 11 and 14 for `ew_left` = 1, 2 and 3, giving `best < 5 + 3·ew_left`;
* full rules: about 6, 8.5 and 11, giving `3.5 + 2.5·ew_left`.

With `ew_left = 0`, P(hold) never exceeds 50 %, so there is no hold. Holds in EW-through (8 cases) are not
modelled.

## 6. Choices compared (by hand, a handful of alternatives, not searched)

* explore9: I evaluated five values (0 to 1) of a common weight for lane 0 in an early model and did not keep
  any of them. The final lane weights come from the gradients and finite differences above.
* Bonus estimates: mean-matching of the linear model to the mean logits on near-boundary states vs the
  tie-point medians. They agreed to within about 1 point of fidelity. I kept the tie-point method because the
  location of the decision boundary is what matters for an argmax.
* A simpler 4-group score (dropping the smallest cross terms) lost about 1 point. A pairwise "tournament" with
  pair-specific offsets gained only 0.3 points and was not used.
* Upstream terms were added to the small rules after they gave about 1 point and better agreement on saturated
  synthetic states.

## 7. Checks

| file | comparisons | honoured fidelity (all recorded) | network's own runs | fixed-time runs | actuated runs |
|---|---|---|---|---|---|
| rules_small.py | 6 | 0.902 | 0.904 | 0.910 | 0.888 |
| rules_full.py | 6 | 0.912 | 0.915 | 0.919 | 0.900 |

Measured with `tools.fidelity` and `tools.count_comparisons` (and `check_rules.py`). Other checks:

* **Per-episode agreement** on the network's own runs: 0.87–0.92 (small) and 0.87–0.94 (full). Agreement is
  stable over the hour, 0.89–0.92 per 15-minute block (`bias_check.py`).
* **Phase-choice bias.** The rules pick each phase about as often as the network. The one shortfall is EW-left
  holds: the rules make 50–56 of them against the network's 97, with 35–39 in common.
* **Disagreements are close calls** (`margins.py`): the median network logit margin is 1.0–1.2, and 69–74 % are
  within 2 logits.
* **Synthetic probes** (`facts.py`): agreement on 256 uniform-level grid states is 0.92 (small) and 0.965 (full).
  The rules reproduce the main-road-priority probes exactly.
* **Saturated synthetic states** (`saturation_check.py`, out of distribution): agreement is 0.68 (small) and
  0.81 (full).
* **Robustness.** Random inputs always return an int in 0–3.

## 8. Predictions (predictions.json)

* **Fidelity.** On the recorded network runs I measured 0.904 (small) and 0.915 (full). The bonuses and hold
  thresholds were read off these same states, and the test mix (0.8/1.0/1.15 × nominal, no directional bias)
  differs a little from training. I therefore predict 0.89 (small) and 0.90 (full).
* **TTS.** The network scored 44.8 veh-h on the dev episodes. The rules keep its main properties: minimum greens,
  the same transition habits, no bias in phase choice. Their disagreements are mostly close calls, so I expect
  closed-loop TTS within about 1 % of the network. I predict 45.1 (small) and 45.0 (full) veh-h.
