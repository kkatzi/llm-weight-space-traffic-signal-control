# How the rules were derived, and the checks I ran

Python: `/home/traffic-ntua/anaconda3/envs/dlr/bin/python`. All scripts are in `work/`. I used only forward passes,
the network's analytic gradients, and conditional statistics on `states.npz`. No model was fitted to the
decisions, no simulator was run, and no automated search over thresholds was made.

## 1. Basic behaviour (`explore1.py`, `explore3.py`, `stats_for_docs.py`)

- On the 5,095 states with `ok = 1`, the network keeps the current phase in only 2.9 % of them (4.7 % on its own
  runs). 97 % of its switches happen at green age 12 s, the first honoured decision.
- So the policy is "switch at minimum green". The work is in modelling *which* phase comes next, and the rare
  extensions.

## 2. Which inputs matter (`explore2.py`, `explore5.py`, `explore13.py`, `explore14.py`)

- Mean input gradients of the logits per current phase: every logit rises with the queues of its own lanes. The
  lane-to-phase mapping matches TASK.md. Served lanes have zero queue at decision time (`explore13.py`).
- Single-lane probes (filling one lane with 0–3 halted vehicles from an empty junction): left pockets have the
  largest effect, and N1 and W0 have small effects split between the two through phases.

## 3. Shape of the decision boundaries (`explore8.py`)

- Cross-tables of the network's choice between two candidate phases against vehicle counts per movement
  (NS-through, NS-left, EW-through, EW-left) show clean, roughly linear boundaries. One left-pocket vehicle is
  worth about 1.6–2.2 through vehicles when the through sum includes N1 and W0.
- A first rule (argmax of movement densities, left ×2, other phases only) gave 82 %. Adding hand-read
  ring offsets gave 86 % (`try1.py`–`try4.py`). `try1.py` compared six reasoned variants: left weight
  1/1.5/2 × density/queue. None of these values was kept; the final weights come from step 4.

## 4. Lane weights from the network's own gradients (`explore15.py`–`explore22.py`)

- Linearising the network around the mean state per current phase gave only 87 %, but the intercept there is
  wrong. Using the **average gradient over states near a decision boundary** (top-2 logit gap < 2) with the mean
  logit residual as intercept gave **95 %** (per current phase), **92 %** with one gradient shared across
  current phases, and 91 % after grouping lanes by movement (`explore18.py`, `explore26.py`; diagnostics only,
  not used as rules). The policy is therefore close to "argmax of linear scores", and per-current-phase weights
  add only about 1 %.
- From the shared boundary gradient I read the **pairwise push per halted vehicle** of each lane (own phase
  minus each competitor, `explore17.py`, `explore21.py`). In units of one N0 vehicle: N0 1, S0 0.95, S1 0.7,
  N1 ≈ 0.3–0.55 (for both through phases), N2 1.4, S2 1.8, E0 0.65, E1 1.05, W1 0.8, W0 ≈ 0.35–0.55 (for both
  through phases), E2 1.3, W2 1.3. A moving vehicle counts about half a halted one, hence lane value =
  1.5·(q + d).
- Group-level cross terms (`explore22.py`, min-shifted per group): through phases and EW-left receive about 0.2–0.3
  per vehicle waiting elsewhere, and NS-left receives none. E/W upstream vehicles add about 0.09–0.1 per vehicle to
  the E/W phases relative to NS-through. N/S upstream adds 0.03 to NS-through.
- These rounded numbers are the scores in `rules_full.py`. `rules_small.py` keeps only movement sums, N1/W0 at
  0.5, left ×1.6/×1.3, and E/W upstream ×0.1.

## 5. Preference (offset) tables (`explore24.py`, `try9.py`, `trysmall.py`, `explore29.py`, `explore34.py`)

- For each current phase and each pair of candidates, I took the states whose top-2 logits are that pair (gap < 2)
  and computed the mean of (logit difference / 2.3 − score difference). This closed-form statistic is the offset
  between the two candidates in vehicle units. I reconciled the 3–6 pairwise numbers per current phase into one
  row by hand.
- Then I read, by eye, the point where P(network picks a) crosses 50 % in binned tables of the rule margin
  (`explore29.py` / `explore36.py` for full, `explore34.py` / `explore35.py` for small). Where it was clearly off
  zero I shifted the row by hand: two rounds for each file, a few entries each. Some pairs conflict: moving one
  boundary would move another. I left those as compromises.
- Things I tried and dropped because they did not help: a phase-dependent NS-left weight (`try10*.py`: 90.0 %
  vs 90.8 %), stronger upstream weights (`try11*.py`: 90.5 %), and a compact 3-number offset structure for the
  small file (88.4 % vs 89.3 %).

## 6. Comparison count

The 4-way maximum is written as a knock-out with three explicit comparisons, so `tools.count_comparisons`
(3 for each file) equals the comparisons actually made per decision. Nothing is hidden in `max`, `argmax`
or loops. `np.argmax` is used only to read the one-hot current phase.

## 7. Checks (`check_rules.py`, `explore30.py`, `explore32.py`, `explore33.py`)

| file | fidelity, ok states (5,095) | all states | network's own runs | fixed-time runs | actuated runs | comparisons |
|---|---|---|---|---|---|---|
| rules_small.py | 0.903 | 0.821 | 0.894 | 0.924 | 0.884 | 3 |
| rules_full.py | 0.910 | 0.831 | 0.891 | 0.931 | 0.900 | 3 |

- Extension rate on ok states: rules 2.9 % (small) and 2.85 % (full), network 2.9 %.
- Errors concentrate on near-ties. Where the network's top-2 logit gap is ≥ 1 (88 % of states), the rules agree
  96.0 % (full) and 95.0 % (small). Where the gap is ≥ 2 (77 %), they agree 98.2 % and 97.3 %. Where the gap
  is < 1, they agree only 55–57 %.
- By load (vehicles on stop-line + upstream lanes; full / small): < 10 vehicles 0.81 / 0.82 (many extensions
  there), 10–40 vehicles 0.90–0.93, ≥ 40 vehicles 0.86 / 0.85. The main high-load errors involve EW-left vs
  NS-through after EW-through. The network gives E/W upstream traffic slightly more weight there, but raising
  that weight did not improve overall agreement (`try11*.py`).
- Synthetic saturated states (`explore32.py`): when every red lane is full, the network strongly avoids NS-left
  and often prefers EW-through (even extending it). The rules give closer scores there. These states are outside
  the recorded range, but they may matter in the high-demand scenario.

## 8. Predictions (`predictions.json`)

- Held-out fidelity is measured on the network's own runs, where both files score about 0.89. The offsets were
  set partly from these same states, so I expect slightly less on new episodes: about 0.88 for both.
- TTS: the network has 44.5 veh-h on the dev episodes. The disagreements are mostly near-ties and the extension
  rate is matched, so I expect the rules to be close but a little worse: full about 44.9, small about 45.0.
