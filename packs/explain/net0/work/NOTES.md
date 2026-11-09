# Notes: how the rules were derived

Python: `/home/traffic-ntua/anaconda3/envs/dlr/bin/python`. No simulator was run, no model was fitted to
decisions, and nothing searched over rules or thresholds. Every number comes from the weights (Jacobians),
from forward passes on recorded or synthetic states, or from medians over the network's own near-tie
states. Scripts are in `work/` (`explore*.py` in order, plus the checkers listed at the end).

## 1. What the recorded states show (explore1–4, 11)
- In states where a switch would be honoured (`ok = 1`), the network almost never keeps NS_through,
  NS_left or EW_left (0, 0 and 1 times in its own runs). It keeps EW_through in 383 of 855 states.
- 345 of those 383 holds are one **stall**: episode 2, EW_through green, with only N0, N1 and W2 occupied
  (1 car each), from 3625 s to the 5400 s cap. The logits were 2.09 for EW_through and 1.96 for EW_left.
  This is 19 % of the network's own `ok` states, so I report fidelity with and without it.
- Green durations: every phase switches at the first honoured decision (12 s). EW_through is the
  exception, held in light traffic.

## 2. Reading the network (explore5–7, 12, 17, 23, 40, 46)
- Mean input Jacobian per current phase, and one-vehicle probes on an empty junction (halted vs moving,
  each lane, each upstream lane, `t_green`, `ok`).
- Findings:
  - an empty junction always gives EW_through;
  - each phase's own stop-line lanes raise its logit; a moving vehicle counts about half a halted one;
  - N/S upstream lanes have about 0 weight; E/W upstream lanes favour EW_through by about 0.2–0.3 logit
    per vehicle against the N/S phases;
  - `t_green` has a small effect;
  - W0 is about 0; E2 > W2 and N2 > S2.
- Per-vehicle sensitivity falls as total load rises (about 4 logits per vehicle below 3 vehicles, about
  1.4 above 20), which is tanh saturation.
- Enumerated all 0/1 combinations on the 8 side lanes with EW_through green, at 12 s and 60 s
  (explore12), and over green time (explore40). This mapped the light-traffic hold region, which is where
  the stall lives.

## 3. Linear reference and grouping (explore10, 14–16, 18, 24, 35–38)
- A full first-order Taylor expansion per current phase (all 46 inputs) agrees 89.2 % on the network's
  own non-stall states. That is roughly the ceiling for linear rules. Its residuals are about 2 logits,
  spread over many lanes and non-linear.
- Grouped features (N/S through, N/S left, E/W through, E/W left, E/W upstream): slopes come from the
  Jacobian averaged over the network's own states, shifted per column to the readable form and rounded:
  `[nsT − 0.6 ewL, 1.1 nsL, 0.6 ewT − 0.5 nsL + 0.1 up, ewL]`.
- Variants checked, each with handicaps re-derived the same way:
  - lane weighting inside groups (N0/S0 > N1/S1, drop W0, E2 > W2, N2 > S2);
  - moving weight 0.3 or 0.7;
  - no upstream term;
  - per-phase slopes;
  - a `t_green` term.

  All landed within ±1 % of each other, so I kept the plain sums.

## 4. Handicaps (explore33–35, `tiebias.py`)
- Taylor intercepts depend on the slope and gave poor handicaps for some phases (NS_through green went
  from 83 % to 88 % once corrected). So I read them from the network's own **near-ties** instead.
- For each current phase and pair of candidates (p, q), I took the recorded states whose top-two logits
  are p and q, within 0.5 of each other. The median of my un-handicapped score difference S_p − S_q there
  is where the network is indifferent, so handicap_p − handicap_q = −median.
- With EW_through fixed at 0, the 2–5 such equations per current phase were combined by weighted least
  squares. This only combines the tie medians consistently: it uses no decisions and does not maximise
  agreement. Values were rounded to 0.1.
- Keep handicaps for NS_through, NS_left and EW_left (not identifiable from data, since the network never
  keeps them) come from the empty-junction probes: logit gap ÷ 3.5, the one-vehicle response scale.

## 5. Light traffic
- `rules_small` uses an explicit EW_through hold region and two light-traffic overrides:
  - **hold:** no E2, no N/S left-turners, W2 ≤ 1 and N/S through ≤ 2 + W2;
  - **lone N2 car:** a single halted N2 car on an otherwise empty junction is served at once;
  - **after NS_left:** below 4 vehicles a stronger EW_through default applies, read from the empty-junction
    probes for that phase.
- `rules_full` replaces all of this with one light-traffic model, used below about 4 vehicles:
  - per-lane "calls" are the network's one-vehicle responses at an empty junction (÷3.5, grouped,
    rounded), and barely differ between current phases;
  - each current phase gets a base row (the empty-junction logit gaps);
  - a green-time term comes from the 12 s vs 60 s probes.
- W2's coefficient was set to the mean of its 12 s and 60 s responses (0.85). That makes a lone W2 car
  held at every green time, as the network does.
- With NS_left green, the network's one-vehicle responses are about 0.8× the shared coefficients (for
  example N0 0.95 vs 1.14, N2 1.56 vs 1.93, E2 1.07 vs 1.48). That row's base values were divided by 0.8
  (−2.44 → −3.0, −1.40 → −1.75, −1.37 → −1.7) so the shared coefficients can be used. Agreement on the
  recorded states was unchanged. Light-traffic agreement with NS_left green rose from 0.894 to 0.932.

## 6. Checks
- Fidelity on recorded `ok` states (`tools.fidelity`, honoured):

  | file | all | network's own runs | own runs, no stall | fixed-time states | actuated states | comparisons |
  |---|---|---|---|---|---|---|
  | rules_small | **0.9208** | 0.923 | 0.905 | 0.947 | 0.882 | **8** |
  | rules_full | **0.9219** | 0.926 | 0.908 | 0.947 | 0.882 | **1** |

- Per own episode: 0.89–0.93 for normal episodes, 0.95 for the stalled one.
- Per current phase, own runs without the stall (full): NS_through 0.875, NS_left 0.911, EW_through 0.938,
  EW_left 0.892.
- `synthetic_light.py`: 4000 random light-traffic states (0–5 vehicles, halted or moving, any phase). The
  network agrees with small 0.871 and full 0.911.
- `synthetic_hold.py`: EW_through held 12–60 s with 0–4 side vehicles. Small 0.847, full 0.889. One- and
  two-vehicle cases: full 0.97 and 0.89.
- `stall_cases.py`: a table of the 1–3-vehicle cases that decide end-of-episode stalls. Both files hold for
  single N/S through cars, most N/S through pairs, lone W2 at any green time, lone S2 after 17 s, and
  N0+N1+W2.
  - Both serve a lone N2 car even after 57 s, where the network holds it.
  - `rules_small` also holds N0+S0 and N2+S1 at short greens, where the network serves them, and releases
    S2+W2 to EW_left, where the network holds.
  - `rules_full` misses N0+S0 after 12 s and N0+W2 / N0+N1+S1 at 57–60 s.
- Random uniform inputs: both files always return a phase in 0–3.

## 7. Size accounting
`tools.count_comparisons` gives 8 for `rules_small` and 1 for `rules_full`. Both also choose a phase with
`max(...)` over at most 4 phase scores, which is 3 pairwise comparisons not counted by the tool. Counting
them, the totals are 11 for `rules_small` and 7 for `rules_full`: 1 + 3 + 3, because only one of its two
argmaxes runs per decision.

## 8. Tried and rejected
- A load-proportional bonus for EW_through: worse (−2 %).
- Separate tie handicaps for light and heavy load: too few ties, unstable (−2 %).
- Per-phase slopes: no gain.
- Forcing "never keep" in rules_full: no gain on data, worse in light traffic.

`rules_full_v1.py`, `rules_full_v2.py` and `rules_full_v3.py` are intermediate versions of `rules_full.py`:
- v1: hand-written light-traffic branches;
- v2: light-traffic model for EW_through only;
- v3: light-traffic model for all phases, before the NS_left rescaling.

They are kept for the record, and `explore45.py` imports v1.

## 9. Predictions (`predictions.json`)
- **Fidelity.** On the network's own normal runs the rules agree about 90.5 %. In-sample handicaps should
  lose a little on held-out states, but any stalls in the test runs that the hold logic covers will add to
  agreement. Hence 0.90 for small and 0.905 for full. A stall whose configuration the rules mis-classify
  would cost several points.
- **TTS.** The network's dev mean is 44.7 veh-h. Disagreements are near-ties, so closed-loop TTS should
  stay close. Small has a few more stall configurations that the network does not have (N0+S0 and N2+S1
  held). Hence 45.3 for small and 45.0 for full.
