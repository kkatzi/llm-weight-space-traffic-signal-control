# Repair of the network for the dead W2 detector

Delivered: `work/weights.npz` (built by `work/build.py`, `build.build(1.0)`). **364 of 7428 parameters changed**
(W1: 172, b1: 64, W2: 64, b2: 64, W3/b3: 0). Simulator calls used: 3 of 5.

## What I built and why

**1. Stand-in for the dead detector (192 parameters).** The network reads q_W2 and d_W2 as "west left-turn demand".
With the detector stuck at 0, its phase-3 (EW_left) logit lost on average about 2.4 (sensitivity × mean pocket
occupancy). It served phase 3 in only 21 % of the honoured decisions, and the west left-turners piled up. That cost
the most at high demand (60.9 vs 55.0). In the network's own pre-activation I replace the two dead inputs by an
estimate made from inputs that still work:

    q_W2_hat = d_W2_hat = 0.5 * d_E2 + 0.25 * (1 - phase3)

- *Mirror term.* The opposite pocket E2 has the same demand (E and W always get the same bias) and is served by the
  same phase, so it is the natural proxy. It is only weakly correlated, though: on the N/S pair, which I can check,
  the correlation between the two left pockets is 0.3–0.5. Their conditional means give a slope of about 0.45 per
  vehicle, so I shrink the proxy to half.
- *Floor.* Outside phase 3 the pocket keeps filling, so I add a constant that keeps the imputation's mean close to
  the E2 mean outside phase 3 (0.41–0.46). It also stops "W2 is empty" from ever being taken as a fact. In phase 3
  (the pocket's own green) there is no constant, because the pocket is draining. 0.25 is the smallest floor that
  makes an empty network still visit phase 3 (see the checks). It is a little above the pure mean-match value of
  0.18–0.23.
- *Folding.* `W1[:,17] += 0.5*(W1[:,11]+W1[:,23])`, `W1[:,43] -= 0.25*(...)`, `b1 += 0.25*(...)`. One column (d_E2)
  feeds both estimates, which saves 64 parameters. q_E2 and d_E2 correlate at 0.92.

**2. Fix for drain deadlocks (172 parameters).** Forward passes on frozen near-empty observations showed that the
trained network, even with every detector working, never serves some lone cars. A single car on N0, S1 or W0, or a
hidden W2 car, gets no green: the network cycles 0→1→3→0 or holds 2 forever. Once the hour ends, such a car waits
until the 30-minute cap, about 0.5 veh-h per car. The recorded start-network episodes show this in 2 of 6
episodes, and my first simulator call left 1–8 cars at the cap in 4 of 9 episodes.
- **Layer-1 unit 50 is rewritten as a "lone through-traffic" detector.** I picked unit 50 because it is the least
  important: freezing it at its mean changes 0.15 % of decisions. It computes
  `h = tanh(20*(A - 0.1 - O) - 40*phase2)`.
  - A = q_E0 + q_E1 + q_W0 + q_W1, the halted cars waiting for phase 2.
  - O = all other visible demand: N/S stop-line q+d (pockets N2/S2 weighted 0.3), 3×(q_E2+d_E2), 10×(N/S
    upstream d), and 10×(q_E1u+q_W1u), the left-turn spillback.
  - So h ≈ +1 only when E/W through cars wait, nothing else is visible, and phase 2 is not green. Otherwise h ≈ −1.
- **Its outgoing column** `W2[:,50] = u` raises the phase-2 logit, with `u_k = W3[2,k] − mean_{c≠2} W3[c,k]`.
- **Compensation:** `b2 += u + W2_old[:,50]*mean(h_old)`. In the off state (h = −1), layer 2 then gets exactly
  what it got with the old unit frozen at its mean.
- **Why this is enough:** the floor makes an empty network leave phase 2 for phase 3, and from phase 2 the network
  already handles every other lone car correctly.

No weights were fitted. All values come from symmetry, a few data means/correlations, and hand-set thresholds in
units of one car (one car = 0.335 on a stop-line lane, 0.029 on an upstream lane).

## Checks run

- Edit size 364 ≤ 371 (`tools.edit_size`). Shapes and finiteness pass (`tools.save`).
- Recorded states: 20 % of honoured decisions change against the start network. Most of the change moves picks to
  phase 3 (share 0.21 → 0.34). The detector flips only 29 main-hour decisions; the rest of its flips are in the
  recorded deadlock tails.
- Drain check (`drain_check.py`). These are forward passes on frozen synthetic observations with the switching rule
  applied, not a traffic model. It covers every set of 1–3 lanes (including hidden W2 and W2+W1u spillback) with 1
  or 3 cars, from each start phase: 2940 cases.

| network | cases never cleared |
|---|---|
| start network | 797 |
| mirror only | 784 |
| imputation only | 485 |
| delivered | 0 (slowest 90 s) |

- Simulator (dev, 9 episodes), mean TTS in veh-h:

| weights | all | low | nominal | high | cars left at cap |
|---|---|---|---|---|---|
| start network (reference) | 47.9 | 35.9 | 46.9 | 60.9 | — |
| call 1: full mirror only (128 parameters) | 45.9 | 36.5 | 45.8 | 55.5 | 8, 1, 1, 1 in 4 episodes |
| **call 2: delivered** | **45.2** | **35.0** | **45.4** | **55.2** | none |
| call 3: same, mirror coefficient 1.0 | 45.2 | 34.9 | 45.6 | 55.0 | none |

  Calls 2 and 3 differ only by noise. I kept 0.5 because it keeps the imputation inside [0, 1] and is the
  better-justified estimate. For reference, the working-detector network scores 44.8 (34.5 / 44.9 / 55.0).

## What I expect to go wrong

- A hidden W2 queue that does not match E2 (for example, a burst of west left-turners while E2 is empty) is still
  seen only as about 1–1.5 cars. Phase 3 can come late, and W2 spillback can block W1u. This is the remaining gap
  to the working-detector network, mostly at high demand.
- The floor makes the network think about 0.75 cars wait in W2 even when it is empty, so at low demand phase 3 is
  sometimes served for nothing. That costs a few seconds per cycle.
- The drain check uses frozen observations and only up to 3 occupied lanes. Cars that are moving, or larger mixed
  tails, could still form cycles I did not test (none showed up on dev).
- The detector pushes phase 2 hard whenever E/W through cars are the only visible demand. If the hidden W2 pocket
  holds cars at that moment, they wait one more phase.
- Dev has only 9 episodes, and per-episode differences of about ±1.5 veh-h between near-identical variants were
  seen at high demand. The test mean could move by a few tenths.
