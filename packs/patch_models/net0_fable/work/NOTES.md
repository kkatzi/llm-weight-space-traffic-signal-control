# NOTES: a hand-built "leave phase 2" detector for the trained network

## What was wrong with the original network

On the recorded states the trained network behaves like a good actuated controller: it switches at the
first allowed decision (green = 12 s) in phases 0, 1 and 3 and almost always picks a phase with waiting
cars. The one exception is phase 2 (EW through), the main-road phase, where it "rests": it stays in
phase 2 whenever the EW-through lanes are empty and the other queues are small.

That rest becomes a trap. In recorded network episode 2 the hour of demand ended with one car on N0,
one on N1 and one on W2. The network stayed in phase 2 for the whole 30-minute drain cap (about 350
consecutive decisions at t_green = 60 s), so three cars never left. That is about 1.45 veh-h of TTS
on a 35-55 veh-h episode (3 %), and it happened in 1 of the 6 recorded network episodes. During the
hour the same situation is harmless because new EW or N/S cars arrive and change the state; the
recorded data shows the network picking phase 2 with empty EW lanes and waiting cars elsewhere only
about 7 times outside that stuck tail.

Everything else I measured (green lengths, wasted-green decisions, choice of next phase versus the
phase with the most waiting cars) looked close to the tuned actuated controller or better, so I did
not try to change the normal control logic. The patch only removes the trap.

## What I built (246 parameters changed of 7428, limit 371)

Two existing units are repurposed; nothing else is touched apart from bias compensation.

**Layer-1 unit 13 = the detector.** Its pre-activation is

    60 * [ 3*(ok - 1) + 3*(phase2 - 1) + 0.3*(q_N0+q_N1+q_N2+q_E2+q_S0+q_S1+q_S2+q_W2)
           - 30*(d_E0+d_E1+d_W0+d_W1) - 2*(d_E0u+d_E1u+d_W0u+d_W1u) - 0.05 ]

so tanh is about +1 only when: minimum green has elapsed, the current green is phase 2, no vehicle is
on an EW-through stop-line lane (one car there gives -10), at most about one EW-through car is on the
upstream lanes (each upstream car gives -0.06), and at least one car is halting on a lane phase 2
does not serve (one car gives +0.1). Otherwise it is about -1. The ok/phase2 gates (-3 each) dominate
the largest possible queue term (8 lanes * 0.3 = 2.4), so the unit cannot fire in another phase or
before the minimum green. W1 row 13 (46 values) and b1[13] are set; the old column W2[:,13] is
zeroed, and b2 gets the old column times the old unit's mean activation on the recorded states so
the other layer-2 units keep their average input (64 values in b2).

**Layer-2 unit 5 = relay.** W2[5,:] is zeroed except W2[5,13] = 10, b2[5] = 0, so h2_5 = tanh(10*h1_13)
is +1 when firing and -1 otherwise. Its old output column W3[:,5] is replaced and b3 gets the old
column times the unit's old mean activation (compensation).

**Output.** W3[2,5] = -10 with b3[2] -= 10: logit 2 is lowered by 20 when the detector fires and by 0
otherwise. W3[0,5] = +0.5 with b3[0] += 0.5: logit 0 (NS through) is raised by 1 when firing. The
second term is a tie-break: without it a single car on N0, N1 or S1 produced a near-tie among logits
0/1/3 that resolved to phase 3, and from phase 3 the network returns to phase 2, giving an endless
2 -> 3 -> 2 loop with the car never served. With it, every single-car and two-car configuration I
enumerated is served within at most two switches.

Parameter count: 46 + 1 (W1 row, b1) + 64 (W2 column of unit 13, including the relay entry) + 63
(rest of W2 row 5) + 64 (b2 compensation) + 4 (W3 column) + 4 (b3) = 246.

## Checks

- `tools.edit_size`: 246 changed, 3.3 %.
- On all 13,644 recorded states the patched network never picks phase 2 where the detector condition
  holds (1,169 states; the original picked phase 2 in 358 of them, essentially the stuck tail), and it
  picks the same phase as the original on all other honoured decisions except 14 of 4,247. Those 14
  (90 over all states) were near-ties in the original: top-1 minus top-2 logit gap below 0.37, versus a
  median gap of 4.1. The 99th percentile of the logit change outside the detector's own terms is 0.27.
- Synthetic stuck states: a single halting car on each of the 8 non-EW-through lanes, two cars on one
  lane, and all pairs of single cars, always from phase 2 after the minimum green: the chosen phase
  serves a waiting car in every case, and from the follow-up phase the network either serves the rest
  or returns to phase 2 where the detector fires again toward the remaining car.
- The detector does not fire with one EW-through car at the stop line, with 1 or 3 EW-through cars
  upstream and a single car waiting elsewhere, before the minimum green, or in other phases.
- `tools.trainability` is essentially unchanged (median max-prob 0.980, no dead units); the
  detector unit is saturated by design (sat_l1 rises from 0.00003 to 0.016).
- Simulator: 4 of 10 calls used.
  1. original weights: mean 44.7 (low 34.7, nominal 44.9, high 54.4), no stranded vehicles;
  2. patch without tie-break: mean 44.7 (34.8 / 44.7 / 54.6);
  3. patch with the tie-break bias sign wrong (logit 0 lowered by 1 everywhere): mean 44.7
     (34.7 / 44.8 / 54.6), a bug found and fixed from the offline flip count;
  4. final weights: mean 44.8 (34.8 / 44.9 / 54.6), no stranded vehicles, 241-246 switches/episode.
  None of the 9 dev episodes hits the phase-2 trap with the original network, so the dev score cannot
  show the fix; the per-episode differences (within +-0.6 veh-h both ways) come from the few
  near-tie flips and are noise.

## What I expect

- On test episodes that would have ended in the phase-2 trap (about 1 in 6 of the recorded
  training-like episodes, 0 of 9 on dev, so perhaps 1 in 10), the patch saves about 1-1.5 veh-h each.
  Averaged over 30 test episodes that is roughly 0.1-0.25 veh-h, so the predictions are the dev values
  minus about 0.2.
- Otherwise the network is the original one, including its weaknesses under high demand.
- Risks: the 14 near-tie flips could by chance go the wrong way on some episode (zero-mean noise); the
  detector can switch away from phase 2 while an EW platoon is 2 or more cars upstream only if at
  least 2-3 cars are already halting elsewhere, which is the intended trade; if a stranded car were
  halting on an upstream lane only (not at a stop line) the detector would not see it, but I found no
  such state in the recordings.

## Files

- `build.py`: builds `weights.npz` (`python build.py 13 5`; `python build.py search` ranks unit pairs by
  side-effect flips). `check.py`, `check2.py`, `check3.py`: the offline checks above. `analyze*.py`:
  the data analysis that found the trap. `baseline.npz`: copy of the original weights used for call 1.
