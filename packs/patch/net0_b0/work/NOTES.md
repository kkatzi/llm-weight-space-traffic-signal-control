# Patch notes: empty-phase veto

## What I found

In the recorded start-network episodes, episode 2 never empties. Demand stops at 3600 s. From then on the network holds
EW_through (phase 2) green with zero E/W traffic, and three vehicles wait on N0, N1 and W2 until the 5400 s cap.
The original logits in that state are [-1.7, -4.0, **2.09**, 1.96]. This costs about 1.5 veh-h in that one
episode.

I traced decision chains on synthetic sparse states: every set of waiting stop-line lanes (1–3 vehicles each),
starting from every phase. These traces use forward passes only, with no traffic model. The original net deadlocks
in 87 configurations. In every one it ends holding phase 2 at t_green = 60 s. Typical cases are a single vehicle on
N0, N1, S0, S1, S2 or W2. Phase 2 is the "parking" phase: when the junction is nearly empty, the net goes there
and stays, even when another approach has a waiting car.

Under fixed or actuated control, 3 of 6 and 2 of 6 recorded episodes pass through end-of-episode states where the
original net would make this mistake. So I put the deadlock risk at roughly 15–30 % of episodes.

During the demand hour (t < 3600 s) the net never picks a phase with no vehicles on its approach (0 cases). The
failure is specific to clearing.

## What I built

A veto. If phase q has **no vehicle at all** on the lanes that feed it, its logit drops by V = 40. "The lanes that
feed it" means its stop-line lanes plus their upstream lanes (300 m in total):

| phase | density inputs checked |
|---|---|
| 0 NS_through | d_N0 d_N1 d_S0 d_S1 d_N0u d_N1u d_S0u d_S1u |
| 1 NS_left | d_N2 d_S2 d_N1u d_S1u |
| 2 EW_through | d_E0 d_E1 d_W0 d_W1 d_E0u d_E1u d_W0u d_W1u |
| 3 EW_left | d_E2 d_W2 d_E1u d_W1u |

The veto covers all four phases. If the whole junction is empty, every logit drops by the same amount, so the
original choice stands. Otherwise the argmax is always a phase that has a vehicle. Each switch therefore serves
someone, and a junction with no new arrivals always clears.

How it is wired (per phase q; units chosen as below):

- **Layer-1 detector** i_q ∈ {57, 32, 60, 59} for phases {0, 1, 2, 3}. Its weights on the density inputs above are
  set to -K (K = 2000), with b1 = K·θ, where θ is half an upstream vehicle (0.5 · 7.5/256). The other 30–38 row
  entries are left alone. Their total |weight| is under 9, while the margin is K·θ ≈ 29. The unit is therefore
  exactly +1 when the phase is empty and -1 otherwise, for any input. I checked this on recorded, uniform-random,
  all-zero and all-one inputs (minimum |pre-activation| 27).
- Its **old W2 column is zeroed**, and its old mean contribution `W2[:, i]·mean(h1_i)` moves into b2. One entry is
  set to B = 25 into a layer-2 unit j_q.
- **Layer-2 gate** j_q ∈ {37, 35, 39, 45}. The pre-activation is the old pre-activation (|·| ≤ 12.5) plus 25·Z_q,
  so the unit saturates and copies the detector (±1). Minimum |pre-activation| is 21 on all tested inputs.
- **W3 column** of j_q is zero except -V/2 = -20 on logit q. b3 absorbs the constant +V/2 and the old mean
  contribution of j_q.

Checked numerically (max error 4e-7): patched logits equal the original network with those 8 units frozen at their
mean, minus 40 on each empty phase.

**Choosing which units to sacrifice.** I picked them greedily, using a statistic: freezing them at their mean must
change as few original decisions as possible. The states used were recorded honoured decisions before 3600 s
(weight 0.6), recorded honoured decisions after 3600 s (0.2) and random sparse end-of-episode-like states (0.2).
No weights were fitted.

**Parameters changed: 368 of 7428** (limit 371):

| block | changed |
|---|---|
| W1 | 24 |
| b1 | 4 |
| W2 columns | 4 × 64 = 256 |
| b2 | 64 |
| W3 columns | 4 × 4 = 16 |
| b3 | 4 |

## Checks run (all forward passes; scripts in work/)

- `verify.py`:
  - Edit size: 368.
  - Honoured decisions on recorded states: 400 of 5416 change.
    - 369 are vetoes, almost all in the deadlock episode.
    - 31 are collateral, all at t < 3600 and all near-ties: the original top-2 logit gap is at most 0.36, with
      median 0.10 (the typical gap is 4.6).
  - Static sparse traces: 0 failures (the original has 87).
  - 200k random sparse states: never picks an empty phase while another phase has vehicles. Agreement with the
    original elsewhere is 99.2%.
  - The recorded deadlock state now picks phase 3, then phase 0.
- `verify_extra.py`:
  - All-empty junction: the same choice as the original in 50 of 52 (phase, t_green) states.
  - A single upstream vehicle is always served by a phase that serves it.
- `deadlock_rate.py`: how often recorded episode endings are deadlock-prone (see above).

## What I expect / what could go wrong

- Main effect: no end-of-episode deadlocks. Each deadlock costs about 0.5 veh-h per stuck vehicle. With a deadlock
  rate of 15–30 % of episodes, I expect an average gain of about 0.2–0.3 veh-h per episode, roughly equal across
  scenarios. If the test seeds produce no deadlocks, the gain is about 0. Test-to-dev seed noise (±0.5 veh-h) can
  outweigh it.
- Collateral: freezing 8 units flips about 0.6 % of honoured decisions during the demand hour, all near-ties. I
  expect this to be noise-level, but it could cost or gain about ±0.1 veh-h.
- The veto counts every vehicle in the 300 m approach, so it never fires while a vehicle is still approaching that
  phase. It only removes picks of phases that have nobody to serve. One assumption is untested: that density
  counts are exact integers (they are in all recorded data). A vehicle inside the junction box is not counted. It
  clears on its own, so this should be harmless.
- Large weights (K = 2000, 20 in W3) make max |logit| about 48 and saturate the detectors. That is irrelevant for
  greedy evaluation, but bad for any further PPO training.
- Nothing else about the policy changed: it still switches mostly at the 12 s minimum green, like the original.
