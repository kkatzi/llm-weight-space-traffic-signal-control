# Repair for the failed W2 detector: notes

Edited network: `work/weights.npz` (built by `work/build.py`, checked by `work/checks.py`).
**345 of 7428 parameters changed** (W1 144, b1 2, W2 190, b2 1, W3 4, b3 4). Every value was set by
construction; nothing was fitted.

## What goes wrong with the failed detector

The recorded start-network episodes already have the failed detector. Two things stand out in them.

1. **Deadlock at the end of the episode.** Two of the six start-network episodes, and one actuated episode, ran to the
   90-minute cap. After demand stops, every visible input is 0 (or only W1u holds halted cars) and the network sits in
   EW_through (phase 2) for 25+ minutes. The cars left in the invisible W2 pocket, and in one episode about 6 more
   left-turners queued on W1u behind them, are never served. With nothing visible, the network's "parking" choice is
   phase 2. A halted queue on W1u also reads to it as "through traffic, keep phase 2".
2. **W2 starves during the hour.** The network calls EW_left only from E2, so W2 overflows and its left-turners block
   W1u, the upstream lane shared with W through traffic. On the W side, phase 2 green with a halted car on W1u happens
   418 times during demand (plus 357 times in the deadlock). On the visible E/N/S sides that same signature means the
   left pocket is full in 98–100 % of states. Seeing W2 full, the original network would pick EW_left in 92 % of those
   states. With W2 at 0 it does so in only 50 %.

## What I built (four parts)

1. **Mirror stand-in (62 W1 entries in column 5, plus the overwritten entries of units 4 and 28).**
   `W1[:,5] += W1[:,11] + W1[:,23]`: every first-layer unit now reads q_E2 as if it were also q_W2 and d_W2. W2 and
   E2 have the same demand and are served by the same phase, so E2 is the natural stand-in.
   - **Mirror test on N/S** (both pockets visible): hide S2 or N2 and replace it by the other pocket's halting count.
     Decision agreement with the true input rises from 0.90 / 0.87 (zero) to 0.92 / 0.89. Using q only was as good as
     q and d, so one column is enough.
   - **Calibration:** hiding S2 drops the NS_left share from 18.1 % to 5.9 %. The stand-in restores it to 19.1 %
     (21.3 % when N2 is hidden).
   - **Ruled out:** the mean, a shrunk stand-in, and a linear term on q_W1u were no better.
2. **IDLE detector: h1 unit 4, repurposed** (38 W1 entries and b1).
   - pre = 3 − 300·(sum of all lane inputs except q_W1u, d_W1u, x11, x23) − 300·(d_W1u − q_W1u).
   - It is +1 only when no vehicle is visible except cars halted on W1u (with W1 empty, those can only be
     left-turners blocked by a full W2). Any single other vehicle drives it to −1: the smallest nonzero input is
     1/34.1, which gives −8.8.
   - The old phase/ok/t_green weights (|sum| ≤ 0.8) were kept; the dead x11/x23 entries were left untouched.
   - Unit 4 had mean activation −0.02 and the least influence of all units (ablating it changes 0.2 % of decisions).
3. **BLOCK detector: h1 unit 28, repurposed** (44 W1 entries and b1).
   - pre = −205 + 202·phase2 + 200·q_W1u.
   - It is +1 only when EW_through is green and at least one car halts on W1u, −1 otherwise. It never fires in other
     phases. In phase 3 that matters: halted W1u cars there are straight-goers, and firing would hold EW_left forever.
4. **RELAY: h2 unit 37, repurposed** (62 W2 row entries, b2, W3 column, b3).
   - It reads only units 4 (gain 3) and 28 (gain 1.778). Its baseline pre-activation is −3.
   - Its W3 column is (0, 0, 0, 4), so it adds only to the EW_left logit.
   - b3 absorbs its baseline (−0.995·4) and the mean of its old contribution.
   - Push when IDLE fires: about +8 logits. When BLOCK alone fires: +6.
   - The push of 6 comes from a forward-pass statistic (`diag_block_push.py`). On blockage states, switching W2 from
     the stand-in to "full" raises the original network's EW_left margin by 7.4 (E2 empty), 4.4 (one car),
     1.3 (two) and 0 (three). The weighted average where the stand-in under-reports is 5.4; I rounded up to 6 because
     under-serving a blocked W1u is the costlier mistake.
   - The old W2 columns of units 4 and 28 were zeroed except the relay row, which removes their old effect.

## Checks run (forward passes only; `work/checks.py`)

- **Edit count:** `tools.edit_size` on the saved float32 file gives 345 ≤ 371.
- **IDLE activations:** fires on 1140 recorded states. All of them are empty apart from halted W1u cars, none misses,
  and none falls in 300–3600 s (demand never empties the junction). Off-state |h| ≥ 0.99998. With exactly one
  visible vehicle anywhere, h ≤ −0.99993.
- **BLOCK activations:** fires exactly on the recorded states with phase 2 and q_W1u > 0 (1409 states).
- **Synthetic idle states** (all 4 phases × green times 2–120 s, with 0/3/6/20 halted cars on W1u): every honoured one
  picks EW_left, with a margin of at least 2.99.
- **Recorded deadlock states:** 339 states (only W1u halted) went from 339× phase 2 to 339× EW_left. 770 all-empty
  tail states went from mostly phase 2 to all EW_left.
- **In-hour blockage states (418):** decisions are now [25, 11, 0, 382]. The original net seeing W2 full gives
  [22, 11, 0, 385], so agreement is 0.95. The faulty original gave [115, 81, 14, 208].
- **All other honoured states (4615):** 99.5 % agree with the original network fed the stand-in. The three
  repurposed units cost about 0.5 % of decisions. They agree 81 % with the faulty original; the stand-in is meant to
  change those.
- **Share of in-hour honoured choices:** EW_left goes from 21 % to 41 %. On the N/S analog, hiding a pocket cuts the
  left share about 3×, so a large rebound is expected. It is measured on states recorded under the faulty policy,
  where E2 is often full, so the closed-loop share should be lower.

## What I expect to go wrong

- **E2 is a noisy stand-in.** Mirror pockets correlate only about 0.35–0.55. When E2 is full and W2 is not, EW_left
  is called with a single active approach. When W2 holds 1–2 cars and E2 is empty, they wait until E2 fills or W2
  overflows; the second case is then caught by BLOCK.
- **BLOCK is a constant push.** It over-serves EW_left when E2 is already full and the NS queues are large. That is a
  few percent of blockage states, a small cost. It can also fire about 12 s into a phase-2 green while a long W1u
  queue is still discharging, but W2 is then full anyway (98.6 % on the E-side analog).
- **IDLE parks the junction in EW_left** when nothing is visible. That costs a yellow plus at most 10 s for the first
  arrival. It only matters in the first seconds of an episode and in the clearing tail, never during demand.
- **Repurposing three units** changes about 0.5 % of busy-state decisions in unknown directions.
- **PPO trainability got worse** (large weights K = 300 and A = 200, saturated units, max |logit| 13.6). That does not
  matter here, because no further training happens.
- **The predictions are guesses** from the dev references, with no simulation. I expect most of the faulty-vs-healthy
  gap to close: the deadlock fix accounts for most of the low-demand gap, and the stand-in plus BLOCK for much of the
  high-demand gap. I do not expect to fully reach the healthy network.
