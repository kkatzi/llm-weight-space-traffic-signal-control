# NOTES: patch of start_weights.npz (net2_b0)

## What I found in the original network

I ran forward passes over the 4432 recorded decisions from the network's own episodes (`explore*.py`).

- The policy is almost pure minimum green. At an honoured decision (green age 12 s, `ok`=1) it switches 97% of
  the time, and it picks the next phase acyclically, usually a phase with the most vehicles. A Webster-style
  estimate says short cycles suit these demand levels, so I left this behaviour alone.
- **Flaw:** about 42 times in 6 episodes it extended a green. 39 of these were on **EW_left (phase 3)**, and in 17
  of them **both EW left pockets (E2, W2) were completely empty** while E/W through vehicles sat at a red. Some
  of these holds repeat 3 to 5 times in a row: ep1 t=40-60 and t=1545-1555, ep2 t=2690-2700. That is 15-25 s of
  green with nobody to serve.
- The same flaw shows up as a switch: at the start of every episode (t=10, junction empty, E/W cars approaching)
  it switches to **NS_left (phase 1) with both NS pockets empty**. It does the same near the end of the episode,
  when the only vehicle left is waiting in an EW pocket.

## What I built

Two hard gates. Each overrides only the logit of a left-turn phase whose two pockets are empty:

- **EW gate:** `ok`=1 and `d_E2`=0 and `d_W2`=0 → logit 3 (EW_left) is lowered by 40.
- **NS gate:** `ok`=1 and `d_N2`=0 and `d_S2`=0 → logit 1 (NS_left) is lowered by 40.

When a gate is on, the network still chooses among the other three phases by its own logits. Typically it now
switches from a dead EW_left green to EW_through, which has halted cars. A pocket holds only about 3 cars and sits
at the stop line, so "pocket empty" really does mean "no left-turner can use this green". A left-turner stuck in
the shared upstream lane behind a straight car could not reach the pocket anyway.

## Which units and weights do what (298 parameters changed, limit 371)

| unit | role | weights set |
|---|---|---|
| L1 unit 50 | EW detector: pre-activation = 20·ok − 60·d_E2 − 60·d_W2 − 10, which is ±10, so the output is exactly ±1 | W1 row (46, rest zeroed), b1 |
| L1 unit 38 | NS detector: the same with d_N2, d_S2 | W1 row, b1 |
| L2 unit 1 | EW relay: W2[1,50] = 20. Its other inputs sum to at most ±11, so it is saturated at −1 (off) or +1 (on) | W2[1,50], b2[1] (centred) |
| L2 unit 56 | NS relay: W2[56,38] = 20 | W2[56,38], b2[56] |
| output | W3[3,1] = −20 and W3[1,56] = −20, other entries of these two columns are 0; b3[3] and b3[1] −20 cancel the "off" state exactly | W3 columns 1 and 56, b3 |
| compensation | the old outputs of L1 units 50 and 38 were removed from every L2 unit (W2 columns zeroed) and replaced by their mean in b2. The old contributions of L2 units 1 and 56 were replaced by their mean in b3 | W2 cols 50 and 38, b2, b3 |

I chose the units by single-unit ablation: replace a unit with its mean activation (from the network's own
episodes) and count how many honoured decisions change. I took the two least influential units in each layer.
Means, ablation counts and medians are statistics of recorded activations. Nothing was fitted.

Counts: W1 92, b1 2, W2 128, b2 64, W3 8, b3 4, total **298**.

## Checks run (`checks.py`, `show_changes.py`, `check_choice.py`)

- `tools.edit_size` on the saved float32 file: 298 changed of 7428.
- The detectors are exactly ±1 and fire if and only if (`ok` and both pockets empty), on all 13306 recorded states
  and on 200k random synthetic observations. The relay pre-activations are within ±(17-23), always saturated, and
  the bound holds for any input.
- After the edit, no honoured decision picks phase 3 with the EW pockets empty or phase 1 with the NS pockets
  empty. Before the edit this happened 42 and 57 times across all controllers' states.
- Network's own episodes: 39 honoured decisions change. 34 are the intended ones: 15 are a dead EW_left hold that
  now goes to EW_through, and 28 of the 39 new picks have vehicles at their stop line. 5 are side effects of the
  removed units, all at near-ties (logit gap < 0.6). Where the gates are off, logits move by at most 0.69 (mean
  0.12).
- On the fixed and actuated states (off-distribution for the network) there are 16 and 15 side-effect changes
  out of about 1500-2000 honoured decisions.
- `tools.trainability`: max_abs_logit rises to 50 because of the gate. The other numbers are almost unchanged.

## What I expect to go wrong / uncertainty

- The gain is small. About 3 dead holds per episode, plus one wasted NS_left green at the start and some at the
  end. I estimate 0.05-0.15 veh-h per episode, below the episode-to-episode noise. Any changed decision changes
  the rest of the trajectory, so individual test episodes can come out worse.
- The 5 side-effect changes per 6 episodes (removed units) are close calls. A few of them pick the phase with
  fewer halted cars, which could cost a little.
- A left-turner about to enter an empty pocket (within about 2 s) will now sometimes miss its green and wait
  about one cycle. The original network may have been anticipating such cars in a few of its holds.
- I did not touch the min-green or phase-order logic. If larger gains exist, for example in green splits at high
  demand, this patch does not capture them.

Predictions (mean test TTS, veh-h): low 34.4, nominal 44.8, high 54.9 (original on dev: 34.5 / 44.9 / 55.0).
