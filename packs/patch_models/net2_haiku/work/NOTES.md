# NOTES: hand-built max-pressure readout (does NOT beat the start network)

## Result in one line
The delivered `weights.npz` scores **46.5 veh-h** on the 9 dev episodes, worse than the start network (**44.8**)
and the actuated reference (45.3). Per scenario: low 34.2 (start 34.5), nominal 44.9 (start 44.9), high 60.4
(start 55.0). The edit is a valid, budget-compliant network, but it is not an improvement.

## What I built
The start network switches phase whenever it is allowed (98% of recorded states with the minimum green met
request a switch, and it switches about 243 times per episode). Its choice of next phase agrees with the
max-queue rule on only about half of those states. I tried to replace the phase choice with a simple,
interpretable rule: **max-pressure**, i.e. pick the phase whose stop-line lanes hold the most queued traffic.

Pressures (queue fraction + 0.5 x density fraction of the phase's stop-line lanes):
- P0 NS_through = N0 + N1 + S0 + S1
- P1 NS_left = N2 + S2
- P2 EW_through = E0 + E1 + W0 + W1
- P3 EW_left = E2 + W2

The rule is wired into the network by rewriting three hidden units per layer, chosen as the units with the
smallest outgoing weight mass (layer 1: units 50, 42, 62 of 64; layer 2: units 1, 25, 2):
- Layer 1, unit I_k: `h1 = tanh(0.5 * (P_k - P_3))`, k = 0,1,2 (row rewritten, bias 0).
- Layer 2, unit J_k: `h2 = tanh(h1[I_k])` (row rewritten to a single weight of 1, bias 0).
- Output: `logit_k = 100 * h2[J_k]` for k = 0,1,2; `logit_3 = 0`; old `b3` kept.

Every step is monotone, so the argmax equals the max-pressure choice (ties are broken by the untouched old units
and `b3`). Everything was set by hand; nothing was fitted, and no optimiser or regression touched the weights.

## Parameters changed
348 of 7428 (4.7%), checked with `tools.edit_size` (limit 371). The 348 are the three rewritten layer-1 rows
(46 weights and bias each), the three rewritten layer-2 rows (64 weights and bias each), and 12 output-weight
entries. Everything else is untouched. Built by `work/build_maxpressure.py`.

## Simulator calls (3 of 10 used; all logged in `eval_log.jsonl`)
1. BETA = 0 (this delivered file): **46.5** (low 34.2, nominal 44.9, high 60.4). About 246 switches per episode.
2. BETA = 1 (stay bonus of 1 pressure unit for the current green): **47.1** (low 34.4, nominal 45.5, high 61.4).
3. BETA = 3: **57.3**, with 9 to 24 vehicles left at the drain cap. A bonus this large starves the other phases.

The sweep was a check on one design choice, not a search. The stay bonus only made things worse, so I kept BETA = 0.
The alternative of choosing a BETA by further sim calls was not pursued.

## Checks run (offline, on `states.npz`)
- Agreement of the edited network with the max-pressure rule (stay-bonus version): 0.90. The gap is mostly
  states where pressures are tied or nearly tied, where the untouched units decide.
- Agreement with the start network: 0.68 of recorded states.
- `tools.trainability`: median max-probability 1.0, 90% of states above 0.99, median logit gap 29. The network
  is nearly deterministic, so PPO would have almost no exploration to continue from. Gate-like saturation is
  mild (layer-2 saturation 0.095).

## Why it fails, as far as I can tell
- Max-pressure ignores the phase currently green in the decision. The start network already switches at almost
  every allowed moment, so its extra value must be in phase choice, and a pure queue rule does not capture it.
- At high demand, switching more often costs 3 s of yellow each time. Adding a stay bonus helped only slightly
  (BETA 0 to 1), and a larger bonus starves other phases (BETA 3). The queue measure is not a good enough
  proxy for when to switch.
- The untouched hidden units still add noise. Changing units with low outgoing weight reduces that, but
  each rewritten unit still feeds the other 61 layer-2 units through the old `W2` columns, so their outputs
  shift. I did not budget for zeroing those columns.

## What I expect to go wrong on the held-out test
- High demand: about 60 veh-h in dev, which is well above the start network. I expect the test to show the same
  gap, so the edit is worse overall.
- Low and nominal: close to the start network, with a small expected spread.

## Predictions (`predictions.json`)
Dev means for this exact file: low 34.2, nominal 44.9, high 60.4. These are used directly as the test
prediction for the file, though the held-out seeds will differ.

## Files
- `build_maxpressure.py`: builds `weights.npz` from `start_weights.npz`. `BETA` is set to 0.
- `weights.npz`: the delivered network (sha1 054642ac1e07..., matches eval_log call 1).
- `weights_beta3.npz` is not kept (deleted before finishing). Its result (57.3) is in `eval_log.jsonl`, call 3.
  The BETA = 1 candidate (call 2) is not saved as a file; `build_maxpressure.py` with `BETA = 1.0` regenerates it.
