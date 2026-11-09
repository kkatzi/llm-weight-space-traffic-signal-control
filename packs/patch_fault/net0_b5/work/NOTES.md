# Repairing the network for the dead W2 detector

## What was wrong

The W2 detector (west left-turn pocket) is dead, so x[11] = q_W2 and x[23] = d_W2 always read 0. The network
learned to depend on these inputs a lot. In the recorded states, setting W2 to "full" moves the share of honoured
decisions that pick phase 3 (EW_left) from 17% to 65%. With W2 at 0, phase 3 is chosen only when the *east*
pocket E2 has vehicles (E2 = 0 means phase 3 is never picked).

The recorded faulted episodes show what this costs. In 2 of the 6 start-network episodes the junction **deadlocks
during the drain**: phase 2 stays green from about t = 3700 s to the 5400 s cap. Hidden W left-turners sit in the
W2 pocket, and in one episode 6 more queue on W1u behind them (q_W1u = 0.18). Since E2 is empty, the network
never switches to phase 3. During normal running, W left-turners are also served late, which backs W1u up.

## What I built (320 of 7428 parameters changed; all in W1 and b1)

**1. An estimate of the dead inputs, folded into layer 1.** Layer 1 is linear before its tanh. So feeding the
network an estimate `q_hat = c + Σ a_j x_j` in place of q_W2 is the same as adding `W1[:,11]·a_j` to column j
(and `W1[:,11]·c` to b1); the same goes for d_W2 through `W1[:,23]`. I checked this is exact (logit difference
about 1e-15). All coefficients are set by hand:

    q_hat = 0.04 + 0.8·q_E2 + 3·q_W1u + 0.4·t_green − 0.2·phase3
    d_hat = 0.12 + 0.8·q_E2 + 3·q_W1u + 0.4·t_green − 0.2·phase3

- **q_E2 (0.8)**: E and W have the same demand, and their pockets are emptied together by phase 3, so the opposite
  pocket is the best single proxy. I shrank it a little toward a small constant. In the N/S analog the opposite
  pockets correlate at only about 0.35. d_hat also uses q_E2 rather than d_E2, to free a column; its constant
  is 0.08 higher because d_E2 runs about 0.08 above q_E2.
- **q_W1u (3)**: a halted queue on W1u (which feeds W1 and W2) is spillback, and it means W2 is full. On the
  analog lanes, P(pocket full | own upstream queue > 0) is 0.6–0.8. The coefficient makes 6 queued cars count
  as about +0.5.
- **t_green (0.4)**: the pocket fills while phase 3 is not green. W left-turners arrive at about 0.06 veh/s, so
  after 60 s the pocket would be about full. This term is also the drain timer: after about 60 s of phase 2 with
  nothing visible, q_hat reaches about 0.44, which is above the roughly 0.40 at which the network picks phase 3.
  That breaks the deadlock.
- **phase3 (−0.2)**: cancels most of the t_green growth while phase 3 is itself green, so phase 3 cannot latch on
  (otherwise an empty phase 3 at 60 s would keep choosing itself).

**2. A lone-car timer for N/S through lanes** (on t_green and b1, which were already changed, so no extra
parameters). The original network ignores a single car on N0, N1, S0 or S1: it needs 2–3 cars before it serves
phase 0. During the drain, nothing more arrives, so such a car waits until the cap. This is a flaw of the
original network that the fault exposes more often; designs A and C left 1–2 cars at the cap in 2 of 9 dev
episodes each. I add phantom N/S through demand of `0.4·(t_green − 0.3)` to q and d of N0, N1, S0 and S1. It is
about 0 at the usual 12–17 s decisions and +0.28 after 60 s of green. That is enough for one real car to win
phase 0, but not enough on its own: an empty junction still goes to phase 3, so hidden W2 cars get served.

Changed parameters: b1 (64), and the W1 columns q_E2, q_W1u, t_green and phase3 (4 × 64). W1[:,11] and
W1[:,23] were left alone (they multiply inputs that are always 0). Layers 2 and 3 are untouched.

## Checks run

- Exactness of the folding: logits of the folded network equal those of the original network fed the imputed
  inputs.
- Analog test (`analog.py`, `analog2.py`): hide the real N2 or S2 values and rebuild them with the same kind of
  scheme. The start network's decisions then agree with its decisions on the real values about as well for
  "copy" as for "half". This only showed that the obvious schemes are about equal.
- Recorded deadlock states, all-empty states and lone-car states (`drain_check.py`). Toy drain replay
  (`drain_sim.py`: 600 random leftover patterns, with the signal logic from TASK.md). Situations left stuck:
  start 58%, A 52%, final 2.5%. Normal-operation decisions of the final design agree with A and C on about 96%
  of honoured recorded states.
- Simulator, all 5 calls (dev mean TTS; low / nominal / high; vehicles left at the cap):

| call | design | mean | low | nominal | high | left at cap |
|---|---|---|---|---|---|---|
| 1 | A: 0.5·E2, own d_E2 term, spill, t_green | 45.2 | 34.8 | 45.2 | 55.7 | 3 |
| 2 | C: like A, 0.8·E2 | 45.2 | 34.8 | 45.9 | 54.9 | 3 |
| 3 | E: 0.65·E2, phase3 term, lone-car timer | 45.1 | 34.8 | 45.3 | 55.3 | 0 |
| 4 | **E8: E with 0.8·E2 (delivered)** | **44.9** | 34.9 | 45.3 | 54.4 | 0 |
| 5 | E10: E with 1.0·E2 | 45.2 | 34.9 | 45.4 | 55.2 | 0 |

  For reference: start network with the fault 48.0 (35.5 / 46.8 / 61.8); same network with every detector
  working 44.7 (34.7 / 44.9 / 54.4).
- `work/weights.npz` is the same file as the evaluated `w_E8.npz`. `tools.edit_size` gives 320 changed.

## What I expect to go wrong

- The 0.8 E2 weight was the best of three tried (0.65, 0.8, 1.0) on only 9 dev episodes. Part of its high-demand
  edge (54.4) is likely seed luck, so I expect about 55 on the test set.
- The drain timers rely on fairly thin decision margins. The lone-car timer only works for a phantom strength of
  about 0.26–0.30 at 60 s. If real end-of-run states differ from my synthetic ones (moving cars, partial
  counts), some episodes can still leave a car or two at the cap; each costs about 0.5 veh-h. A single car on S1
  alone is still never served. Empty phase 0 or phase 3 held for about 60 s would stick; in the recorded states
  the network never holds phase 0 or 3 past its first honoured decision, so I expect this to be rare.
- Using E2 as a proxy is wrong whenever the two pockets differ by chance. At low demand the estimate over-serves
  phase 3 slightly (low demand is 0.1–0.2 worse than with working detectors).
- The network is more confident than before (median max-probability 0.977 → 0.987). This only matters if it is
  ever trained further.
