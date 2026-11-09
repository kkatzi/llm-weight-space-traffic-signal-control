"""Build the L2W network from the L2 network by construction (no fitting).

The source network is applied to an L2-layout observation built from the L2W observation by a fixed affine map:
    x_L2 = M @ x_L2W + c          (M: 46 x 54, c: 46)
so  W1_L2W = W1_L2 @ M,  b1_L2W = b1_L2 + W1_L2 @ c,  and W2, b2, W3, b3 are copied unchanged.

M (lane correspondence):
  * N/S lanes: same lanes; stop-line q/d scaled by NS_SCALE (L2W's N/S stop-line lanes are 19.2 m, not 22.4 m).
  * E/W stop line: E0 -> E0; mean(E1, E2) -> E1 (straight); E3 (left pocket) -> E2 (left pocket).
  * E/W upstream: E0u -> E0u; mean(E1u, E2u) -> E1u.
  * phase one-hot, ok, t_green unchanged.
Unstick term (M's t_green column + c): the q and d inputs of every stop-line lane that phase 2 (EW through) does NOT
serve (N0 N1 N2 S0 S1 S2 and the two left pockets) get  KAPPA * (t_green/60 - 0.2)  added. This is exactly zero at
t_green = 12 s, where the source network takes nearly all of its honoured decisions, and grows only when a green is
held longer. It stops the source network's habit of holding EW-through green indefinitely while 1-2 cars wait on
another movement (see NOTES.md).
"""
import numpy as np
from common import *

NS_SCALE = 0.93   # between 19.2/22.4 = 0.857 (matches 1-2 car counts) and 1.0 (matches a full lane)
KAPPA = 0.75      # unstick strength: +0.25 fake occupancy at t_green 32 s, +0.6 at 60 s (t_green is clipped at 60 s)
UNSTICK_LANES_L2 = ['N0', 'N1', 'N2', 'S0', 'S1', 'S2', 'E2', 'W2']   # L2 names (E2/W2 = left pockets)


def affine_map():
    M = mapping(ns_scale=NS_SCALE, mid=(0.5, 0.5), up=(0.5, 0.5))
    c = np.zeros(46)
    for lane in UNSTICK_LANES_L2:
        for k in 'qd':
            M[I2[f'{k}_{lane}'], IW['t_green']] += KAPPA
            c[I2[f'{k}_{lane}']] -= 0.2 * KAPPA
    return M, c


if __name__ == '__main__':
    src = tools.load('../source_weights_L2.npz')
    M, c = affine_map()
    w = build(M, src)
    w['b1'] = src['b1'] + src['W1'] @ c
    out = os.path.join(HERE, 'weights.npz')
    tools.save(w, out)
    np.savez(os.path.join(HERE, 'affine_map.npz'), M=M, c=c)

    z = np.load(out); s = np.load(os.path.join(os.path.dirname(HERE), 'source_weights_L2.npz'))
    for k in ['W2', 'b2', 'W3', 'b3']:
        assert z[k].dtype == s[k].dtype and np.array_equal(z[k], s[k]), k
    print('saved', out, '- W2 b2 W3 b3 bit-identical to source')
