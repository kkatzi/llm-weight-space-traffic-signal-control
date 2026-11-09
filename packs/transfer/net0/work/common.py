"""Shared helpers for the L2 -> L2W transfer (construction only, no fitting)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

N2 = ['q_N0','q_N1','q_N2','q_E0','q_E1','q_E2','q_S0','q_S1','q_S2','q_W0','q_W1','q_W2',
      'd_N0','d_N1','d_N2','d_E0','d_E1','d_E2','d_S0','d_S1','d_S2','d_W0','d_W1','d_W2',
      'q_N0u','q_N1u','q_E0u','q_E1u','q_S0u','q_S1u','q_W0u','q_W1u',
      'd_N0u','d_N1u','d_E0u','d_E1u','d_S0u','d_S1u','d_W0u','d_W1u',
      'phase0','phase1','phase2','phase3','ok','t_green']
NW = tools.names()
I2 = {n: i for i, n in enumerate(N2)}
IW = {n: i for i, n in enumerate(NW)}


def mapping(ns_scale=1.0, mid=(0.5, 0.5), up=(0.5, 0.5), qd_same=True):
    """M (46 x 54): x_L2 = M @ x_L2W.
    ns_scale: factor applied to N/S stop-line lane features (shorter lanes in L2W).
    mid: weights of L2W (E1, E2) forming L2's E1 (straight lane).
    up:  weights of L2W (E1u, E2u) forming L2's E1u (upstream straight+left lane)."""
    M = np.zeros((46, 54))
    def s(dst, src, v=1.0):
        M[I2[dst], IW[src]] += v
    for k in ['q', 'd']:
        for a in ['N', 'S']:
            for l in '012':
                s(f'{k}_{a}{l}', f'{k}_{a}{l}', ns_scale)
            for l in '01':
                s(f'{k}_{a}{l}u', f'{k}_{a}{l}u')
        for a in ['E', 'W']:
            s(f'{k}_{a}0', f'{k}_{a}0')
            s(f'{k}_{a}1', f'{k}_{a}1', mid[0]); s(f'{k}_{a}1', f'{k}_{a}2', mid[1])
            s(f'{k}_{a}2', f'{k}_{a}3')
            s(f'{k}_{a}0u', f'{k}_{a}0u')
            s(f'{k}_{a}1u', f'{k}_{a}1u', up[0]); s(f'{k}_{a}1u', f'{k}_{a}2u', up[1])
    for n in ['phase0', 'phase1', 'phase2', 'phase3', 'ok', 't_green']:
        s(n, n)
    return M


def build(M, src=None, b1_shift=None):
    src = tools.load(os.path.join(os.path.dirname(HERE), 'source_weights_L2.npz')) if src is None else src
    w = {k: src[k].copy() for k in ['W2', 'b2', 'W3', 'b3']}
    w['W1'] = src['W1'] @ M
    w['b1'] = src['b1'].copy() if b1_shift is None else src['b1'] + b1_shift
    return w


def plain(ns_scale=0.93):
    """The plain lane-mapping transfer (no unstick term) used during exploration."""
    return build(mapping(ns_scale=ns_scale))
