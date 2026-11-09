"""build.py: hand-constructed repair of start_weights.npz for the failed W2 detector.

Part A (imputation): the W2 pocket is served by the same phase (3) as the opposing E2 pocket and gets the same
    demand, so feed q_E2 / d_E2 into the weights that used to read q_W2 / d_W2:
        W1[:, q_E2] += alpha * W1[:, q_W2];  W1[:, d_E2] += alpha * W1[:, d_W2]
    (optionally also a spill-back term from q_W1u, see GAMMA_W1U).
Part B (deadlock breaker / drain rotation): 4 repurposed layer-1 units u_p (p = phase), each ON only when
    phase p is green, ok == 1, phase p's own stop-line lanes are empty and the junction is nearly empty
    (or the green has lasted very long). Each u_p drives one repurposed layer-2 unit r_p that adds a large bonus
    to the logit of the next phase (p+1) % 4. The shift of the pinned units' resting value (-1) is compensated
    in b2 / b3 with the units' mean activation on the recorded states.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
import numpy as np
import tools

names = tools.names()
ix = {n: i for i, n in enumerate(names)}

STOP = ['N0', 'N1', 'N2', 'E0', 'E1', 'E2', 'S0', 'S1', 'S2', 'W0', 'W1']          # W2 is dead
UP = ['N0u', 'N1u', 'E0u', 'E1u', 'S0u', 'S1u', 'W0u', 'W1u']
OWN = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2']}
CAP_STOP, CAP_UP = 22.4 / 7.5, 256.0 / 7.5        # vehicles per unit of density


def build(alpha=1.0, gamma_w1u=0.0, breaker=True, J=(50, 42, 48, 51), K=(25, 14, 52, 63),
          A=30.0, s=3.0, Kveh=3.0, gam=4.0, own_extra=6.0, G=10.0, B=20.0, verbose=True):
    w0 = tools.load(os.path.join(PACK, 'start_weights.npz'))
    w = {k: v.copy() for k, v in w0.items()}
    X = tools.states()['obs'].astype(np.float64)

    # ---------- Part A: imputation of the dead inputs ----------
    if alpha:
        w['W1'][:, ix['q_E2']] += alpha * w0['W1'][:, ix['q_W2']]
        w['W1'][:, ix['d_E2']] += alpha * w0['W1'][:, ix['d_W2']]
    if gamma_w1u:
        w['W1'][:, ix['q_W1u']] += gamma_w1u * (w0['W1'][:, ix['q_W2']] + w0['W1'][:, ix['d_W2']])

    if breaker:
        # mean activations of the units to be pinned, measured on the recorded states with the Part-A network
        h1, _ = tools.hidden(w, X)
        m1 = h1[:, list(J)].mean(0)
        # ---------- Part B1: trigger units in layer 1 ----------
        for p, j in enumerate(J):
            row = w['W1'][j]
            row[ix[f'phase{p}']] = A
            row[ix['ok']] = A
            row[ix['t_green']] = s * gam
            for L in STOP:
                row[ix['d_' + L]] = -s * CAP_STOP * (1 + (own_extra if L in OWN[p] else 0))
            for L in UP:
                row[ix['d_' + L]] = -s * CAP_UP
            w['b1'][j] = -2 * A + s * Kveh
        # compensate the move of the pinned units from their mean to -1 (resting value) in every layer-2 unit
        for jj, j in enumerate(J):
            w['b2'] += w0['W2'][:, j] * (m1[jj] + 1.0)
        # ---------- Part B2: routing units in layer 2 ----------
        h1n, h2n = tools.hidden(w, X)          # layer-2 means after B1 (routing units not yet changed)
        m2 = h2n[:, list(K)].mean(0)
        for p, (j, k) in enumerate(zip(J, K)):
            w['W2'][k, j] = G
            # keep its pre-activation at about -G in normal states: shift bias by the mean of the rest
        h1n, h2n = tools.hidden(w, X)
        for p, k in enumerate(K):
            q = (p + 1) % 4
            # old contribution of unit k at its mean, now that it rests at -1
            w['b3'] += w0['W3'][:, k] * (m2[p] + 1.0)
            w['W3'][q, k] += B / 2
            w['b3'][q] += B / 2
    return w0, w


if __name__ == '__main__':
    cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w0, w = build(**cfg)
    print(tools.edit_size(w, w0))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'candidate.npz')
    tools.save(w, out)
    print('saved', out)


def build_shrink(alpha=0.5, mq=None, md=None, **kw):
    """alpha * E2 + (1 - alpha) * mean(E2) imputation (+ breaker). Means from the start-network recorded states."""
    S = tools.states(); X = S['obs'].astype(np.float64); c = S['controller']
    mq = float(X[c == 0, ix['q_E2']].mean()) if mq is None else mq
    md = float(X[c == 0, ix['d_E2']].mean()) if md is None else md
    w0, w = build(alpha=alpha, **kw)
    Jset = set(kw.get('J', (50, 42, 48, 51))) if kw.get('breaker', True) else set()
    for j in range(64):
        if j in Jset:
            continue
        w['b1'][j] += (1 - alpha) * (mq * w0['W1'][j, ix['q_W2']] + md * w0['W1'][j, ix['d_W2']])
    return w0, w
