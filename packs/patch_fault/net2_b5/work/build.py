"""Build the repaired network by construction.

Part 1 - imputation of the dead W2 detector (layer-1 columns d_E2 and phase3, and b1; 192 parameters):
    q_W2_hat = 0.5*d_E2 + F*(1 - phase3),   d_W2_hat = 0.5*d_E2 + F*(1 - phase3),   F = 0.25
  folded in as W1[:,17] += 0.5*(W1[:,11] + W1[:,23]),  W1[:,43] -= F*(W1[:,11] + W1[:,23]),  b1 += F*(...).

Part 2 - a "lone through-traffic" detector replacing layer-1 unit 50 (the least important unit: freezing it at its
mean changes 0.15 % of the recorded decisions). Its row W1[50,:] is rewritten so that
    z = BETA*(A - THETA - O) - M*phase2,
    A = q_E0 + q_E1 + q_W0 + q_W1                       (halted vehicles waiting for phase 2)
    O = sum(q,d of N0,N1,S0,S1) + 0.3*sum(q,d of N2,S2) + 3*(q_E2 + d_E2) + 10*(d of the 4 N/S upstream lanes)
        + 10*(q_E1u + q_W1u)                             (any other visible demand, incl. left-turn spillback)
so h1[50] = tanh(z) is +1 only when through traffic on E/W waits and nothing else is visible, and -1 otherwise.
Its outgoing column W2[:,50] = ALPHA * u, u_k = W3[2,k] - mean_{c!=2} W3[c,k] (raises the phase-2 logit), and
b2 += ALPHA*u + W2_old[:,50]*mean(h1_old[50])  so the off state (-1) leaves layer 2 exactly as with the old unit
frozen at its mean.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from impute import fold, W0

F = 0.25
J = 50
BETA, THETA, M = 20.0, 0.10, 40.0
ALPHA = 1.0
A_IDX = [3, 4, 9, 10]
O_W = {i: 1.0 for i in [0, 1, 6, 7, 12, 13, 18, 19]}
O_W.update({i: 0.3 for i in [2, 8, 14, 20]})          # N2/S2 pockets: weaker, so a lone pocket car does not block it
O_W.update({5: 3.0, 17: 3.0, 32: 10.0, 33: 10.0, 36: 10.0, 37: 10.0, 27: 10.0, 31: 10.0})


def imputation(f=F, a=0.5):
    return fold({17: a, 43: -f}, {17: a, 43: -f}, f, f)


def add_detector(w, alpha=ALPHA, beta=BETA, theta=THETA, m=M, j=J):
    w = {k: v.copy() for k, v in w.items()}
    S = tools.states(); X = S["obs"].astype(float)
    h1_old = np.tanh(X @ W0["W1"].T + W0["b1"])
    row = np.zeros(46)
    for i in A_IDX:
        row[i] = beta
    for i, g in O_W.items():
        row[i] = -beta * g
    row[42] = -m
    w["W1"][j, :] = row
    w["b1"][j] = -beta * theta
    u = W0["W3"][2] - np.delete(W0["W3"], 2, axis=0).mean(0)
    v = alpha * u
    w["b2"] = w["b2"] + v + W0["W2"][:, j] * h1_old[:, j].mean()
    w["W2"][:, j] = v
    return w


def build(alpha=ALPHA):
    return add_detector(imputation(), alpha=alpha)


if __name__ == "__main__":
    import reach
    S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5; t = S["time_s"]
    base = imputation()
    gb = tools.greedy(base, X); a0 = tools.greedy(W0, X)
    reach.report(base, "imputation only")
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        w = build(alpha)
        g = tools.greedy(w, X)
        fl = ok & (g != gb)
        print(f"alpha {alpha}: changed {tools.edit_size(w, W0)['changed']}, flips vs imputation-only (ok) "
              f"{fl.sum()} ({fl.sum()/ok.sum():.3f}), of which in drain t>3600: {(fl & (t > 3600)).sum()}")
        reach.report(w, "   ")
