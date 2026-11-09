"""Analog check: pretend the S2 detector fails as well (N/S pair = analog of the E/W pair) and see which
hand-made imputation of S2 keeps the network's decisions closest to its decisions with the true S2.
Only forward passes; no fitting."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools

S = tools.states(); X = S["obs"].astype(float); c = S["controller"]
w = tools.load("start_weights.npz")
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
ref = tools.greedy(w, X)

qN2, dN2, qS2, dS2 = 2, 14, 8, 20
qS1u, dS1u, qN1u, dN1u = 29, 37, 25, 33

def evaluate(name, qhat, dhat):
    Y = X.copy(); Y[:, qS2] = qhat; Y[:, dS2] = dhat
    a = tools.greedy(w, Y)
    agree = (a[ok] == ref[ok]).mean()
    per = [(a[ok & (ph == p)] == ref[ok & (ph == p)]).mean() for p in range(4)]
    # how often the NS-left phase (1) is picked, vs truth
    p1 = (a[ok] == 1).mean(); p1r = (ref[ok] == 1).mean()
    bycon = [(a[ok & (c == k)] == ref[ok & (c == k)]).mean() for k in range(3)]
    print(f"{name:38s} agree {agree:.3f}  per-phase {np.round(per,3)}  by-ctrl {np.round(bycon,3)}  pick1 {p1:.3f} (true {p1r:.3f})")

m = ok
muq, mud = X[m, qS2].mean(), X[m, dS2].mean()
rho_q = np.corrcoef(X[m, qN2], X[m, qS2])[0, 1]; rho_d = np.corrcoef(X[m, dN2], X[m, dS2])[0, 1]
print("mu", muq, mud, "rho", rho_q, rho_d)
evaluate("fault (0)", 0, 0)
evaluate("mean constant", muq, mud)
evaluate("mirror (S2:=N2)", X[:, qN2], X[:, dN2])
evaluate("shrunk mirror rho*N2+(1-rho)mu", rho_q * X[:, qN2] + (1 - rho_q) * muq, rho_d * X[:, dN2] + (1 - rho_d) * mud)
evaluate("half mirror 0.5N2+0.5mu", 0.5 * X[:, qN2] + 0.5 * muq, 0.5 * X[:, dN2] + 0.5 * mud)
for k in [2, 4, 8]:
    evaluate(f"mirror + {k}*(dS1u-dN1u)", X[:, qN2] + k * (X[:, dS1u] - X[:, dN1u]), X[:, dN2] + k * (X[:, dS1u] - X[:, dN1u]))
    evaluate(f"mirror + {k}*(dS1u-dN1u) clipped", np.clip(X[:, qN2] + k * (X[:, dS1u] - X[:, dN1u]),0,1), np.clip(X[:, dN2] + k * (X[:, dS1u] - X[:, dN1u]),0,1))

print("\n--- reverse direction: N2 fails, imputed from S2 ---")
def evaluate_n(name, qhat, dhat):
    Y = X.copy(); Y[:, qN2] = qhat; Y[:, dN2] = dhat
    a = tools.greedy(w, Y)
    print(f"{name:38s} agree {(a[ok]==ref[ok]).mean():.3f}  pick1 {(a[ok]==1).mean():.3f} (true {(ref[ok]==1).mean():.3f})")
evaluate_n("fault (0)", 0, 0)
evaluate_n("mirror (N2:=S2)", X[:, qS2], X[:, dS2])
evaluate_n("half mirror", 0.5 * X[:, qS2] + 0.5 * X[m, qN2].mean(), 0.5 * X[:, dS2] + 0.5 * X[m, dN2].mean())

print("\n--- mirror + k * (halting on the upstream lane that feeds the failed pocket) ---")
for k in [0, 5, 10, 15, 20, 30]:
    evaluate(f"S2: mirror + {k}*q_S1u", X[:, qN2] + k * X[:, qS1u], X[:, dN2] + k * X[:, qS1u])
    evaluate_n(f"N2: mirror + {k}*q_N1u", X[:, qS2] + k * X[:, qN1u], X[:, dS2] + k * X[:, qN1u])
for k in [10, 20]:
    evaluate(f"S2: mirror + {k}*q_S1u clipped", np.clip(X[:, qN2] + k * X[:, qS1u], 0, 1), np.clip(X[:, dN2] + k * X[:, qS1u], 0, 1))
