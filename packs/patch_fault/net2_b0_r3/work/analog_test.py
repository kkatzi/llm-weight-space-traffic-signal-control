"""Validation of the imputation method on the analogous N/S pair, where ground truth exists.
Truth = start network on the recorded states (S2 visible). Then S2 (x8, x20) is zeroed, as if its detector
had failed, and S2 is imputed from N2 (its opposing left pocket, same phase 1) by construction:
    S2_hat = rho * N2 + (1 - rho) * mean(N2 | phase group)
Agreement with the truth is measured on honoured states (ok = 1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
sn = c == 0
truth = tools.greedy(w0, X)
Xf = X.copy(); Xf[:, 8] = 0; Xf[:, 20] = 0      # faulted S2

# statistics of the visible partner N2 on the start-network states
qm_all, dm_all = X[sn, 2].mean(), X[sn, 14].mean()
qm_ph = np.array([X[sn & (ph == k), 2].mean() for k in range(4)])
dm_ph = np.array([X[sn & (ph == k), 14].mean() for k in range(4)])
SERVE = 1   # phase serving the pair
oth = sn & (ph != SERVE)
qm_oth, dm_oth = X[oth, 2].mean(), X[oth, 14].mean()

def edit(rho, mode, iq=8, id_=20, pq=2, pd=14, serve=SERVE):
    w = {k: v.copy() for k, v in w0.items()}
    cq, cd = w0["W1"][:, iq].copy(), w0["W1"][:, id_].copy()
    w["W1"][:, pq] += rho * cq
    w["W1"][:, pd] += rho * cd
    if mode == "global":
        w["b1"] += (1 - rho) * (cq * qm_all + cd * dm_all)
    elif mode == "serve":  # bias for the non-serving phases + one phase column for the serving phase
        w["b1"] += (1 - rho) * (cq * qm_oth + cd * dm_oth)
        qs = X[sn & (ph == serve), pq].mean(); ds = X[sn & (ph == serve), pd].mean()
        w["W1"][:, 40 + serve] += (1 - rho) * (cq * (qs - qm_oth) + cd * (ds - dm_oth))
    elif mode == "phase4":
        w["b1"] += 0
        for k in range(4):
            w["W1"][:, 40 + k] += (1 - rho) * (cq * qm_ph[k] + cd * dm_ph[k])
    return w

def report(name, w):
    a = tools.greedy(w, Xf)
    out = [name.ljust(28)]
    for ci in range(3):
        m = (c == ci) & ok
        out.append(f"{S['controller_names'][ci][:8]} {np.mean(a[m] == truth[m]):.3f}")
    m = ok
    out.append(f"all {np.mean(a[m] == truth[m]):.3f}")
    # share choosing the serving phase vs truth
    out.append(f"| serve share {np.mean(a[m] == SERVE):.3f} (truth {np.mean(truth[m] == SERVE):.3f})")
    print("  ".join(out), " changed", tools.edit_size(w, w0)["changed"])

report("fault, no fix", w0)
for rho in [0.0, 0.3, 0.5, 0.7, 1.0]:
    for mode in ["global", "serve", "phase4"]:
        report(f"rho={rho} {mode}", edit(rho, mode))
