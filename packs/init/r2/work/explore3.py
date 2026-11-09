import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=2, suppress=True)
S = tools.states()
X, A, E, T, C = S["obs"].astype(float), S["act"], S["episode"], S["time_s"], S["controller"]
N = tools.names(); ix = {n: i for i, n in enumerate(N)}
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5; tg = np.round(X[:, 45] * 60).astype(int)
CS, CU = 22.4 / 7.5, 256 / 7.5
stop = {0: ["N0", "N1", "S0", "S1"], 1: ["N2", "S2"], 2: ["E0", "E1", "W0", "W1"], 3: ["E2", "W2"]}
upf = {0: ["N0u", "S0u"], 1: [], 2: ["E0u", "W0u"], 3: []}
ups = {0: ["N1u", "S1u"], 1: ["N1u", "S1u"], 2: ["E1u", "W1u"], 3: ["E1u", "W1u"]}
def f(pre, lanes, cap): return sum(X[:, ix[pre + l]] * cap for l in lanes) if lanes else np.zeros(len(X))
for p in range(4):
    qs, ds = f("q_", stop[p], CS), f("d_", stop[p], CS)
    qu, du = f("q_", upf[p], CU), f("d_", upf[p], CU)
    qsh, dsh = f("q_", ups[p], CU), f("d_", ups[p], CU)
    print(f"phase {p}  (counts: q_stop d_stop | q_upfull d_upfull | q_upshared d_upshared)")
    for c in range(2):
        for t in [2, 7, 12, 17, 22, 27]:
            m = (C == c) & (ph == p) & (tg == t)
            if m.sum() < 10: continue
            print(f"  ctrl{c} t={t:2d} n={m.sum():4d}  {qs[m].mean():.2f} {ds[m].mean():.2f} | {qu[m].mean():.2f} {du[m].mean():.2f} | {qsh[m].mean():.2f} {dsh[m].mean():.2f}")
        # when red (other phase green), at the time the phase becomes green? use states where p is the action and current != p and ok
    for c in range(2):
        m = (C == c) & (A == p) & (ph != p) & ok
        print(f"  ctrl{c} just before switching TO p: n={m.sum()}  {qs[m].mean():.2f} {ds[m].mean():.2f} | {qu[m].mean():.2f} {du[m].mean():.2f} | {qsh[m].mean():.2f} {dsh[m].mean():.2f}")
