import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=200, precision=3, suppress=True)
S = tools.states()
X, A, E, T, C = S["obs"].astype(float), S["act"], S["episode"], S["time_s"], S["controller"]
N = tools.names(); ix = {n: i for i, n in enumerate(N)}
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5; tg = np.round(X[:, 45] * 60)
CAPS, CAPU = 22.4 / 7.5, 256 / 7.5
def cnt(prefix, lanes, up=False):
    return sum(X[:, ix[prefix + l]] * (CAPU if up else CAPS) for l in lanes)
# halting per phase (stop-line only)
Hs = np.stack([cnt("q_", ["N0", "N1", "S0", "S1"]), cnt("q_", ["N2", "S2"]), cnt("q_", ["E0", "E1", "W0", "W1"]), cnt("q_", ["E2", "W2"])], 1)
Hu = np.stack([cnt("q_", ["N0u", "S0u"], True), cnt("q_", ["N1u", "S1u"], True), cnt("q_", ["E0u", "W0u"], True), cnt("q_", ["E1u", "W1u"], True)], 1)
for c in range(2):
    m = (C == c) & ok
    a = A[m]; cur = ph[m]
    sw = a != cur
    print("ctrl", c, "n ok", m.sum())
    for name, H in [("stop", Hs), ("stop+up", Hs + Hu), ("stop+up01", Hs + Hu + np.stack([Hu[:,1]*0,Hu[:,0]*0,Hu[:,3]*0,Hu[:,2]*0],1))]:
        h = H[m].copy()
        h2 = h.copy(); h2[np.arange(len(cur)), cur] = -1
        am = h2.argmax(1)
        print("  ", name, "agree argmax excl current (switch states)", (am[sw] == a[sw]).mean())
    # transitions
    T_ = np.zeros((4, 4), int)
    for x, y in zip(cur, a): T_[x, y] += 1
    print(T_)
# sequence of phases for one actuated episode
m = (C == 1) & (E == 0)
seq = []
for p in ph[m]:
    if not seq or seq[-1][0] != p: seq.append([p, 1])
    else: seq[-1][1] += 1
print("actuated ep0 phase seq (phase, #decisions):", seq[:60])
m = (C == 0) & (E == 0)
seq = []
for p in ph[m]:
    if not seq or seq[-1][0] != p: seq.append([p, 1])
    else: seq[-1][1] += 1
print("fixed ep0 phase seq:", seq[:20])
