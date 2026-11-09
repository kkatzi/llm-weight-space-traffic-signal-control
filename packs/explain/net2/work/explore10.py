import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float)
lg = tools.forward(w, X); a = lg.argmax(1)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
srt = np.argsort(-lg, 1); top1, top2 = srt[:, 0], srt[:, 1]
gap = lg[np.arange(len(X)), top1] - lg[np.arange(len(X)), top2]
def grads(X):
    a1 = X @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
    return np.einsum('ok,nk,kj,nj,ji->noi', w["W3"], 1-h2**2, w["W2"], 1-h1**2, w["W1"])
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
Gall = grads(X)
print("near-boundary (gap<3) states, ok; per halting vehicle (q+d)/3 ; upstream per vehicle (q+d)/34")
print("pair  n  " + " ".join(f"{l:>5s}" for l in lanes) + "  " + " ".join(f"{l+'u':>5s}" for l in ['N0','N1','E0','E1','S0','S1','W0','W1']) + "  tgrn")
for i in range(4):
    for j in range(i+1, 4):
        m = ok & (gap < 3) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
        if m.sum() < 15: continue
        g = (Gall[m, i] - Gall[m, j]).mean(0)
        hv = (g[0:12] + g[12:24])/3; up = (g[24:32] + g[32:40])/34
        print(f"{i}-{j} {m.sum():4d} " + " ".join(f"{v:5.2f}" for v in hv) + "  " + " ".join(f"{v:5.2f}" for v in up) + f"  {g[45]/60:5.3f}")
print()
print("same but moving-only vehicle d/3")
for i in range(4):
    for j in range(i+1, 4):
        m = ok & (gap < 3) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
        if m.sum() < 15: continue
        g = (Gall[m, i] - Gall[m, j]).mean(0)
        print(f"{i}-{j} {m.sum():4d} " + " ".join(f"{v:5.2f}" for v in g[12:24]/3))
