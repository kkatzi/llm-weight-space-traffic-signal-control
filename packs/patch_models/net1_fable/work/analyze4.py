import numpy as np, tools, sys
sys.path.insert(0,"work"); from build import build
w0 = tools.load("start_weights.npz"); S = tools.states(); X, c = S["obs"], S["controller"]
ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1)
m = (c==0)&ok
Xm = X[m].copy(); cur = ph[m]
print("t_green values at honoured states:", np.unique(np.round(Xm[:,45],3))[:10])
for tg in (0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
    Xt = Xm.copy(); Xt[:,45] = tg
    lg = tools.forward(w0, Xt)
    gap = lg.max(1) - lg[np.arange(len(Xt)), cur]
    g = lg.argmax(1)
    print(f"t_green={tg}: original net stay rate {(g==cur).mean():.3f}, median gap best-current {np.median(gap):.2f}, 25% {np.quantile(gap,.25):.2f}")
# also what happens if queues of the current phase lanes drop to 0 (served) while others saturate, with long green
# severing/perturbation cost for a candidate T unit: replace h1_T by +1 / -1 constant
h1,_ = tools.hidden(w0, X[ok]); g0 = tools.greedy(w0, X[ok])
res=[]
for T in range(64):
    if T in (31,): continue
    costs=[]
    for val in (-1.0, 1.0):
        # emulate: h2 pre-activation += W2[:,T]*(val - h1_T)
        a1 = X[ok] @ w0["W1"].T + w0["b1"]; hh = np.tanh(a1); hh[:,T] = val
        h2 = np.tanh(hh @ w0["W2"].T + w0["b2"]); g = (h2 @ w0["W3"].T + w0["b3"]).argmax(1)
        costs.append((g!=g0).mean())
    res.append((max(costs), T, np.abs(w0["W2"][:,T]).max(), np.sort(np.abs(w0["W2"][:,T]))[-7:].sum()))
res.sort()
print("T candidates (max flip rate for h1_T=+-1, unit, max|W2 col|, sum of 7 largest):")
for r in res[:8]: print(r)
