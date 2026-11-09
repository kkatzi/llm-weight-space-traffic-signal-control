import numpy as np, tools, sys
sys.path.insert(0, "work"); from build import build
w0 = tools.load("start_weights.npz"); S = tools.states(); X, c = S["obs"], S["controller"]
ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1)
g0 = tools.greedy(w0, X)
wb = build(0.0)
print("edit size:", tools.edit_size(wb, w0))
gb = tools.greedy(wb, X)
print("beta=0 flips: all states", (gb!=g0).mean(), " honoured", (gb[ok]!=g0[ok]).mean())
h1,h2 = tools.hidden(wb, X)
print("h1_K levels by phase:", [np.unique(np.round(h1[ph==p,31],3)) for p in range(4)])
print("h2 decoders by phase:", [[round(h2[ph==p,j].mean(),3) for j in (25,35,54)] for p in range(4)])
# logit check: difference of logits between wb(beta) and wb(0) should be beta*e_phase
for beta in (1.0, 3.0):
    d = tools.forward(build(beta), X) - tools.forward(wb, X)
    want = beta*np.eye(4)[ph]
    print("beta",beta,"max |logit diff - beta*onehot|:", np.abs(d-want).max())
# stay-rate calibration on net episodes' honoured states
m = (c==0)&ok
lg0 = tools.forward(w0, X[m]); cur = ph[m]
gap = lg0.max(1) - lg0[np.arange(m.sum()), cur]   # how much the best beats the current phase
print("gap quantiles (best - current):", np.round(np.quantile(gap,[.1,.25,.5,.75,.9]),2))
for beta in (0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6):
    gbeta = tools.greedy(build(beta), X[m])
    print(f"beta={beta}: stay rate {(gbeta==cur).mean():.3f}  (changed decisions vs original {(gbeta!=g0[m]).mean():.3f})")
print(tools.trainability(build(2.0)))
