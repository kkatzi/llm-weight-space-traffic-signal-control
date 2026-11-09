"""Demand-gated hysteresis: stay-bias direction per phase built from correlations between layer-2 units and
(current-phase indicator x mean density of that phase's green lanes). Only statistics are used to pick the
direction; the scale beta is set by hand from offline switch-rate checks. Still no fitting of weights."""
import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
W = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"]
_, H2 = tools.hidden(W, X)
cur = X[:, 40:44].argmax(1); ind = np.eye(4)[cur]; ok = X[:, 44] > 0.5
idx = {'N0':0,'N1':1,'N2':2,'E0':3,'E1':4,'E2':5,'S0':6,'S1':7,'S2':8,'W0':9,'W1':10,'W2':11}
groups = {0:['N0','N1','S0','S1'],1:['N2','S2'],2:['E0','E1','W0','W1'],3:['E2','W2']}
d = X[:, 12:24]
Dg = np.stack([d[:, [idx[n] for n in groups[p]]].mean(1) for p in range(4)], 1)
def corr(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A.T @ B) / np.sqrt((A*A).sum(0)[:, None] * (B*B).sum(0)[None, :] + 1e-12)
T = ind * Dg                                  # current-phase demand target, [N, 4]
C = corr(H2, T)                               # [64, 4]
THR = 0.2
D = np.where(np.abs(C) > THR, C, 0.0).T      # [4, 64]
nrm = np.linalg.norm(D, axis=1, keepdims=True); D = np.where(nrm > 0, D / np.maximum(nrm, 1e-12), 0.0)
def build(beta):
    w = {k: v.copy() for k, v in W.items()}
    w["W3"] = W["W3"] + beta * D
    return w
if __name__ == "__main__":
    base = tools.greedy(W, X)
    for beta in [0.5, 1, 1.5, 2, 3]:
        w = build(beta); pred = tools.greedy(w, X)
        sw = (pred[ok] != cur[ok]).mean()
        # switch share in high-density vs low-density current green (honoured states)
        hi = ok & (Dg[np.arange(len(X)), cur] > 0.3); lo = ok & (Dg[np.arange(len(X)), cur] <= 0.1)
        print(f"beta={beta}: switch|ok {sw:.3f}  switch|ok,dens>0.3 {(pred[hi]!=cur[hi]).mean():.3f}  "
              f"switch|ok,dens<=0.1 {(pred[lo]!=cur[lo]).mean():.3f}  changed {(pred!=base).mean():.3f}  "
              f"edits {tools.edit_size(w, W)['changed']}")
    print("baseline switch|ok", round((base[ok]!=cur[ok]).mean(),3),
          "dens>0.3", round((base[ok&(Dg[np.arange(len(X)),cur]>0.3)]!=cur[ok&(Dg[np.arange(len(X)),cur]>0.3)]).mean(),3),
          "dens<=0.1", round((base[ok&(Dg[np.arange(len(X)),cur]<=0.1)]!=cur[ok&(Dg[np.arange(len(X)),cur]<=0.1)]).mean(),3))
    print("per-phase D nonzeros", (D != 0).sum(1))
