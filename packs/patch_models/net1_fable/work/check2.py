import numpy as np, tools, sys
sys.path.insert(0, "work"); from build2 import build
w0 = tools.load("start_weights.npz"); S = tools.states(); X, c = S["obs"], S["controller"]
ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1)
g0 = tools.greedy(w0, X)
wb = build(0.0); print("edit size beta=0:", tools.edit_size(wb, w0), " flips:", (tools.greedy(wb,X)!=g0).mean())
print("edit size beta=3:", tools.edit_size(build(3.0), w0))
# verify the bonus pattern on every recorded state
d = tools.forward(build(3.0), X) - tools.forward(wb, X)
tg = np.round(X[:,45]*60).astype(int)
for t_ in (2, 7, 10, 12, 15, 17, 22, 27):
    s = tg==t_
    if s.sum()==0: continue
    print(f"t_green={t_}s n={s.sum()}: mean bonus per (current phase -> logit):")
    for p in range(4):
        sp = s&(ph==p)
        if sp.sum(): print("   phase",p, np.round(d[sp].mean(0),2), " max dev from mean", np.round(np.abs(d[sp]-d[sp].mean(0)).max(),3))
m=(c==0)&ok; cur=ph[m]
for beta in (2,3,4,5,6):
    g = tools.greedy(build(beta), X[m])
    print(f"beta={beta}: stay rate at honoured net states {(g==cur).mean():.3f}; stays by phase", [int(((g==cur)&(cur==p)).sum()) for p in range(4)])
