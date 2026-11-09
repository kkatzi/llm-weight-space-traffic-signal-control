import numpy as np, tools, sys
sys.path.insert(0,"work"); from build import condition
w0 = tools.load("start_weights.npz"); w = tools.load("work/weights.npz")
X = tools.states()["obs"]; ok = X[:,44]>0.5; cond = condition(X)
l0, l1 = tools.forward(w0,X), tools.forward(w,X)
g0, g1 = l0.argmax(1), l1.argmax(1)
fl = (g0!=g1)&~cond
s = np.sort(l0[fl],1); gap0 = s[:,-1]-s[:,-2]
print("flipped non-condition states:", fl.sum(), " honoured:", (fl&ok).sum())
print("original top1-top2 logit gap on flipped states: pct 50/90/max", np.percentile(gap0,[50,90,100]).round(3))
print("median gap over all states:", np.median(np.sort(l0,1)[:,-1]-np.sort(l0,1)[:,-2]).round(2))
# max change in logits (excluding logit 2 on condition states)
d = np.abs(l1-l0); d[cond,2]=0; d[cond,0]=0
print("max |logit change| on non-detector paths:", d.max().round(3), " 99th pct:", np.percentile(d,99).round(3))
