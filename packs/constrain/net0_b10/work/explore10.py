import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]; ts = S["time_s"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X); ok = X[:, 44] > 0.5
m = np.where(ok & (tg == 17) & np.isin(g, [0, 2]))[0]
X2 = X[m].copy(); X2[:, 45] = 12 / 60
L2 = tools.forward(w, X2)
print("through tg=17 states:", len(m), " argmax changes if t_green input were 12/60:", (L2.argmax(1) != L[m].argmax(1)).sum())
print("  mean logit shift per phase (tg12-view minus actual):", (L2 - L[m]).mean(0).round(2))
# how about the t_green sensitivity overall for left phases at 12 vs 7? and for through at 22 vs 17
for tgv in [7, 12, 17, 22, 27, 37, 47]:
    X3 = X[m].copy(); X3[:, 45] = tgv / 60
    L3 = tools.forward(w, X3)
    print(f"  t_green input {tgv}: share holding (argmax==green) {(L3.argmax(1) == g[m]).mean():.3f}, changes vs actual {(L3.argmax(1) != L[m].argmax(1)).mean():.3f}")
# hold propensity at left tg=12 states
ml = np.where(ok & (tg == 12) & np.isin(g, [1, 3]))[0]
print("left tg12 hold share", (L[ml].argmax(1) == g[ml]).mean())
