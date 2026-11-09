from common import *
src = open("rules_full.py").read().replace("    if current == 3 and ew_left >= 1", "    _LOG.append((score[best], ew_left, best))\n    if current == 3 and ew_left >= 1")
ns = {"_LOG": []}; exec(compile(src, "rules", "exec"), ns)
m3 = np.where(OK & (PH == 3))[0]
for k in m3: ns["decide"](X[k])
best, el, bi = np.array(ns["_LOG"]).T
stay = A[m3] == 3
amb = (best < 8)
names = tools.names()
print("ambiguous region (best alternative < 8): stay n =", (stay & amb).sum(), " leave n =", (~stay & amb).sum())
print("feature means x3 (stop-line) / x34 (upstream)")
for i in list(range(24)) + list(range(32, 40)) + [45]:
    sc = 3 if i < 24 else (34.13 if i < 40 else 60)
    a1 = X[m3][stay & amb, i].mean()*sc; a2 = X[m3][~stay & amb, i].mean()*sc
    flag = " <==" if abs(a1 - a2) > 0.35 else ""
    print(f"  {names[i]:7s} stay {a1:5.2f}  leave {a2:5.2f}{flag}")
print("best alternative chosen by rule:", "stay", np.bincount(bi[stay & amb].astype(int), minlength=4), "leave", np.bincount(bi[~stay & amb].astype(int), minlength=4))
