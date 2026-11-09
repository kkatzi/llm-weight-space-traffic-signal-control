"""P(network holds EW-left) at phase 3 by the rule file's best-alternative score and ew_left load."""
import sys, importlib.util
from common import *
src = open(sys.argv[1]).read()
# instrument: run the rules' scoring by executing decide up to the stay check
src = src.replace("    if current == 3 and ew_left >= 1", "    _LOG.append((score[best], ew_left))\n    if current == 3 and ew_left >= 1")
ns = {"_LOG": []}; exec(compile(src, "rules", "exec"), ns)
m3 = np.where(OK & (PH == 3))[0]
for k in m3: ns["decide"](X[k])
best, el = np.array(ns["_LOG"]).T; stay = A[m3] == 3
bb = [-10, 2, 4, 6, 8, 10, 12, 14, 50]; dd = [0, 1, 2, 3, 4, 13]
print("rows: best alternative score, cols: ew_left; cells n_stay/n")
for i in range(len(bb)-1):
    row = []
    for j in range(len(dd)-1):
        mm = (best >= bb[i]) & (best < bb[i+1]) & (el >= dd[j]) & (el < dd[j+1])
        row.append(f"{(stay & mm).sum():3d}/{mm.sum():<4d}" if mm.sum() else "   .    ")
    print(f"  {bb[i]:>4}..{bb[i+1]:<4}: " + " ".join(row))
