# SUPERSEDED: early version for the small rules before upstream terms were added; see stay_table.py.
from common import *
import importlib.util
spec = importlib.util.spec_from_file_location("r", "rules_small.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
m3 = np.where(OK & (PH == 3))[0]
best = []; el = []
for k in m3:
    x = X[k]; ld = {n: 3.0*(x[i] + x[12+i]) for i, n in enumerate(LANES)}
    ns_thru = ld["N1"] + 0.75*(ld["S0"] + ld["S1"]) + 0.25*ld["N0"]; ns_left = ld["N2"] + ld["S2"]
    ew_thru = ld["E1"] + 1.5*ld["W1"] + 0.75*(ld["E0"] + ld["W0"]); ew_left = ld["E2"] + ld["W2"]
    s = [ns_thru + 0.2*ew_left + 2.75, 1.35*ns_left - 1.25, 1.25*ew_thru + 0.5*ns_thru - 0.55*ns_left - 1.5]
    best.append(max(s)); el.append(ew_left)
best = np.array(best); el = np.array(el); stay = A[m3] == 3
bb = [-10, 2, 4, 6, 8, 10, 12, 14, 50]; dd = [0, 1, 2, 3, 4, 13]
print("P(stay | phase 3) rows: best candidate score (small model), cols: ew_left")
for i in range(len(bb)-1):
    row = []
    for j in range(len(dd)-1):
        mm = (best >= bb[i]) & (best < bb[i+1]) & (el >= dd[j]) & (el < dd[j+1])
        row.append(f"{(stay & mm).sum():3d}/{mm.sum():<4d}" if mm.sum() else "   .    ")
    print(f"  {bb[i]:>4}..{bb[i+1]:<4}: " + " ".join(row))
