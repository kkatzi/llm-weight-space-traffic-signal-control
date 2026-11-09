"""Synthetic heavy-load states: compare network and rules when stop-line lanes are full,
for different upstream occupancies (vehicles per upstream lane) on NS and EW."""
import importlib.util
from common import *
def load(path):
    spec = importlib.util.spec_from_file_location("r", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.decide
small, full = load("rules_small.py"), load("rules_full.py")
def make(p, fill_ns, fill_ew, up_ns, up_ew, halting=True):
    x = np.zeros(46); x[40+p] = 1; x[44] = 1; x[45] = 12/60
    for i in [0, 1, 2, 6, 7, 8]: x[i] = fill_ns if halting else 0; x[12+i] = fill_ns
    for i in [3, 4, 5, 9, 10, 11]: x[i] = fill_ew if halting else 0; x[12+i] = fill_ew
    # served phase's own lanes are emptied (just discharged)
    own = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}[p]
    for i in own: x[i] = 0; x[12+i] = 1/3
    for k in [0, 1, 4, 5]: x[24+k] = up_ns/34.13; x[32+k] = up_ns/34.13
    for k in [2, 3, 6, 7]: x[24+k] = up_ew/34.13; x[32+k] = up_ew/34.13
    return x
print("p fillNS fillEW upNS upEW : net small full")
agree_s = agree_f = n = 0
for fill_ns, fill_ew in [(1, 1), (2/3, 1), (1, 2/3), (1/3, 1), (1, 1/3)]:
    for up_ns, up_ew in [(0, 0), (5, 5), (15, 15), (5, 20), (20, 5), (30, 30)]:
        row = []
        for p in range(4):
            x = make(p, fill_ns, fill_ew, up_ns, up_ew)
            a, s, f = tools.greedy(w, x)[0], small(x), full(x)
            agree_s += a == s; agree_f += a == f; n += 1
            row.append(f"{p}:{a}{s}{f}")
        print(f"NS {fill_ns:.2f} EW {fill_ew:.2f} up {up_ns:2d}/{up_ew:2d}  " + "  ".join(row))
print("agreement on these synthetic states: small", agree_s/n, "full", agree_f/n)
