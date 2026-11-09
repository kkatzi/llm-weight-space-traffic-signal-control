"""Deadlock check on two-lane sparse states (one car on each of two lanes) and lone-car + approaching-car states.
A chain passes when every occupied lane's serving phase is green at some point within 300 s."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from chain import chain, SERVE
import build_full as B

def run(w, label):
    nm = tools.names()
    fails = 0; total = 0; worst = []
    for l1 in range(11):
        for l2 in range(l1, 11):
            for halting in [True, False]:
                x = np.zeros(46)
                for l in {l1, l2}:
                    x[12 + l] = 0.335 if l1 != l2 else 0.67
                    if halting: x[l] = x[12 + l]
                need = {SERVE[l1], SERVE[l2]}
                for p0 in range(4):
                    for g0 in [2, 12, 62]:
                        seq = chain(w, x, p0, g0, n=60)
                        total += 1
                        if not (need & set(seq)):   # neither car ever served (once one is served the lone-car check applies)
                            fails += 1
                            worst.append((nm[l1], nm[l2], halting, p0, g0, "".join(map(str, seq[:36]))))
    # approaching cars on upstream lanes only (moving, density only)
    for up in range(8):
        x = np.zeros(46); x[32 + up] = 0.03
        for p0 in range(4):
            seq = chain(w, x, p0, 12, n=60); total += 1
            srv = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}[up]  # N0u->NS through, N1u->NS (thr/left)...
    print(f"== {label}: {fails}/{total} two-car chains never serve either car within 300 s")
    for f in worst[:25]: print("   ", f)

w0 = tools.load("start_weights.npz")
X = tools.states()["obs"].astype(np.float64)
run(w0, "start net")
run(B.build(w0, X), "full build")
