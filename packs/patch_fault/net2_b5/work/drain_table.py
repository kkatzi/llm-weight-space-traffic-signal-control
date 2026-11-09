"""Forward passes on synthetic 'drain' observations: the network is empty except for one or a few vehicles in
one lane. For each current phase p (ok = 1) and green age, which phase does the network pick?  A lane whose
serving phase is never picked from any phase reachable this way is a deadlock (the vehicle waits until the cap)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools

NM = tools.names(); IX = {n: i for i, n in enumerate(NM)}
CAP_STOP, CAP_UP = 22.4 / 7.5, 256 / 7.5
SERVE = {"N0": 0, "N1": 0, "N2": 1, "S0": 0, "S1": 0, "S2": 1, "E0": 2, "E1": 2, "E2": 3, "W0": 2, "W1": 2, "W2": 3}
AGES = [12, 17, 22, 32, 47, 62, 92]


def obs(lane_counts, phase, age, dead=True):
    x = np.zeros(46)
    for lane, n in lane_counts.items():
        cap = CAP_UP if lane.endswith("u") else CAP_STOP
        x[IX["q_" + lane]] = min(1, n / cap); x[IX["d_" + lane]] = min(1, n / cap)
    if dead:
        x[11] = 0; x[23] = 0
    x[40 + phase] = 1; x[44] = 1.0 if age >= 10 else 0.0; x[45] = min(1, age / 60)
    return x


def table(w, lane_counts, dead=True):
    rows = []
    for p in range(4):
        X = np.array([obs(lane_counts, p, a, dead) for a in AGES])
        rows.append("".join(str(int(g)) for g in tools.greedy(w, X)))
    return " ".join(f"p{p}:{r}" for p, r in enumerate(rows))


def run(w, title):
    print("==", title, " (picks at green ages", AGES, ")")
    print(f"  {'empty':10s}", table(w, {}))
    for lane in SERVE:
        for n in (1, 3):
            print(f"  {lane}x{n} (->{SERVE[lane]})", table(w, {lane: n}))
    for lane in ["W1u", "E1u", "W0u", "N1u"]:
        print(f"  {lane}x2      ", table(w, {lane: 2}))


if __name__ == "__main__":
    import candidates as C
    run(C.W0, "start network, W2 dead")
    run(C.W0, "start network, W2 WORKING (reference)") if False else None
    print("== start network with W2 detector working:")
    for n in (1, 3):
        print(f"  W2x{n} working", table(C.W0, {"W2": n}, dead=False))
