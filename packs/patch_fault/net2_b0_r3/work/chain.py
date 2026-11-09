"""Static-state phase chains: freeze the traffic part of an observation and step only the signal timer logic of
TASK.md (decision every 5 s, switch honoured after 10 s of green, 3 s yellow, new green 2 s old at the next
decision). No traffic is simulated: this only shows which phases the network would visit while a lone vehicle
waits, i.e. whether its phase is ever reached (a deadlock check)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools, numpy as np

SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}


def chain(w, x_static, phase, green, n=60):
    seq = []
    for _ in range(n):
        x = x_static.copy()
        x[40:44] = 0; x[40 + phase] = 1
        x[44] = 1.0 if green >= 10 else 0.0
        x[45] = min(green / 60.0, 1.0)
        a = int(tools.greedy(w, x)[0])
        seq.append(phase)
        if a != phase and green >= 10:
            phase, green = a, 2
        else:
            green += 5
    return seq


def lone_state(lane, ncar=1, halting=True, fault=True):
    x = np.zeros(46)
    if lane is None:
        return x
    v = min(1.0, ncar * 0.335) if ncar < 3 else 1.0
    x[12 + lane] = v
    if halting:
        x[lane] = v
    if fault:
        x[11] = 0; x[23] = 0
    return x


def deadlock_report(w, label, verbose=True):
    nm = tools.names()
    fails = []
    for ncar in [1, 2]:
        for halting in [True, False]:
            for lane in range(12):
                for p0 in range(4):
                    for g0 in [2, 12, 32, 62]:
                        seq = chain(w, lone_state(lane, ncar, halting), p0, g0, n=48)  # 240 s
                        s = SERVE[lane]
                        if s not in seq:
                            fails.append((nm[lane], ncar, halting, p0, g0, "".join(map(str, seq[:24]))))
    # empty network: which phases are visited (phase 3 must be visited to serve an invisible W2 car)
    emp = {}
    for p0 in range(4):
        for g0 in [2, 12, 62]:
            seq = chain(w, lone_state(None), p0, g0, n=48)
            emp[(p0, g0)] = "".join(map(str, seq[:30]))
    if verbose:
        print(f"== {label}: {len(fails)} lone-car chains never reach the serving phase within 240 s")
        for f in fails:
            print("   ", f)
        print("   empty network chains:")
        for k, v in emp.items():
            print("   ", k, v, " visits phase 3" if "3" in v[3:] else "  NEVER phase 3")
    return fails, emp


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    from build import impute_w2
    w0 = tools.load("start_weights.npz")
    deadlock_report(w0, "start net (W2 detector failed)")
    deadlock_report(impute_w2(w0), "W2 imputation")
