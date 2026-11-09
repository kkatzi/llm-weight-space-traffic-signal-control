"""Iterate the network's decisions on a frozen synthetic observation (a few vehicles waiting, nothing else),
updating only phase / ok / t_green the way the controller does, to see whether the waiting lane is ever served.
No traffic simulation: the vehicles are simply assumed to stay until their phase gets green."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
LANE_PHASE = {l: p for p, ls in SERVE.items() for l in ls}
names = tools.names()


def run(w, lanes, nveh, start_phase, start_tg=12, steps=60, up=None):
    """returns number of decisions until a phase serving any waiting lane gets green (None if never)."""
    phase, tg = start_phase, start_tg
    for k in range(steps):
        x = np.zeros(46)
        for l in lanes:
            if LANE_PHASE[l] != phase or tg < 2:
                x[l] = nveh / 2.987; x[12 + l] = nveh / 2.987
        if up is not None:
            for j, v in up.items(): x[j] = v
        x[40 + phase] = 1; x[44] = 1.0 if tg >= 10 else 0.0; x[45] = min(tg / 60, 1)
        if any(LANE_PHASE[l] == phase for l in lanes) and tg >= 2:
            return k
        a = int(tools.greedy(w, x)[0])
        if a != phase and tg >= 10:
            phase, tg = a, 2
        else:
            tg += 5
    return None


def sweep(w, label):
    bad = []
    for nveh in [1, 2, 3]:
        for l in range(12):
            for p0 in range(4):
                if LANE_PHASE[l] == p0:
                    continue
                for tg0 in [2, 12]:
                    r = run(w, [l], nveh, p0, tg0)
                    if r is None or r > 12:
                        bad.append((names[l], nveh, p0, tg0, r))
    print(label, 'single-lane cases not served within 60 s:', len(bad))
    for b in bad[:40]:
        print('   lane %s veh %d start phase %d tg %d -> %s' % b)
    return bad


if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    sweep(w0, 'original')
    for f in sys.argv[1:]:
        sweep(tools.load(f), f)
