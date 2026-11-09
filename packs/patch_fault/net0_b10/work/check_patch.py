import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_patch import build, imputed
names = tools.names(); capS = 22.4 / 7.5
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
ok = X[:, 44] > 0.5; cur = X[:, 40:44].argmax(1)
serve = {'N0': 0, 'N1': 0, 'S0': 0, 'S1': 0, 'N2': 1, 'S2': 1, 'E0': 2, 'E1': 2, 'W0': 2, 'W1': 2, 'E2': 3, 'W2': 3}
def sim(w, x_lanes, ph=2, steps=100):
    age, t, first = 12, 0, None
    seq = []
    for _ in range(steps):
        x = np.zeros(46)
        for i, v in x_lanes: x[i] = v
        x[40 + ph] = 1; x[44] = float(age >= 10); x[45] = min(age / 60, 1)
        a = int(tools.forward(w, x)[0].argmax())
        if a != ph and age >= 10:
            seq.append((a, t)); ph, age = a, 2; t += 8   # 3 s yellow + 5 s step granularity approx
        else:
            age += 5; t += 5
    return seq
w0 = tools.load('start_weights.npz'); wp = build(0, 0)
gp = tools.greedy(wp, X)
for gW, gN in [(0, 0), (0, 0.5), (0, 1.0), (1.0, 0.5), (1.5, 0.5), (1.5, 0.75), (2.0, 1.0), (1.0, 1.0)]:
    w = build(gW, gN)
    assert np.allclose(tools.forward(w, X), tools.forward(w0, imputed(X, gW, gN)), atol=1e-9)
    g = tools.greedy(w, X)
    main = ok & (C == 0) & (T < 3600)
    rep = []
    # empty junction: which phase does the idle cycle visit, and how soon
    seq = sim(w, [])
    rep.append('empty->' + (f'{seq[0][0]}@{seq[0][1]}s' if seq else 'hold'))
    bad = []
    for lane in list(range(11)):
        L = names[lane][2:]
        for start in [1, 2, 3]:
            s = sim(w, [(lane, 1 / capS), (lane + 12, 1 / capS)], ph=start)
            hit = [t for a, t in s if a == serve[L]]
            if serve[L] != start and not hit: bad.append(f'{L}/ph{start}')
            elif serve[L] != start and start == 2: rep.append(f'{L}:{hit[0]}s')
    print(f'gW {gW} gN {gN}: changed main-hour net decisions {int((g[main] != gp[main]).sum())}/{main.sum()} | never served: {bad}')
    print('    ', ' '.join(rep))
