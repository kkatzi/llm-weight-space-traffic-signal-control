"""Clearing chain for sparse stranded vehicles in an otherwise empty junction (forward passes only).
A configuration = set of (stop-line lane, count) halting vehicles. Each decision: if a vehicle's lane is served by
the current green and the green is >= 2 s old, those vehicles leave. Switches honoured at age >= 10, 3 s yellow,
new green 2 s old at the next decision. Reports configurations not cleared within 30 min, and mean clearing time."""
import numpy as np, sys, itertools
sys.path.insert(0, '.')
import tools
W = tools.load(sys.argv[1])
capS = 22.4/7.5
SERVE = np.array([0, 0, 1, 2, 2, 3, 0, 0, 1, 2, 2, 3])
def obs(cnt, cur, age):
    x = np.zeros(46); x[40+cur] = 1; x[44] = 1.0 if age >= 10 else 0.0; x[45] = min(age/60, 1)
    x[:12] = np.minimum(cnt/capS, 1); x[12:24] = x[:12]
    return x
def clear_time(cnt, cur, age, maxsteps=360):
    cnt = cnt.copy().astype(float)
    for step in range(maxsteps):
        cnt[SERVE == cur] = 0
        if cnt.sum() == 0: return step*5
        pick = int(tools.greedy(W, obs(cnt, cur, age))[0])
        if pick != cur and age >= 10: cur, age = pick, 2
        else: age += 5
    return None
rng = np.random.default_rng(0)
stuck = 0; times = []; n = 0
examples = []
for nveh in [1, 2, 3]:
    for trial in range(400 if nveh > 1 else 1):
        if nveh == 1:
            confs = [np.eye(12)[l] for l in range(12)]
        else:
            lanes = rng.integers(0, 12, nveh); c = np.zeros(12); np.add.at(c, lanes, 1); confs = [c]
        for c in confs:
            for cur in range(4):
                for age in [2, 12]:
                    r = clear_time(c, cur, age); n += 1
                    if r is None:
                        stuck += 1
                        if len(examples) < 8: examples.append((np.nonzero(c)[0].tolist(), cur, age))
                    else: times.append(r)
print(sys.argv[1], 'configs', n, 'never cleared in 30 min:', stuck, ' mean clear time (s) of cleared', round(np.mean(times), 1), ' max', max(times))
print('  examples (lanes, start phase, start age):', examples)
