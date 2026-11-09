"""Stranded groups of 1-14 halting vehicles; a stop-line lane holds 3, the excess queues on its upstream lane
(lanes 1 and 2 share upstream lane 1). Served lanes empty when their phase is green. Never-cleared count."""
import numpy as np, sys
sys.path.insert(0, '.')
import tools
W = tools.load(sys.argv[1]); seed = int(sys.argv[2])
capS = 22.4/7.5; capU = 256/7.5
SERVE = np.array([0, 0, 1, 2, 2, 3, 0, 0, 1, 2, 2, 3])
UP = np.array([0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7])     # stop-line lane -> upstream lane index
def obs(cnt, cur, age):
    x = np.zeros(46); x[40+cur] = 1; x[44] = 1.0 if age >= 10 else 0.0; x[45] = min(age/60, 1)
    sl = np.minimum(cnt, 3); ex = cnt - sl
    x[:12] = np.minimum(sl/capS, 1); x[12:24] = x[:12]
    up = np.zeros(8); np.add.at(up, UP, ex)
    x[24:32] = np.minimum(up/capU, 1); x[32:40] = x[24:32]
    return x
def run(cnt, cur, age, maxsteps=360):
    cnt = cnt.astype(float).copy()
    for step in range(maxsteps):
        if age >= 2: cnt[SERVE == cur] = np.maximum(cnt[SERVE == cur] - 6, 0)   # ~6 per lane per 5 s window max
        if cnt.sum() == 0: return step*5
        pick = int(tools.greedy(W, obs(cnt, cur, age))[0])
        if pick != cur and age >= 10: cur, age = pick, 2
        else: age += 5
    return None
rng = np.random.default_rng(seed)
stuck = 0; times = []; ex = []; held = np.zeros(4, int)
for trial in range(2000):
    nh = rng.integers(1, 15)
    cnt = np.zeros(12); np.add.at(cnt, rng.integers(0, 12, nh), 1)
    cur = int(rng.integers(0, 4)); age = int(rng.choice([2, 7, 12, 17]))
    r = run(cnt, cur, age)
    if r is None:
        stuck += 1
        if len(ex) < 4: ex.append((cnt.astype(int).tolist(), cur))
    else: times.append(r)
print(f'{sys.argv[1]:32s} never cleared {stuck}/2000  mean clear {np.mean(times):.1f}s  max {max(times)}  ex {ex[:2]}')
