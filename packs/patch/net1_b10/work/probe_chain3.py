"""Like probe_chain2 but with 1-6 stranded halting vehicles (random lanes) plus 0-2 'moving' vehicles that sit on
stop-line lanes as density only for the first 2 decisions, then halt. Reports never-cleared configurations."""
import numpy as np, sys
sys.path.insert(0, '.')
import tools
W = tools.load(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
capS = 22.4/7.5
SERVE = np.array([0, 0, 1, 2, 2, 3, 0, 0, 1, 2, 2, 3])
def obs(cnt, mov, cur, age):
    x = np.zeros(46); x[40+cur] = 1; x[44] = 1.0 if age >= 10 else 0.0; x[45] = min(age/60, 1)
    x[:12] = np.minimum(cnt/capS, 1); x[12:24] = np.minimum((cnt + mov)/capS, 1)
    return x
def run(cnt, mov, cur, age, maxsteps=360):
    cnt = cnt.astype(float).copy(); mov = mov.astype(float).copy()
    for step in range(maxsteps):
        cnt[SERVE == cur] = 0; mov[SERVE == cur] = 0
        if step == 2: cnt += mov; mov[:] = 0
        if cnt.sum() + mov.sum() == 0: return step*5
        pick = int(tools.greedy(W, obs(cnt, mov, cur, age))[0])
        if pick != cur and age >= 10: cur, age = pick, 2
        else: age += 5
    return None
rng = np.random.default_rng(seed)
n = 0; stuck = 0; times = []; ex = []
for trial in range(1500):
    nh = rng.integers(1, 7); nm = rng.integers(0, 3)
    cnt = np.zeros(12); np.add.at(cnt, rng.integers(0, 12, nh), 1)
    mov = np.zeros(12); np.add.at(mov, rng.integers(0, 12, nm), 1)
    cur = int(rng.integers(0, 4)); age = int(rng.choice([2, 7, 12, 17]))
    r = run(cnt, mov, cur, age); n += 1
    if r is None:
        stuck += 1
        if len(ex) < 5: ex.append((np.nonzero(cnt)[0].tolist(), np.nonzero(mov)[0].tolist(), cur, age))
    else: times.append(r)
print(f'{sys.argv[1]:32s} configs {n} never cleared: {stuck}  mean clear {np.mean(times):.1f} s  p95 {np.percentile(times,95):.0f}  max {max(times)}  ex {ex[:3]}')
