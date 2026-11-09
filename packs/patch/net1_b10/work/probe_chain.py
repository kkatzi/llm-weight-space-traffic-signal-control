"""Deterministic decision chain for a lone halting vehicle in an otherwise empty junction (forward passes only).
State: (current phase, green age). Every 5 s the net decides; a switch is honoured only if age>=10 (ok);
then 3 s yellow and the new green is 2 s old at the next decision. The vehicle is served once its phase is green
(we count it served at the first decision where its phase is current)."""
import numpy as np, sys
sys.path.insert(0, '.')
import tools
W = tools.load(sys.argv[1] if len(sys.argv) > 1 else 'start_weights.npz')
capS = 22.4/7.5; capU = 256/7.5
SERVE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 0, 7: 0, 8: 1, 9: 2, 10: 2, 11: 3}
lanes = tools.names()[:12]
def obs(lane, cur, age, nveh=1, mode='halt'):
    x = np.zeros(46); x[40+cur] = 1; x[44] = 1.0 if age >= 10 else 0.0; x[45] = min(age/60, 1)
    if mode == 'halt':
        x[lane] = nveh/capS; x[12+lane] = nveh/capS
    return x
def chain(lane, cur, age, nveh=1, maxsteps=360):
    for step in range(maxsteps):          # 360 decisions = 30 min
        if cur == SERVE[lane]:
            return step*5
        pick = int(tools.greedy(W, obs(lane, cur, age, nveh))[0])
        if pick != cur and age >= 10:
            cur = pick; age = 2
        else:
            age += 5
    return None
stuck = []; waits = []
for nveh in [1, 2]:
    for lane in range(12):
        for cur in range(4):
            if cur == SERVE[lane]: continue
            for age in [2, 7, 12, 17, 22]:
                r = chain(lane, cur, age, nveh)
                if r is None: stuck.append((nveh, lanes[lane], cur, age))
                else: waits.append(r)
print('stuck configurations (nveh, lane, start phase, start age):', len(stuck), 'of', 2*12*3*5)
for s in stuck: print('  ', s)
print('wait (s) when served: mean', np.mean(waits), ' max', np.max(waits))
