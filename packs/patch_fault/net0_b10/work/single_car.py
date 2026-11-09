"""For each lane, put one halting car (q = d = 1/capacity) in an otherwise empty junction and follow the
closed-loop phase sequence the network would produce (deterministic, ignoring the car leaving)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
names = tools.names()
serve = {}  # lane -> phases that give it green
for a, (th, lt) in {'N': (0, 1), 'S': (0, 1), 'E': (2, 3), 'W': (2, 3)}.items():
    serve[a + '0'] = th; serve[a + '1'] = th; serve[a + '2'] = lt
    serve[a + '0u'] = th; serve[a + '1u'] = None  # shared straight+left upstream
def sim(w, lanes, steps=80, ph0=2):
    ph, age, seq = ph0, 12, []
    for _ in range(steps):
        x = np.zeros(46)
        for i, v in lanes: x[i] = v
        x[40 + ph] = 1; x[44] = 1.0 if age >= 10 else 0.0; x[45] = min(age / 60, 1)
        a = int(tools.forward(w, x)[0].argmax())
        if a != ph and age >= 10:
            ph, age = a, 2; seq.append(a)
        else:
            age += 5
    return seq
for f in sys.argv[1:]:
    w = tools.load(f)
    print('==', f)
    for lane in range(12):
        cap = 22.4 / 7.5
        L = names[lane][2:]
        if lane == 11: continue
        lanes = [(lane, 1 / cap), (lane + 12, 1 / cap)]
        for ph0 in range(4):
            seq = sim(w, lanes, ph0=ph0)
            need = serve[L]
            served = (need == ph0) or (need in seq)
            if not served:
                print(f'  lane {L}: one car, start phase {ph0}: NEVER served; phase sequence {seq[:12]}')
    for k, lane in enumerate(range(24, 32)):
        cap = 256 / 7.5
        L = names[lane][2:]
        for n in [1, 3]:
            lanes = [(lane, n / cap), (lane + 8, n / cap)]
            for ph0 in range(4):
                seq = sim(w, lanes, ph0=ph0)
                print(f'  upstream {L} {n} car(s) start ph {ph0}: seq {seq[:8]}') if len(set(seq + [ph0])) < 2 or (serve[L] is not None and serve[L] not in seq + [ph0]) else None
