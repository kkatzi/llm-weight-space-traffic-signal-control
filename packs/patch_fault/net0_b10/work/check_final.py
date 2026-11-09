import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_final import build
names = tools.names(); capS = 22.4 / 7.5
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
ok = X[:, 44] > 0.5; cur = X[:, 40:44].argmax(1)
serve = {'N0': 0, 'N1': 0, 'S0': 0, 'S1': 0, 'N2': 1, 'S2': 1, 'E0': 2, 'E1': 2, 'W0': 2, 'W1': 2, 'E2': 3, 'W2': 3}
def sim(w, x_lanes, ph=2, steps=120):
    age, t, seq = 12, 0, []
    for _ in range(steps):
        x = np.zeros(46)
        for i, v in x_lanes: x[i] = v
        x[40 + ph] = 1; x[44] = float(age >= 10); x[45] = min(age / 60, 1)
        a = int(tools.forward(w, x)[0].argmax())
        if a != ph and age >= 10: seq.append((a, t)); ph, age = a, 2; t += 5
        else: age += 5; t += 5
    return seq
wp = tools.load('work/proxy_a1.npz'); gp = tools.greedy(wp, X)
main = ok & (T < 3600)
if __name__ == "__main__":
  for vW, vN in [(0.67, 0.35)]:
      w = build(vW, vN)
      g = tools.greedy(w, X)
      es = tools.edit_size(w, tools.load('start_weights.npz'))['changed']
      s = sim(w, [])
      rep = ['empty->' + ','.join(f'{a}@{t}' for a, t in s[:3])]
      bad = []
      for lane in range(11):
          L = names[lane][2:]
          for start in [1, 2, 3]:
              sq = sim(w, [(lane, 1 / capS), (lane + 12, 1 / capS)], ph=start)
              hit = [t for a, t in sq if a == serve[L]]
              if serve[L] != start and not hit: bad.append(f'{L}/ph{start}')
              elif serve[L] != start and start == 2: rep.append(f'{L}:{hit[0]}s')
      # one N/S-through car together with an E-W through car (phase 2 legitimately busy)
      sq = sim(w, [(0, 1 / capS), (12, 1 / capS), (4, 1 / capS), (16, 1 / capS)])
      rep.append('N0+E1->' + ','.join(f'{a}@{t}' for a, t in sq[:3]))
      print(f'vW {vW} vN {vN}: edits {es}, changed main-hour decisions net {int((g[main & (C==0)] != gp[main & (C==0)]).sum())} all {int((g[main] != gp[main]).sum())}/{main.sum()} | never served {bad}')
      print('    ', ' '.join(rep))
