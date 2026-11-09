import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
w0 = tools.load('start_weights.npz')
w = tools.load('work/gate_v3.npz')
tools.save(w, 'work/weights.npz')
wf = tools.load('work/weights.npz')
print('edit size', tools.edit_size(wf, w0))
for k in tools.KEYS:
    d = np.argwhere(wf[k] != w0[k])
    print(k, 'changed', len(d), 'rows' if wf[k].ndim == 2 else '', sorted(set(int(r[0]) for r in d)) if wf[k].ndim == 2 else d.ravel().tolist())
print('W2 changed entries:', [(int(i), int(j), float(wf['W2'][i, j])) for i, j in np.argwhere(wf['W2'] != w0['W2'])])
print('W3 changed columns:', sorted(set(int(j) for _, j in np.argwhere(wf['W3'] != w0['W3']))))
print('b3 old', w0['b3'].round(3), 'new', wf['b3'].round(3))
# recorded deadlock (start_network episode 2, t >= 3625): what does the edited net do?
S = tools.states(); X = S['obs'].astype(np.float64)
m = (S['controller'] == 0) & (S['episode'] == 2) & (S['time_s'] >= 3625) & (S['time_s'] <= 3640)
print('recorded lock-up states: original picks', tools.greedy(w0, X[m]), ' edited picks', tools.greedy(wf, X[m]))
print('logits edited at t=3625:', tools.forward(wf, X[m][0]).round(2))
print(tools.trainability(wf))
json.dump({"low": 34.8, "nominal": 44.7, "high": 54.2}, open('work/predictions.json', 'w'))
print(open('work/predictions.json').read())
