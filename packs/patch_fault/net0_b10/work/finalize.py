"""Write work/weights.npz from make_final.build(vW=0.67, vN=0.35) and run the final checks."""
import numpy as np, sys, os, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_final import build
w0 = tools.load('start_weights.npz')
w = build(0.67, 0.35)
tools.save(w, 'work/weights.npz')
wf = tools.load('work/weights.npz')
ref = tools.load('work/final_w067_n035.npz')
print('identical to evaluated file (call 4):', all(np.array_equal(wf[k], ref[k]) for k in wf))
print('edit size:', tools.edit_size(wf, w0))
for k in tools.KEYS:
    d = np.abs(wf[k] - w0[k]) > 0
    print(f'  {k}: {int(d.sum())} changed')
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']; T = S['time_s']
ok = X[:, 44] > 0.5
# detector unit 57 must be OFF (-1) on every recorded main-hour state and ON only on >=60 s phase-2 holds
h1, _ = tools.hidden(wf, X)
on = h1[:, 57] > 0
print('detector ON states:', int(on.sum()), '| all are phase-2 with t_green==1:', bool(np.all((X[on, 42] == 1) & (X[on, 45] >= 0.999))),
      '| main-hour ON states:', int((on & (T < 3600)).sum()))
print('OFF level range', h1[~on, 57].min(), h1[~on, 57].max())
# equivalence: away from the detector, the patched net on failed obs ~= original net on imputed obs (x11:=x5, x23:=x17)
Xi = X.copy(); Xi[:, 11] = X[:, 5]; Xi[:, 23] = X[:, 17]
g_imp = tools.greedy(w0, Xi); g = tools.greedy(wf, X)
m = ok & ~on
print('agreement with original-net-on-imputed-obs (honoured, detector OFF): %.4f' % np.mean(g[m] == g_imp[m]))
print('agreement with start network on failed obs (honoured): %.4f' % np.mean(g[ok] == tools.greedy(w0, X)[ok]))
print('trainability:', tools.trainability(wf))
print('sha1', hashlib.sha1(open('work/weights.npz', 'rb').read()).hexdigest()[:12])
