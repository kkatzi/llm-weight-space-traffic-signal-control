"""Reproduce work/weights.npz from start_weights.npz (max-green safety gate, phases 2 and 3, non-destructive
carriers; see maxgreen3.py / maxgreen4.py for the construction)."""
import sys, importlib.util
sys.path.insert(0, '.')
import numpy as np, tools
spec = importlib.util.spec_from_file_location('mg4', 'work/maxgreen4.py'); mg4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg4)
w, plan = mg4.mg3.build(phases=(2, 3), t_units=(31, 0), n_carriers=20)
tools.save(w, 'work/weights.npz')
wb = tools.load('work/weights.npz'); w0 = tools.load('start_weights.npz')
print('edit size', tools.edit_size(wb, w0))
print('identical to evaluated v6:', all(np.array_equal(wb[k], tools.load('work/v6_gate23nd_n20.npz')[k]) for k in tools.KEYS))
for k in tools.KEYS:
    d = np.argwhere(wb[k] != w0[k]); print(k, 'changed', len(d))
for p, car in plan.items(): print('phase', p, 'carriers', [(c[0], int(c[1])) for c in car])
