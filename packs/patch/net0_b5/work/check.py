import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import build, demand
np.set_printoptions(linewidth=220, precision=3, suppress=True)

def report(w, w0, S, label=''):
    X = S['obs'].astype(np.float64); C = S['controller']; ep = S['episode']; T = S['time_s']
    ok = X[:, 44] > 0.5
    dead = (C == 0) & (ep == 2) & (T > 3620)
    Dm = demand(X); E = Dm == 0
    g0 = tools.greedy(w0, X); g1 = tools.greedy(w, X)
    lg0 = tools.forward(w0, X)
    anyE = E.any(1) & ~E.all(1)
    norm = ok & ~anyE & ~dead
    out = {'edit': tools.edit_size(w, w0)['changed'],
           'agree_ok_noempty': float((g0[norm] == g1[norm]).mean()),
           'n_ok_noempty': int(norm.sum())}
    # states with some (not all) phases empty: reference = original choice restricted to non-empty phases
    m = ok & anyE
    ref = np.where(E[m], -1e9, lg0[m]).argmax(1)
    out['picks_nonempty'] = float((~E[m][np.arange(m.sum()), g1[m]]).mean())
    out['agree_restricted'] = float((g1[m] == ref).mean())
    out['n_ok_someempty'] = int(m.sum())
    allE = ok & E.all(1)
    out['agree_allempty'] = float((g0[allE] == g1[allE]).mean())
    for c in range(3):
        mm = norm & (C == c)
        out[f'agree_noempty_{S["controller_names"][c]}'] = float((g0[mm] == g1[mm]).mean())
    print(label, out)
    return out

if __name__ == '__main__':
    S = tools.states(); w0 = tools.load('start_weights.npz')
    report(w0, w0, S, 'orig')
    report(build(), w0, S, 'default')
    report(build(comp_b2=False), w0, S, 'no b2 comp')
