from common import *
import itertools
w = tools.load("../start_weights.npz")
for vs in [(), (2,), (2, 3), (1, 2, 3), (0, 1, 2, 3)]:
    for V in (40.0,):
        f, wasted = all_traces(make_policy(w, vs, V))
        print('veto', vs, 'fails', len(f), 'wasted-switch count', wasted, f[:5])
