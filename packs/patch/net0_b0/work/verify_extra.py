from common import *
w0 = tools.load("../start_weights.npz"); w = tools.load("weights.npz")
# all-empty junction, every phase / t_green: patched choice vs original
same = 0; n = 0
for p in range(4):
    for tg in range(0, 65, 5):
        x = make_state([], p, tg); n += 1
        same += int(tools.greedy(w, x)[0] == tools.greedy(w0, x)[0])
print('all-empty junction: same choice in %d of %d (phase, t_green) states' % (same, n))
# single approaching vehicle on one upstream lane only (others empty): does it pick a phase that serves it?
for u in range(8):
    for p in range(4):
        x = make_state([], p, 60); x[32 + u] = ONEU
        a = int(tools.greedy(w, x)[0]); a0 = int(tools.greedy(w0, x)[0])
        serves = [q for q in range(4) if (32 + u) in DL[q]]
        if a not in serves: print('  upstream lane', u, 'phase', p, 'patched picks', a, 'serving phases', serves, 'orig', a0)
print('upstream-only check done')
import numpy as np
z = np.load('weights.npz'); print({k: (z[k].dtype, z[k].shape) for k in z.files})
