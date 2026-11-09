import sys; sys.path.insert(0, '.')
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=220)
exec(open('work/explore24.py').read().split("g = tools.greedy")[0])
lg = tools.forward(w, obs); g = lg.argmax(1)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
H = W(obs, near=0.0)
srt = np.sort(lg, 1); gap = srt[:, -1] - srt[:, -2]
for G in [0, 10, 20, 30, 45]:
    idxv = H - G*lam
    sm = idxv.copy(); sm[I, ph] = -1e9; s = sm.argmax(1)
    out = []
    for c in range(3):
        m = ok & (ctl == c) & (g != ph)
        out.append(f"ctl{c} agree={(s[m]==g[m]).mean():.2f}")
    m = ok & (ctl == 0) & (g != ph)
    cnt = Counter((g[i], s[i]) for i in np.where(m & (s != g))[0])
    print(f"G={G}: ", " ".join(out), cnt.most_common(6))
