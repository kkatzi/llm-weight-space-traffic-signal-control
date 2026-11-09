import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build4 import build
S = tools.states(); X = S['obs'].astype(np.float64); T=S['time_s']
ok = X[:,44]>0.5; mh = ok&(T<3600)
w4,_ = build(k=3); a4 = tools.greedy(w4, X)
for cfg in [dict(k=4.5), dict(k=6), dict(k=8)]:
    w,w0 = build(**cfg); a = tools.greedy(w,X)
    imp = X[:,5] + cfg['k']*X[:,31]
    print(cfg, f"changed vs v4 {(a[mh]!=a4[mh]).mean():.4f} dist {np.bincount(a[mh],minlength=4)} vs {np.bincount(a4[mh],minlength=4)} imp>1.3 {(imp[mh]>1.3).mean():.3f} imp>1.6 {(imp[mh]>1.6).mean():.3f}")
