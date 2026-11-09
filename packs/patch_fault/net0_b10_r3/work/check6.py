import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build5 import build
S = tools.states(); X = S['obs'].astype(np.float64); T=S['time_s']; C=S['controller']
ok = X[:,44]>0.5; mh = ok&(T<3600)
m0=(C==0)&(T<3600)
print("start-net in-hour means q_E2 %.3f d_E2 %.3f" % (X[m0,5].mean(), X[m0,17].mean()))
w5,_ = build(); a5 = tools.greedy(w5, X)
wv5 = tools.load("work/v5_k6.npz"); print("build5 default == v5 file:", max(np.abs(wv5[k]-w5[k]).max() for k in w5))
for cfg in [dict(alpha=0.5), dict(alpha=0.75)]:
    w,w0 = build(**cfg); a = tools.greedy(w,X)
    print(cfg, f"changed vs v5 {(a[mh]!=a5[mh]).mean():.4f} dist {np.bincount(a[mh],minlength=4)} vs {np.bincount(a5[mh],minlength=4)}", tools.edit_size(w,w0))
