import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build4 import build
S = tools.states(); X = S['obs'].astype(np.float64); T=S['time_s']; C=S['controller']
ok = X[:,44]>0.5; mh = ok&(T<3600)
w3,_ = build(); a3 = tools.greedy(w3, X)
q = X[:,31]
print("q_W1u>0 share", (q[mh]>0).mean(), "quantiles of q_W1u|>0", np.round(np.quantile(q[mh&(q>0)],[0.25,0.5,0.75,0.95]),3))
for k in [2,3,4]:
    w,w0 = build(k=k)
    a = tools.greedy(w,X)
    imp = X[:,5] + k*X[:,31]
    print(f"k={k}: changed honoured in-hour vs v3 {(a[mh]!=a3[mh]).mean():.4f}; dist {np.bincount(a[mh],minlength=4)} vs {np.bincount(a3[mh],minlength=4)};"
          f" imputed q_W2>1 share {(imp[mh]>1).mean():.3f}, >1.3 {(imp[mh]>1.3).mean():.3f}", tools.edit_size(w,w0))
