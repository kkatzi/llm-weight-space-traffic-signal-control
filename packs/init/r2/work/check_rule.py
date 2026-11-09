import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np, rule
np.set_printoptions(linewidth=200, precision=2, suppress=True)
S = tools.states()
X, A, E, T, C = S["obs"].astype(float), S["act"], S["episode"], S["time_s"], S["controller"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5; tg = np.round(X[:, 45] * 60).astype(int)
D = rule.scores(X); r = D.argmax(1)
M, _ = rule.score_matrix()
Draw = X @ M.T
print("D quantiles per phase (raw, no bonus):")
for p in range(4): print(p, np.quantile(Draw[:, p], [0.1, .5, .9, .99, 1]))
for c in range(2):
    m = (C == c) & ok
    print(f"ctrl{c}: ok states {m.sum()}, rule holds {(r[m]==ph[m]).mean():.3f}, ctrl holds {(A[m]==ph[m]).mean():.3f}, agree {(r[m]==A[m]).mean():.3f}")
    sw = m & (A != ph) & (r != ph)
    print(f"   among both-switch: agree on target {(r[sw]==A[sw]).mean():.3f}")
    # hold states by time of day
    hold = m & (r == ph)
    print("   rule-hold by time bucket:", [round(float(hold[m & (T // 900 == k)].sum() / max(1, (m & (T // 900 == k)).sum())), 2) for k in range(5)])
    print("   rule-hold t_green:", np.bincount(tg[hold], minlength=30)[[12, 17, 22, 27]])
    print("   rule picks distribution:", np.bincount(r[m], minlength=4) / m.sum())
