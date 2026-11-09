import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
def base(p, tg=12/60, okv=1.0, fill=0.0):
    x = np.zeros(46); x[40+p] = 1; x[44] = okv; x[45] = tg; x[0:24] = fill; x[32:40] = 0.05
    return x
for fill in [0.0, 1/3, 2/3]:
    print("fill", fill)
    for p in range(4):
        print(" phase", p, tools.forward(w, base(p, fill=fill))[0], "-> ", tools.greedy(w, base(p, fill=fill))[0])
# t_green distribution on ok states
tg = X[ok, 45]*60
print(np.unique(np.round(tg), return_counts=True))
# what about t_green in own controller ok states
c = S['controller']
for cc in range(3):
    m = ok & (c == cc)
    print(cc, np.unique(np.round(X[m, 45]*60), return_counts=True))
