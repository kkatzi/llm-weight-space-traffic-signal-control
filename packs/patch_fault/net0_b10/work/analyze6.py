import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
wp = tools.load('work/proxy_a1.npz')
cap = 22.4 / 7.5
names = tools.names()
# number of halting cars in one NS through lane (q=d=n/cap) needed to leave a phase-2 hold, empty otherwise
for lane in [0, 1, 6, 7]:
    for tg in [0.2, 0.5, 1.0]:
        row = []
        for n in [1, 2, 3]:
            x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = tg; x[lane] = n / cap; x[lane + 12] = n / cap
            lg = tools.forward(wp, x)[0]
            row.append((int(lg.argmax()), round(lg[0] - lg[2], 2)))
        print(names[lane], 'tg', tg, '[(choice, logit0-logit2) for 1,2,3 cars]', row)
# two lanes: e.g. N0 and S0 one car each
x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = 1.0; x[[0, 6]] = 1 / cap; x[[12, 18]] = 1 / cap
print('N0+S0 one car each', tools.forward(wp, x)[0].round(2))
# continuous sweep: add the same value v to q and d of N0,N1,S0,S1 in the EMPTY state; where does phase 0 win?
for tg in [0.2, 1.0]:
    out = []
    for v in np.linspace(0, 0.5, 11):
        x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = tg
        for i in [0, 1, 6, 7]: x[i] += v; x[i + 12] += v
        lg = tools.forward(wp, x)[0]; out.append((round(v, 2), int(lg.argmax())))
    print('empty + v on all 4 NS-through q,d; tg', tg, out)
