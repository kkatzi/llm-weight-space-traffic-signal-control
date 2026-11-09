from common import *
w0 = tools.load("../start_weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64); ok = X[:, 44] > 0.5; t = S['time_s']; ep = S['episode']; ctrl = S['controller']
E = phase_empty(X); g = tools.greedy(w0, X)
pol0 = lambda x: tools.greedy(w0, x)
for c in range(3):
    for e in range(6):
        m = (ctrl == c) & (ep == e)
        Xe = X[m]; te = t[m]
        # total vehicles (approx) over time: stop-line density*3 + upstream density*34.13
        nveh = Xe[:, 12:24].sum(1) * (22.4 / 7.5) + Xe[:, 32:40].sum(1) * (256 / 7.5)
        bad = ok[m] & E[m][np.arange(m.sum()), g[m]] & (~E[m]).any(1) & (te >= 3600)
        # static-deadlock test on the end-of-episode states: halting vehicles only at stop line, no upstream
        stuck_states = 0
        for i in np.where(te >= 3600)[0]:
            x = Xe[i]
            if x[32:40].sum() == 0 and x[12:24].sum() > 0:
                lanes = [l for l in range(12) if x[12 + l] > 0]; counts = [int(round(x[12 + l] * 22.4 / 7.5)) for l in lanes]
                r, _ = trace(pol0, lanes, counts, start=int(x[40:44].argmax()))
                if r != 'served': stuck_states += 1
        print('%-13s ep %d end %4d  mean veh(t<3600) %5.1f  net-would-pick-empty states %3d  static-deadlock-prone end states %3d' %
              (S['controller_names'][c], e, te.max(), nveh[te < 3600].mean(), bad.sum(), stuck_states))
