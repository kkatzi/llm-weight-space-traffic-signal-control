import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; act = S["act"]; ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
for c in range(3):
    print("==== controller", S["controller_names"][c])
    durs = {p: [] for p in range(4)}
    seqs = []
    for e in np.unique(ep[ctrl == c]):
        m = (ctrl == c) & (ep == e)
        idx = np.where(m)[0]
        idx = idx[np.argsort(t[idx])]
        ph = X[idx, 40:44].argmax(1)
        tg = X[idx, 45] * 60
        tt = t[idx]
        # find green periods
        start = 0
        seq = []
        for k in range(1, len(idx) + 1):
            if k == len(idx) or ph[k] != ph[k - 1] or tg[k] < tg[k - 1]:
                durs[ph[k - 1]].append(tg[k - 1])
                seq.append((ph[k - 1], round(tg[k - 1])))
                start = k
        seqs.append(seq)
        if e == np.unique(ep[ctrl == c])[0]:
            print("time range", tt.min(), tt.max(), "n", len(idx), "dt", np.unique(np.diff(tt))[:5])
            print("seq first 40:", seq[:40])
    for p in range(4):
        d = np.array(durs[p])
        print(f" phase {p}: n={len(d)} mean last-tg {d.mean():.1f} median {np.median(d):.1f} max {d.max():.1f}")
    # transitions
    T = np.zeros((4, 4), int)
    for seq in seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            T[a[0], b[0]] += 1
    print(" transitions\n", T)
