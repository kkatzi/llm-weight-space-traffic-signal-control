"""Checks of the repaired network (forward passes only, no simulator, no fitting)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import tools  # noqa: E402
import build  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)
n = tools.names(); I = n.index
S = tools.states(); X = S["obs"].astype(np.float64); t = S["time_s"]; ctrl = S["controller"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
w0 = tools.load("start_weights.npz")
CAP_U = 256 / 7.5


def synth(phase, tg_s, **kw):
    x = np.zeros(46); x[40 + phase] = 1; x[45] = min(tg_s / 60, 1); x[44] = 1.0 if tg_s >= 10 else 0.0
    for k, v in kw.items():
        x[I(k)] = v
    return x


def report(w, label):
    print("=" * 30, label, tools.edit_size(w, w0))
    h1, h2 = tools.hidden(w, X)
    idle_on = h1[:, build.U_IDLE] > 0; block_on = h1[:, build.U_BLOCK] > 0
    lane = X[:, :40].copy(); lane[:, [31, 39]] = 0
    only_w1u_halted = (lane.sum(1) == 0) & (np.abs(X[:, 39] - X[:, 31]) < 1e-9)
    sig = (ph == 2) & (X[:, 31] > 0)
    print("IDLE fires on %d recorded states; all of them empty-except-halted-W1u: %s; misses: %d; in 300-3600 s: %d"
          % (idle_on.sum(), bool(only_w1u_halted[idle_on].all()), (only_w1u_halted & ~idle_on).sum(),
             (idle_on & (t >= 300) & (t <= 3600)).sum()))
    print("BLOCK fires on %d recorded states; == (phase2 & q_W1u>0): %s" % (block_on.sum(), bool((block_on == sig).all())))
    print("IDLE |h| in off states min %.5f, BLOCK |h| off min %.5f" %
          (np.abs(h1[~idle_on, build.U_IDLE]).min(), np.abs(h1[~block_on, build.U_BLOCK]).min()))
    # 1. idle synthetic states
    bad = []
    for p in range(4):
        for tg in range(2, 121, 5):
            for extra in [{}, {"q_W1u": 3 / CAP_U, "d_W1u": 3 / CAP_U}, {"q_W1u": 6 / CAP_U, "d_W1u": 6 / CAP_U},
                          {"q_W1u": 20 / CAP_U, "d_W1u": 20 / CAP_U}]:
                x = synth(p, tg, **extra)
                lg = tools.forward(w, x)[0]
                if x[44] > 0.5 and lg.argmax() != 3:
                    bad.append((p, tg, extra, lg))
                srt = np.sort(lg)
    print("idle synthetic states (ok=1) not choosing EW_left:", len(bad), bad[:3])
    m = []
    for p in range(4):
        for tg in range(12, 121, 5):
            lg = tools.forward(w, synth(p, tg))[0]; m.append(lg[3] - np.delete(lg, 3).max())
    print("  idle margin of EW_left: min %.2f" % min(m))
    # single vehicle anywhere -> idle unit must be off
    worst = -1
    for i in [j for j in range(40) if j not in (11, 23, 31)]:
        v = 1 / CAP_U if i >= 24 else 1 / 2.987
        for p in range(4):
            x = synth(p, 30); x[i] = v
            if i == 31:
                continue
            h1s, _ = tools.hidden(w, x)
            worst = max(worst, h1s[0, build.U_IDLE])
    print("  max IDLE activation with exactly one visible (non-halted-W1u) vehicle: %.5f" % worst)
    # 2. deadlock state of recorded start_network episode 5 (only W1u halted, phase 2)
    dl = (t > 3700) & only_w1u_halted & (X[:, 31] > 0)
    print("recorded deadlock states (only W1u halted, t>3700): n=%d, new decisions %s, old %s" %
          (dl.sum(), np.bincount(tools.greedy(w, X[dl]), minlength=4), np.bincount(tools.greedy(w0, X[dl]), minlength=4)))
    em = (t > 3600) & (X[:, :40].sum(1) == 0)
    print("recorded all-empty states after 3600 s: n=%d, new %s, old %s" %
          (em.sum(), np.bincount(tools.greedy(w, X[em]), minlength=4), np.bincount(tools.greedy(w0, X[em]), minlength=4)))
    # 3. blockage states in-hour vs. 'W2 full' emulation with the original network
    bs = sig & ok & (t <= 3600)
    Yf = X[bs].copy(); Yf[:, 11] = 1; Yf[:, 23] = 1
    ref = tools.greedy(w0, Yf); new = tools.greedy(w, X[bs])
    print("in-hour blockage states n=%d: new %s | orig-net with W2 full %s | agreement %.3f | orig faulty %s" %
          (bs.sum(), np.bincount(new, minlength=4), np.bincount(ref, minlength=4), (new == ref).mean(),
           np.bincount(tools.greedy(w0, X[bs]), minlength=4)))
    # 4. all other honoured states: effect of the proxy and the side effects of the repurposing
    rest = ok & ~idle_on & ~block_on
    Xp = X.copy(); Xp[:, 11] = X[:, 5]; Xp[:, 23] = X[:, 5]
    a_new = tools.greedy(w, X); a_proxy = tools.greedy(w0, Xp); a_old = tools.greedy(w0, X)
    print("other honoured states n=%d: agree with orig+proxy-input %.4f, with orig faulty %.4f" %
          (rest.sum(), (a_new[rest] == a_proxy[rest]).mean(), (a_new[rest] == a_old[rest]).mean()))
    print("honoured decisions share by phase: orig faulty %s new %s" %
          (np.bincount(a_old[ok], minlength=4) / ok.sum(), np.bincount(a_new[ok], minlength=4) / ok.sum()))
    # how often 'switch away from current green' is requested (honoured states)
    print("switch requests among honoured: orig %.3f new %.3f" % ((a_old[ok] != ph[ok]).mean(), (a_new[ok] != ph[ok]).mean()))
    print("trainability", tools.trainability(w))
    return bs


if __name__ == "__main__":
    for pb in [0.0, 2.0, 3.0, 4.0, 6.0, 7.5]:
        w, _ = build.build(p_block=pb, verbose=False)
        Yf = None
        bs = (ph == 2) & (X[:, 31] > 0) & ok & (t <= 3600)
        Yf = X[bs].copy(); Yf[:, 11] = 1; Yf[:, 23] = 1
        ref = tools.greedy(w0, Yf); new = tools.greedy(w, X[bs])
        print("P_BLOCK=%.1f: in-hour blockage decisions %s, agreement with orig-net(W2 full) %.3f" %
              (pb, np.bincount(new, minlength=4), (new == ref).mean()))
    w = tools.load(os.path.join(HERE, "weights.npz"))
    report(w, "delivered work/weights.npz")
