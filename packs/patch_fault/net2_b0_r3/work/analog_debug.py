import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analog_test.py")).read().split("report(\"fault")[0])
np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("qm_ph", qm_ph, "dm_ph", dm_ph, "qm_oth", qm_oth, dm_oth)
for name, w in [("nofix", w0), ("0.3 global", edit(0.3, "global")), ("0.3 serve", edit(0.3, "serve")), ("0.3 phase4", edit(0.3, "phase4")), ("0.7 global", edit(0.7, "global")), ("1.0", edit(1.0, "global"))]:
    a = tools.greedy(w, Xf)
    m = sn & ok
    print(name)
    # confusion truth vs a
    M = np.zeros((4, 4), int)
    for t, p in zip(truth[m], a[m]): M[t, p] += 1
    print("  confusion (rows truth, cols pred) start_net honoured:\n", M)
    for k in range(4):
        mk = m & (ph == k)
        print(f"  cur phase {k}: n={mk.sum()} agree {np.mean(a[mk]==truth[mk]):.3f}")
