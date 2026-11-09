"""N/S analog split by traffic load: does the imputation over- or under-serve at low / high load?"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "analog_test.py")).read().split("report(\"fault")[0])
load = X[:, 12:24].sum(1) + X[:, 32:40].sum(1) * 11  # rough vehicles-ish
qs = np.percentile(load[ok], [33, 67])
bins = [load <= qs[0], (load > qs[0]) & (load <= qs[1]), load > qs[1]]
print("load tercile cut points", qs.round(2))
variants = [("fault", w0), ("rho0.7 serve", edit(0.7, "serve")), ("rho0.85 serve", edit(0.85, "serve")), ("rho1.0", edit(1.0, "global")), ("rho0.5 serve", edit(0.5, "serve"))]
for name, w in variants:
    a = tools.greedy(w, Xf)
    out = [name.ljust(14)]
    for b, lab in zip(bins, ["low", "mid", "high"]):
        m = ok & b
        out.append(f"{lab}: agree {np.mean(a[m]==truth[m]):.3f} serve {np.mean(a[m]==SERVE):.3f} (truth {np.mean(truth[m]==SERVE):.3f})")
    print("  ".join(out))
