import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "work"))
exec(open(os.path.join(HERE, "work", "compare.py")).read().split("for args in")[0])
from build import src_vec, K, IX
du = X[:, 32:40]*capu
print("upstream moving veh per lane: mean", du.mean(0).round(2), " q90", np.quantile(du, .9, axis=0).round(1))
def build_h2(rho, split=0.5):
    w = {k: v.copy() for k, v in w0.items()}
    for a in "NESW":
        # up lane 0 -> lane 0 density ; up lane 1 -> split between lane 1 and pocket
        w["W1"][:, IX[f"d_{a}0u"]] += rho * K * w0["W1"][:, IX[f"d_{a}0"]]
        w["W1"][:, IX[f"d_{a}1u"]] += rho * K * ((1-split) * w0["W1"][:, IX[f"d_{a}1"]] + split * w0["W1"][:, IX[f"d_{a}2"]])
    return w
for rho in [0.1, 0.25, 0.5]:
    w = build_h2(rho)
    g = tools.greedy(w, X)
    ch = ok & (g != g0)
    print(f"rho {rho}: honoured decisions changed {ch.mean():.3f}; net eps {(ch & (ctrl==0)).mean()/ (ok&(ctrl==0)).mean():.3f}; stays {(ok & (g==ph)).sum()} (orig {(ok & (g0==ph)).sum()})")
    # in changed decisions, does the new pick have more upstream moving vehicles?
    G = {0: [0, 1, 4, 5], 1: [1, 5], 2: [2, 3, 6, 7], 3: [3, 7]}
    DU = np.stack([du[:, G[p]].sum(1) for p in range(4)], 1)
    r = np.arange(len(X))
    print("   among changed: mean DU new pick", DU[r, g][ch].mean().round(2), "old pick", DU[r, g0][ch].mean().round(2))
