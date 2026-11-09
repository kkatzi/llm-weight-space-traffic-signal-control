"""Build work/weights.npz: repurpose one layer-1 unit (i) and one layer-2 unit (j) into a detector for
'in phase 2 (EW through), min green elapsed, EW-through stop-line lanes empty, few EW-through cars upstream,
and at least one halting car on another lane' and, when it fires, subtract 2*D from logit 2 so the net leaves phase 2.
All values set by hand; see NOTES.md."""
import sys, numpy as np, tools
N = tools.names(); idx = {n:k for k,n in enumerate(N)}
K, M, D = 60.0, 10.0, 10.0

def build(i, j, compensate=True):
    w0 = tools.load("start_weights.npz"); w = {k:v.copy() for k,v in w0.items()}
    h1, h2 = tools.hidden(w0, tools.states()["obs"])
    # layer-1 unit i: a = K*( 3*(ok-1) + 3*(phase2-1) + 0.3*sum(other queues) - 30*sum(EW-through stop-line density)
    #                        - 2*sum(EW-through upstream density) - 0.05 )
    row = np.zeros(46)
    row[idx["ok"]] = 3*K; row[idx["phase2"]] = 3*K
    for n in ["q_N0","q_N1","q_N2","q_E2","q_S0","q_S1","q_S2","q_W2"]: row[idx[n]] = 0.3*K
    for n in ["d_E0","d_E1","d_W0","d_W1"]: row[idx[n]] = -30*K
    for n in ["d_E0u","d_E1u","d_W0u","d_W1u"]: row[idx[n]] = -2*K
    w["W1"][i] = row; w["b1"][i] = (-3-3-0.05)*K
    # remove unit i's old influence on layer 2 (compensate with its mean activation), route it only to unit j
    if compensate: w["b2"] += w0["W2"][:, i] * h1[:, i].mean()
    w["W2"][:, i] = 0
    # layer-2 unit j: reads only unit i
    w["W2"][j, :] = 0; w["W2"][j, i] = M; w["b2"][j] = 0.0
    # output: unit j lowers logit 2 by 2D when firing (h2_j=+1), nothing when not (h2_j=-1)
    if compensate: w["b3"] += w0["W3"][:, j] * h2[:, j].mean()
    w["W3"][:, j] = 0; w["W3"][2, j] = -D; w["b3"][2] += -D
    # tie-break: when firing, also raise logit 0 (NS through) by 1.0 so a lone N/S through car is served
    # rather than cycling 2 -> 3 -> 2 (zero when not firing)
    w["W3"][0, j] = 0.5; w["b3"][0] += 0.5
    return w0, w

def condition(X):
    ok = X[:,44]>0.5; ph2 = X[:,42]>0.5
    oq = X[:,[idx[n] for n in ["q_N0","q_N1","q_N2","q_E2","q_S0","q_S1","q_S2","q_W2"]]].sum(1)
    dew = X[:,[idx[n] for n in ["d_E0","d_E1","d_W0","d_W1"]]].sum(1)
    dup = X[:,[idx[n] for n in ["d_E0u","d_E1u","d_W0u","d_W1u"]]].sum(1)
    return ok & ph2 & (0.3*oq - 30*dew - 2*dup - 0.05 > 0)

if __name__ == "__main__":
    S = tools.states(); X = S["obs"]; ok = X[:,44]>0.5
    cond = condition(X)
    print("condition true on", cond.sum(), "recorded states")
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        g0 = tools.greedy(tools.load("start_weights.npz"), X)
        cands_i = [4,55,16,28,37,13,21,32,14,52]; cands_j = [45,5,25,39,40,38,42,44,21,48]
        out=[]
        for i in cands_i:
            for j in cands_j:
                w0,w = build(i,j)
                g = tools.greedy(w,X)
                flips = (g!=g0) & ~cond & ok
                out.append((flips.sum(), i, j, int(((g==2)&cond).sum())))
        out.sort(); print(out[:10])
    else:
        i, j = int(sys.argv[1]), int(sys.argv[2])
        w0, w = build(i, j)
        g0, g = tools.greedy(w0,X), tools.greedy(w,X)
        print("edit size:", tools.edit_size(w, w0))
        print("flips on non-condition states (ok):", int(((g!=g0)&~cond&ok).sum()), "of", int((~cond&ok).sum()), "; all states:", int(((g!=g0)&~cond).sum()))
        print("condition states where new net still picks 2:", int(((g==2)&cond).sum()), "; picks:", np.bincount(g[cond],minlength=4))
        print("old net picks on condition states:", np.bincount(g0[cond],minlength=4))
        print(tools.trainability(w))
        tools.save(w, "work/weights.npz"); print("saved work/weights.npz")
