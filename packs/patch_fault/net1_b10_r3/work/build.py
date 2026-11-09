"""Build candidate repairs by construction (no fitting).

The dead inputs x[11] (q_W2) and x[23] (d_W2) are always 0, so their W1 columns a11 = W1[:,11], a23 = W1[:,23] do
nothing. We feed the network an estimate of the W2 pocket instead, by adding the dead columns (scaled) onto the
columns of the inputs the estimate is built from (the dead columns themselves are left untouched):

    q_W2hat = q_E2 + c_t*(t_green - t0) - k3*phase3 + k_sp*q_W1u
    d_W2hat = d_E2 + c_t*(t_green - t0) - k3*phase3 + k_sp*q_W1u

  term 1  opposite left pocket E2, served by the same phase (EW_left), same demand:
          W1[:,5] += a11 ; W1[:,17] += a23                                 (exact, 128 params)
  term 2  timer: W left-turners keep arriving while another phase is green; zero at the usual decision point
          (t_green = 0.2 = 12 s); un-strands invisible W2 cars when the junction is otherwise idle:
          W1[:,45] += c_t*a ; b1 -= c_t*t0*a                                 (a = a11 + a23)
  term 3  in EW_left itself the W2 pocket is being served, so the timer is partly cancelled:
          W1[:,43] -= k3*a
  term 4  halted cars on W1u (the upstream lane feeding W1 and W2) mostly mean the W2 pocket has spilled back:
          W1[:,31] += k_sp*a
Terms 2-4 are written only on the `topk` units with the largest |a| (the rest have |a| <= ~0.02) to stay within the
371-parameter budget.
"""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools


def build(c_t=0.0, t0=0.2, k3=0.0, k_sp=0.0, topk=64):
    w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
    w = {k: v.copy() for k, v in w0.items()}
    a11, a23 = w0["W1"][:, 11].copy(), w0["W1"][:, 23].copy()
    a = a11 + a23
    mask = np.zeros(64, bool); mask[np.argsort(-np.abs(a))[:topk]] = True
    am = np.where(mask, a, 0.0)
    w["W1"][:, 5] += a11
    w["W1"][:, 17] += a23
    if c_t:
        w["W1"][:, 45] += c_t * am
        if t0:
            w["b1"] -= c_t * t0 * am
    if k3:
        w["W1"][:, 43] -= k3 * am
    if k_sp:
        w["W1"][:, 31] += k_sp * am
    return w


def idle_table(w):
    out = {}
    for p in range(4):
        row = []
        for tg in [0.2, 0.4, 0.6, 0.8, 1.0]:
            x = np.zeros(46); x[40 + p] = 1; x[44] = 1; x[45] = tg
            row.append(int(tools.greedy(w, x)[0]))
        out[p] = row
    return out


if __name__ == "__main__":
    import json
    name = sys.argv[1]
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    w = build(**kw)
    es = tools.edit_size(w, tools.load(os.path.join(HERE, "start_weights.npz")))
    print(name, kw, es)
    assert es["changed"] <= 371
    print("idle decisions (phase -> choice at t_green .2 .4 .6 .8 1):", idle_table(w))
    tools.save(w, os.path.join(HERE, "work", name + ".npz"))
