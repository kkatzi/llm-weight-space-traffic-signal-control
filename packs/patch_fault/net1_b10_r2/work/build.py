"""build.py: construct patched weights by hand (no fitting).

FINAL deliverable = variant C3:  python work/build.py C3 work/weights.npz   (329 parameters changed)
  mirror1 (64)  +  timer unit 31 (3 + 64)  +  phase-3 max-green unit 0 (3 + 64)  +  spillback unit 4 (3 + 64)
  +  b2 (64, absorbs the replaced units' old mean contributions).  See work/NOTES.md.

Variant A ("mirror"):the dead inputs q_W2 (x11) and d_W2 (x23) are replaced by a proxy taken from the opposite
left-turn pocket E2 (same protected phase 3, same nominal demand):  q_W2_hat = a * q_E2,  d_W2_hat = a * d_E2.
Because the forward pass is linear in x before the tanh, this is exactly
    W1[:, q_E2] += a * W1[:, q_W2]
    W1[:, d_E2] += a * W1[:, d_W2]
(128 parameters). The dead columns themselves are left alone (their inputs are always 0).

Run:  python work/build.py <variant> <out.npz>
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

N = tools.names()
I = {n: i for i, n in enumerate(N)}


def mirror(w, a=1.0):
    w = {k: v.copy() for k, v in w.items()}
    w["W1"][:, I["q_E2"]] += a * w["W1"][:, I["q_W2"]]
    w["W1"][:, I["d_E2"]] += a * w["W1"][:, I["d_W2"]]
    return w


def w2_full_effect(w, Xs):
    """Mean change of the layer-2 pre-activations when the dead W2 detector would read 'full' (q_W2 = d_W2 = 1),
    over the states Xs (a forward-pass statistic, no fitting)."""
    a1 = Xs @ w["W1"].T + w["b1"]
    d = np.tanh(a1 + w["W1"][:, I["q_W2"]] + w["W1"][:, I["d_W2"]]) - np.tanh(a1)
    return (d @ w["W2"].T).mean(axis=0)


def reference_states(w):
    """States the injection vector is averaged over: (a) recorded fixed-time / actuated states with the minimum
    green elapsed and a phase other than 3, (b) the three idle states 'network empty, phase p, long green'."""
    S = tools.states()
    X = S["obs"].astype(np.float64)
    ph = X[:, 40:44].argmax(1)
    real = X[(S["controller"] > 0) & (X[:, I["ok"]] > 0.5) & (ph != 3)]
    idle = []
    for p in range(3):
        z = np.zeros(46)
        z[40 + p] = 1
        z[I["ok"]] = 1
        z[I["t_green"]] = 1
        idle.append(z)
    return real, np.array(idle)


def repurpose_unit(w, u, row, bias, inject):
    """Turn layer-1 unit u into a detector: pre-activation = row @ x + bias (designed to sit at -1 when off, +1
    when on). Its old contribution to layer 2 is replaced by its mean (folded into b2); when on it adds `inject` to
    the layer-2 pre-activations, when off nothing."""
    S = tools.states()
    X = S["obs"].astype(np.float64)
    h1, _ = tools.hidden(w, X)
    w = {k: v.copy() for k, v in w.items()}
    old_mean = h1[:, u].mean()
    w["b2"] = w["b2"] + w["W2"][:, u] * old_mean + inject / 2.0
    w["W2"][:, u] = inject / 2.0
    w["W1"][u, :] = row
    w["b1"][u] = bias
    return w


def timer_unit(w, u=31, thr_s=49.5, k=60.0):
    """Unit u fires once the current green (not phase 3) is older than thr_s seconds: then it acts on layer 2 as if
    the W2 pocket were reported full, so the network visits phase 3 and serves left-turners it cannot see."""
    real, idle = reference_states(w)
    inject = 0.5 * w2_full_effect(w, real) + 0.5 * w2_full_effect(w, idle)
    row = np.zeros(46)
    row[I["t_green"]] = k
    row[I["phase3"]] = -k
    bias = -k * thr_s / 60.0
    return repurpose_unit(w, u, row, bias, inject)


def mirror1(w, a=1.0):
    """One-column mirror (64 parameters): q_W2_hat = d_W2_hat = a * d_E2."""
    w = {k: v.copy() for k, v in w.items()}
    w["W1"][:, I["d_E2"]] += a * (w["W1"][:, I["q_W2"]] + w["W1"][:, I["d_W2"]])
    return w


def leave3_direction(w):
    """Component of logit-3's readout row W3[3] orthogonal to the other three readout rows: a layer-2 direction
    that, to first order, lowers logit 3 and leaves logits 0-2 alone (pure linear algebra on W3)."""
    W3 = w["W3"]
    A = W3[:3].T                                   # 64 x 3
    r = W3[3] - A @ np.linalg.solve(A.T @ A, A.T @ W3[3])
    return r / np.linalg.norm(r)


def two_timers(w, u_t=31, u_3=0, thr_s=29.5, thr3_s=29.5, k=60.0, drop3=6.0):
    """Unit u_t: green of phase 0/1/2 older than thr_s -> act as if W2 reported 'full'.
    Unit u_3: green of phase 3 older than thr3_s -> lower logit 3 by about drop3 (max-green for phase 3)."""
    S = tools.states()
    X = S["obs"].astype(np.float64)
    h1_old, _ = tools.hidden(w, X)
    real, idle = reference_states(w)
    inj_t = 0.5 * w2_full_effect(w, real) + 0.5 * w2_full_effect(w, idle)
    r = leave3_direction(w)
    # scale for unit u_3: measure the mean drop of logit 3 per unit step along r on phase-3 reference states
    ph = X[:, 40:44].argmax(1)
    P3 = X[(ph == 3) & (X[:, I["ok"]] > 0.5)].copy()
    P3[:, I["t_green"]] = 1.0
    a2 = np.tanh(P3 @ w["W1"].T + w["b1"]) @ w["W2"].T + w["b2"]
    base = np.tanh(a2) @ w["W3"].T
    step = np.tanh(a2 - r) @ w["W3"].T
    per_unit = float((base[:, 3] - step[:, 3]).mean())
    inj_3 = -r * (drop3 / per_unit)
    w = {k_: v.copy() for k_, v in w.items()}
    for u, inj in ((u_t, inj_t), (u_3, inj_3)):
        w["b2"] = w["b2"] + w["W2"][:, u] * h1_old[:, u].mean() + inj / 2.0
        w["W2"][:, u] = inj / 2.0
    row = np.zeros(46)
    row[I["t_green"]] = k
    row[I["phase3"]] = -k
    w["W1"][u_t, :] = row
    w["b1"][u_t] = -k * thr_s / 60.0
    row = np.zeros(46)
    row[I["t_green"]] = k
    row[I["phase3"]] = k
    w["W1"][u_3, :] = row
    w["b1"][u_3] = -k * (thr3_s / 60.0 + 1.0)
    return w


def detector_units(w, specs, inject_list, X):
    """Cheap repurposing of layer-1 units (variant C). Each spec is (u, {input_name: weight}, threshold_offset).
    Only the listed input weights and the bias of unit u are overwritten; the unit's other old input weights stay and
    act as a small bounded 'residual' (|residual| < 3 on the recorded states), which the large gains swamp. The
    bias is centred on the residual's mid-range. The unit's old outgoing column is replaced by inject/2 and its old
    mean contribution plus inject/2 is folded into b2, so the unit adds nothing when off (h = -1) and `inject` to the
    layer-2 pre-activations when on (h = +1)."""
    h1_old, _ = tools.hidden(w, X)
    w = {k: v.copy() for k, v in w.items()}
    for (u, weights, offset), inj in zip(specs, inject_list):
        row = w["W1"][u].copy()
        for n in weights:
            row[I[n]] = 0.0
        resid = X @ row + w["b1"][u]
        mid = 0.5 * (resid.min() + resid.max())
        w["b2"] = w["b2"] + w["W2"][:, u] * h1_old[:, u].mean() + inj / 2.0
        w["W2"][:, u] = inj / 2.0
        for n, val in weights.items():
            w["W1"][u, I[n]] = val
        w["b1"][u] = w["b1"][u] - mid + offset
    return w


def leave3_inject(w, X, drop3=6.0):
    r = leave3_direction(w)
    ph = X[:, 40:44].argmax(1)
    P3 = X[(ph == 3) & (X[:, I["ok"]] > 0.5)].copy()
    P3[:, I["t_green"]] = 1.0
    a2 = np.tanh(P3 @ w["W1"].T + w["b1"]) @ w["W2"].T + w["b2"]
    per_unit = float((np.tanh(a2) @ w["W3"][3] - np.tanh(a2 - r) @ w["W3"][3]).mean())
    return -r * (drop3 / per_unit)


def variant_c(w, thr_s=29.5, thr3_s=29.5, p_cars=0.5, use_p=True, u_t=31, u_3=0, u_p=4, gate_e2=False, p_scale=1.0):
    """mirror1 + timer (phase 0/1/2 -> 'W2 full') + phase-3 max green + spillback detector P:
    P fires when phase 2 (EW through) is green and at least one vehicle halts on W1u, the upstream lane that feeds
    W1 and the W2 pocket. On the working approaches this signature means the pocket is full 96-100% of the time."""
    S = tools.states()
    X = S["obs"].astype(np.float64)
    ph = X[:, 40:44].argmax(1)
    real, idle = reference_states(w)
    inj_t = 0.5 * w2_full_effect(w, real) + 0.5 * w2_full_effect(w, idle)
    ph2 = X[(S["controller"] > 0) & (X[:, I["ok"]] > 0.5) & (ph == 2)]
    inj_p = p_scale * (0.5 * w2_full_effect(w, ph2) + 0.5 * w2_full_effect(w, idle[2:3]))
    inj_3 = leave3_inject(w, X)
    kt, gate, kq = 120.0, 200.0, 400.0
    q1 = p_cars / (256.0 / 7.5)                    # halting-vehicle share of W1u for p_cars vehicles
    specs = [(u_t, {"t_green": kt, "phase3": -gate}, -kt * thr_s / 60.0),
             (u_3, {"t_green": kt, "phase3": gate}, -kt * thr3_s / 60.0 - gate)]
    injs = [inj_t, inj_3]
    if use_p:
        if gate_e2:
            # C2 (REJECTED, kept for the record): meant to fire P only while d_E2 < 0.5. A single tanh unit cannot
            # AND three conditions when q_W1u spans 0..1 with a gain of 400: the +1500 offset switched P on in
            # almost every state. Not used.
            specs.append((u_p, {"q_W1u": kq, "phase2": 2.5 * gate, "d_E2": -3000.0},
                          -kq * q1 - 2.5 * gate + 1500.0))
        else:
            specs.append((u_p, {"q_W1u": kq, "phase2": 2.5 * gate}, -kq * q1 - 2.5 * gate))
        injs.append(inj_p)
    return detector_units(w, specs, injs, X)


def build(variant):
    w0 = tools.load("start_weights.npz")
    if variant == "C":
        return variant_c(mirror1(w0, 1.0))
    if variant == "C2":
        return variant_c(mirror1(w0, 1.0), gate_e2=True)
    if variant == "C3":
        # P's injection scaled to 0.75: it already sits on top of the mirror (d_E2 is usually >= 0.67 when P
        # fires), so the full 'W2 0 -> 1' push double-counts. Scale chosen among {1, 0.75, 0.5, 0.35} by agreement
        # with the start network fed q_W2 = d_W2 = 1 on the recorded P-on states.
        return variant_c(mirror1(w0, 1.0), p_scale=0.75)
    if variant == "C4":
        # C3 with the phase-0/1/2 timer firing after 19.5 s instead of 29.5 s (faster service of unseen W2 cars).
        return variant_c(mirror1(w0, 1.0), p_scale=0.75, thr_s=19.5)
    if variant == "A":
        return mirror(w0, 1.0)
    if variant == "AT":
        return timer_unit(mirror(w0, 1.0))
    if variant == "B":
        return two_timers(mirror1(w0, 1.0))
    raise SystemExit(f"unknown variant {variant}")


if __name__ == "__main__":
    variant, out = sys.argv[1], sys.argv[2]
    w = build(variant)
    w0 = tools.load("start_weights.npz")
    print("edit size", tools.edit_size(w, w0))
    tools.save(w, out)
    print("saved", out)
