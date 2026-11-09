"""Build work/weights.npz from start_weights.npz by construction (no fitting).

Patch 1 - substitute the dead W2 detector by its mirror lane E2 (128 parameters):
    The deployed input has x[11] = q_W2 = 0 and x[23] = d_W2 = 0. We make layer 1 behave as if
    q_W2 := q_E2 and d_W2 := d_E2, by adding the W2 weight columns onto the E2 weight columns:
        W1[:, 5]  += W1[:, 11]      (q_E2 now also carries q_W2's weights)
        W1[:, 17] += W1[:, 23]      (d_E2 now also carries d_W2's weights)
    Both left pockets are emptied by the same phase (3, EW_left) and refilled by the same demand (E and W share the
    directional factor e^b), so E2 is the natural stand-in for W2.

Patch 2 - "empty junction -> EW_left" watchdog (repurposes layer-1 unit K=31, the unit whose removal changes the
fewest decisions):
    A lone left-turner in W2 is now invisible. If every visible lane is empty while it waits, the network parks in
    the current phase and the car is stranded until the episode cap (the actuated controller shows this in recorded
    episode 5: 1650 s of 'empty' junction). Unit K becomes a detector:
        pre_K = 5 - 35 * (sum of stop-line densities) - 350 * (sum of upstream densities) + (old small weights)
    One car anywhere visible gives <= -10 on the density term, so h_K = -1 whenever anything is visible and +1 only
    when every visible lane is empty.
    Its old mean contribution is moved into b2 (b2 += W2[:,K] * (mean_old + 1) for the 'off' value -1).
    A few layer-2 units J get a large W2[J, K] whose sign drives them, when K fires, toward the side that raises the
    EW_left logit over the others. b2[J] compensates so that with h_K = -1 nothing changes.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

K = 31                       # repurposed layer-1 unit
A_STOP, A_UP, C_BIAS = 35.0, 350.0, 5.0
STOP_D = list(range(12, 23))          # d_N0 .. d_W1 (d_W2 = 23 is dead)
UP_D = list(range(32, 40))            # d_N0u .. d_W1u
LAM = 6.0                             # |W2[j, K]| for the push units: shift 2*LAM = 12 when K fires
MARGIN = 3.0                          # required EW_left logit lead in every honoured empty state
G = 0                                 # repurposed layer-1 unit for the W-side blockage detector
G_A, G_B, G_GATE = 300.0, 1000.0, 320.0
SHIFT_E2 = 0.3348                     # 'W2 full' shift measured relative to the substitute value E2 = 1 car


def empty_states():
    E = []
    for p in range(4):
        for kk in range(14):
            tg = min(1.0, (2 + 5 * kk) / 60)
            x = np.zeros(46); x[40 + p] = 1; x[45] = tg; x[44] = 1.0 if tg >= 10 / 60 - 1e-9 else 0.0
            E.append(x)
    return np.array(E)


def build():
    w0 = tools.load("start_weights.npz")
    w = {k: v.copy() for k, v in w0.items()}

    # ---- patch 1: W2 := E2 substitution
    w["W1"][:, 5] += w0["W1"][:, 11]
    w["W1"][:, 17] += w0["W1"][:, 23]

    # ---- patch 2: watchdog
    S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
    normal = ~((c == 0) & (ep >= 2))           # leave out the gridlocked start-network episodes
    h1, _ = tools.hidden(w, X)
    m_old = float(h1[normal, K].mean())

    W2_old_col = w["W2"][:, K].copy()
    # detector row
    w["W1"][K, STOP_D] = -A_STOP
    w["W1"][K, UP_D] = -A_UP
    w["b1"][K] = C_BIAS

    # choose push units J: drive layer-2 unit j to s_j in {-1,+1} when K fires. Its gain for phase 3 over phase p
    # in an empty state is (W3[3,j]-W3[p,j]) * (s_j - h2_j). Score = worst case over the empty states and p != 3.
    # Add units in score order until phase 3 wins every honoured empty state by MARGIN.
    W3 = w["W3"]
    diff = W3[3][None, :] - W3[[0, 1, 2], :]          # [3, 64]
    s = np.sign(diff.mean(0))
    E = empty_states(); okE = E[:, 44] > 0.5
    w_try = {k: v.copy() for k, v in w.items()}
    w_try["b2"] = w["b2"] + W2_old_col * m_old + W2_old_col     # detector on, no push yet
    _, h2E = tools.hidden(w_try, E[okE])
    score = np.array([(diff[:, j][None, :] * (s[j] - h2E[:, j])[:, None]).min() for j in range(64)])
    order = [int(j) for j in np.argsort(-score) if score[j] > 0]
    J = []
    for j in order:
        J.append(j)
        col = W2_old_col.copy(); col[J] = LAM * s[J]
        w_try["W2"][:, K] = col
        w_try["b2"] = w["b2"] + W2_old_col * m_old + col
        LE = tools.forward(w_try, E[okE])
        if (LE[:, 3] - np.delete(LE, 3, axis=1).max(1)).min() >= MARGIN:
            break
    W2_new_col = W2_old_col.copy()
    W2_new_col[J] = LAM * s[J]
    w["W2"][:, K] = W2_new_col
    # bias compensation: with h_K = -1 the layer-2 pre-activations equal their old values with unit K at its mean
    w["b2"] = w["b2"] + W2_old_col * m_old + W2_new_col
    info = {"K": K, "m_old": m_old, "J": J, "signs": [int(v) for v in s[J]], "score": [round(float(score[j]), 3) for j in J]}

    # ---- patch 3: W-side blockage unit G (repurposes layer-1 unit G=0, the next least important)
    # Signature: EW_through green for >= 10 s, at least one halting car in W1u, and the straight lane W1 empty.
    # Then the head of W1u is a left-turner that cannot enter a full W2 pocket (mirror E side: pocket full in
    # 98-100 % of such states). When G fires it adds to layer 2 the mean shift that 'W2 full' would have caused
    # (relative to the E2 substitute at E2 = 1 car), measured on the rotated E-side signature states.
    from mirror_test import rot
    h1b, _ = tools.hidden(w, X)                      # unit G is untouched so far
    mG = float(h1b[normal, G].mean())
    ph = X[:, 40:44].argmax(1)
    sig = (c > 0) & (ph == 2) & (X[:, 44] > 0.5) & (X[:, 27] >= 0.029) & (X[:, 16] == 0)
    R = rot(X[sig])
    T = R.copy(); T[:, 5] = SHIFT_E2; T[:, 17] = SHIFT_E2
    Fu = T.copy(); Fu[:, 11] = 1.0; Fu[:, 23] = 1.0
    Im = T.copy(); Im[:, 11] = SHIFT_E2; Im[:, 23] = SHIFT_E2

    def a2(ww, Z):
        return np.tanh(Z @ ww["W1"].T + ww["b1"]) @ ww["W2"].T + ww["b2"]
    delta = (a2(w0, Fu) - a2(w0, Im)).mean(0)
    W2_old_G = w["W2"][:, G].copy()
    w["W1"][G, :] = 0.0
    w["W1"][G, 31] = G_A                             # q_W1u
    w["W1"][G, 22] = -G_B                            # d_W1
    w["W1"][G, 42] = G_GATE                          # phase2
    w["W1"][G, 44] = G_GATE                          # ok
    w["b1"][G] = -2 * G_GATE - G_A * 0.0146          # threshold half-way between 0 and 1 halting car in W1u
    w["W2"][:, G] = delta / 2.0                      # flip -1 -> +1 adds delta
    w["b2"] = w["b2"] + W2_old_G * mG + delta / 2.0  # with h_G = -1 layer 2 sees G's old mean contribution
    info.update({"G": G, "mG": mG, "n_sig": int(sig.sum()), "delta_norm": float(np.linalg.norm(delta))})
    return w0, w, info


if __name__ == "__main__":
    w0, w, info = build()
    print(json.dumps(info))
    print(tools.edit_size(w, w0))
    tools.save(w, os.path.join(HERE, "weights.npz"))
    print("saved", os.path.join(HERE, "weights.npz"))
