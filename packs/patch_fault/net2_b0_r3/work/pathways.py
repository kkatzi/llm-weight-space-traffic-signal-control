"""Step 2: two hand-built rule pathways (max-green style), each = one layer-1 detector unit + one layer-2 relay unit.

A layer-1 unit u gets three extra huge input weights (K-trick): pre_u = old_pre_u + s*K*(condition margin), so it is
saturated at its 'normal' level everywhere except when the condition holds; its old inputs become irrelevant.
A layer-2 unit j gets one huge weight from u, so it copies u (saturated -1 normally, +1 under the condition);
its W3 column is replaced by the push vector P and b3 absorbs the constant normal-state contribution."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np

T_IDX, OK_IDX = 45, 44


def add_pathway(w, w_ref_stats_X, u, u_normal, j, terms, bias_margin, P, K2=12.0):
    """terms: dict input_index -> weight added to W1[u, idx] (condition margin is sum(terms*x) + bias_margin,
    positive when the condition holds). u_normal: +1 or -1, the saturated value of u outside the condition.
    j copies u; P is the logit push (4,) applied as P * h2[j] with h2[j] = -1 normally, +1 under the condition."""
    w = {k: v.copy() for k, v in w.items()}
    X = w_ref_stats_X
    # mean of the relay unit j BEFORE repurposing (to keep the normal-state logits = old logits with j at its mean)
    h1 = np.tanh(X @ w["W1"].T + w["b1"]); h2 = np.tanh(h1 @ w["W2"].T + w["b2"])
    m_j = h2[:, j].mean()
    s = -u_normal   # condition drives u to -u_normal
    for idx, val in terms.items():
        w["W1"][u, idx] += s * val
    w["b1"][u] += s * bias_margin
    # j copies (condition) as +1: pre_j = old + K2 * (h1[u] * s)  -> normal: -K2, condition: +K2
    w["W2"][j, u] += K2 * s
    old_col = w["W3"][:, j].copy()
    w["W3"][:, j] = P
    w["b3"] += old_col * m_j + P   # normal: P*(-1) + P = 0 ; plus the old mean contribution
    return w
