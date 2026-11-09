"""v2: symmetric-twin substitution + a 'time since EW_left' term for the failed W2 detector.

Stand-in fed into the network's W2 channels (both q_W2 and d_W2):
    W2_hat = E2 (twin, same demand, same serving phase)  +  K * (t_green - A*ok - B*phase3)

The time term exists because the twin alone never reveals W left-turners when E2 is empty: in the drain period
(and in quiet moments) the network then holds phase 1 or 2 forever and the hidden W2 cars stay until the drain cap.
While a phase other than 3 is green, W2 can only fill, so a growing t_green is real evidence of W2 demand. A*ok
makes the term about 0 at the first honoured decision (t_green = 0.2, ok = 1); decisions with ok = 0 are never
honoured, so the term's value there does not matter. B*phase3 with A+B = 1 makes the term <= 0 throughout phase 3,
so it never stretches EW_left (W2 is draining then).

Implemented by adding to first-layer columns (u = W1[:, q_W2] + W1[:, d_W2], the original W2 weights):
    W1[:, q_E2] += W1[:, q_W2];  W1[:, d_E2] += W1[:, d_W2]
    W1[:, t_green] += K*u;  W1[:, ok] -= K*A*u;  W1[:, phase3] -= K*B*u
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools  # noqa: E402

QW2, DW2, QE2, DE2, PH3, OK, TG = 11, 23, 5, 17, 43, 44, 45


def build(K=0.75, A=0.3, B=0.7):
    w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
    w = {k: v.copy() for k, v in w0.items()}
    u = w0["W1"][:, QW2] + w0["W1"][:, DW2]
    w["W1"][:, QE2] += w0["W1"][:, QW2]
    w["W1"][:, DE2] += w0["W1"][:, DW2]
    w["W1"][:, TG] += K * u
    w["W1"][:, OK] -= K * A * u
    w["W1"][:, PH3] -= K * B * u
    return w0, w


if __name__ == "__main__":
    w0, w = build()
    print(tools.edit_size(w, w0))
    tools.save(w, os.path.join(HERE, "work", "v2_twin_time.npz"))
