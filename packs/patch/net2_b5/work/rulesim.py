"""Rule-level prototype (no weights): original logits minus large penalties where a rule fires.
Used only to choose which rules to build into the network."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
import loopcheck2
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
w0 = tools.load('start_weights.npz')


def fires(x, rules):
    ok = x[44] > 0.5
    ph = int(np.argmax(x[40:44]))
    pen = np.zeros(4)
    D = [sum(x[12 + l] for l in SERVE[p]) for p in range(4)]
    up = x[32:40].sum()
    others3 = sum(x[l] for l in range(12) if l not in SERVE[3])
    if 'R1' in rules and ok and ph == 3 and D[3] == 0 and others3 > 0:
        pen[3] = 1
    for p in range(4):
        if ('E%d' % p) in rules and ok and D[p] == 0 and up == 0:
            pen[p] = 1
    return pen


class RuleNet:
    def __init__(self, rules): self.rules = rules
def make_greedy(rules):
    def greedy(w, X):
        X = np.atleast_2d(X)
        lg = tools.forward(w0, X)
        pen = np.array([fires(x, rules) for x in X])
        return (lg - 100 * pen).argmax(1)
    return greedy


if __name__ == '__main__':
    orig = tools.greedy
    for rules in [('R1',), ('E0', 'E1', 'E2', 'E3'), ('R1', 'E0', 'E1', 'E2', 'E3'), ('R1', 'E0', 'E2'), ('R1', 'E0', 'E2', 'E3'), ('R1', 'E0', 'E1', 'E2')]:
        tools.greedy = make_greedy(rules)
        loopcheck2.tools.greedy = tools.greedy
        loopcheck2.sweep(w0, str(rules), verbose=True)
    tools.greedy = orig
