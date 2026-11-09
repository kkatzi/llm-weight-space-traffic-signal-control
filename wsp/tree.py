# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Minimal CART classifier in numpy (Gini, axis-aligned splits, depth-limited),
used to distil a network into a decision tree (the 'Explain' baseline), plus an
exporter that writes the tree as a readable Python rules file with the same
interface as the agents' rules: decide(x) -> phase.

Candidate thresholds are midpoints between consecutive distinct values, thinned
to at most 64 quantiles per feature.
"""
import numpy as np


def _gini(counts):
    n = counts.sum()
    return 0.0 if n == 0 else 1.0 - float(((counts / n) ** 2).sum())


def fit(X, y, max_depth=3, min_leaf=10, n_classes=4, weights=None):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)

    def counts(idx):
        return np.bincount(y[idx], weights=w[idx], minlength=n_classes)

    def build(idx, depth):
        c = counts(idx)
        node = {"leaf": int(c.argmax()), "n": int(len(idx))}
        if depth >= max_depth or len(idx) < 2 * min_leaf or (c > 0).sum() <= 1:
            return node
        best = (_gini(c) * c.sum(), None, None)
        for f in range(X.shape[1]):
            v = X[idx, f]
            u = np.unique(v)
            if len(u) < 2:
                continue
            cand = (u[:-1] + u[1:]) / 2
            if len(cand) > 64:
                cand = np.unique(np.quantile(v, np.linspace(0.01, 0.99, 64)))
            for t in cand:
                left = idx[v <= t]
                right = idx[v > t]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                cl, cr = counts(left), counts(right)
                score = _gini(cl) * cl.sum() + _gini(cr) * cr.sum()
                if score < best[0] - 1e-12:
                    best = (score, f, float(t))
        if best[1] is None:
            return node
        f, t = best[1], best[2]
        v = X[idx, f]
        node.update(feature=f, threshold=t, left=build(idx[v <= t], depth + 1),
                    right=build(idx[v > t], depth + 1))
        return node

    return build(np.arange(len(y)), 0)


def predict(tree, X):
    X = np.atleast_2d(np.asarray(X, float))
    out = np.empty(len(X), int)
    for i, x in enumerate(X):
        n = tree
        while "feature" in n:
            n = n["left"] if x[n["feature"]] <= n["threshold"] else n["right"]
        out[i] = n["leaf"]
    return out


def n_splits(tree):
    return 0 if "feature" not in tree else 1 + n_splits(tree["left"]) + n_splits(tree["right"])


def to_python(tree, names, header=""):
    """Python source with decide(x) -> int, one comparison per split."""
    lines = [f'"""{header}"""', "", "", "def decide(x):"]

    def emit(n, ind):
        pad = "    " * ind
        if "feature" not in n:
            lines.append(f"{pad}return {n['leaf']}")
            return
        lines.append(f"{pad}if x[{n['feature']}] <= {n['threshold']:.4f}:  # {names[n['feature']]}")
        emit(n["left"], ind + 1)
        lines.append(f"{pad}else:")
        emit(n["right"], ind + 1)

    emit(tree, 1)
    return "\n".join(lines) + "\n"
