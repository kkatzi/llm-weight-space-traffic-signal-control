"""Reference rule (plain numpy) that the hand-built network implements.
Scores are in 'vehicle' units built from the observation."""
import numpy as np

CS, CU = 22.4 / 7.5, 256 / 7.5          # lane capacities (vehicles) used to normalise the observation
NAMES = None

# weights (vehicle-equivalents)
W_HS = 1.0      # halting vehicle on a stop-line lane of the phase
W_MS = 0.3      # moving vehicle on a stop-line lane
W_HU = 0.8      # halting vehicle on an upstream lane feeding the phase
W_MU = 0.3      # moving vehicle on an upstream lane (arrives within the next green)
SPLIT_H = {"thr": 0.35, "left": 0.65} # share of upstream lane 1 (straight+left) halting: mostly left-turners spilling back from a full pocket
SPLIT_M = {"thr": 0.4, "left": 0.6}   # share of upstream lane 1 moving vehicles
BETA = 1.0      # hold bonus for the current phase (vehicle-equivalents)

STOP = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}         # indices into the 12 stop-line lanes
UP_FULL = {0: [0, 4], 1: [], 2: [2, 6], 3: []}                            # indices into the 8 upstream lanes
UP_SHARED = {0: [1, 5], 1: [1, 5], 2: [3, 7], 3: [3, 7]}
KIND = {0: "thr", 1: "left", 2: "thr", 3: "left"}


def score_matrix():
    """Return M (4, 46) and c (4,) with D = X @ M.T + c (vehicle-equivalent demand per phase)."""
    M = np.zeros((4, 46))
    for p in range(4):
        for l in STOP[p]:
            # hs + W_MS*ms = 3*(q) + W_MS*3*(d - q)
            M[p, l] += CS * (W_HS - W_MS)
            M[p, 12 + l] += CS * W_MS
        lanes = [(u, 1.0, 1.0) for u in UP_FULL[p]] + [(u, SPLIT_H[KIND[p]], SPLIT_M[KIND[p]]) for u in UP_SHARED[p]]
        for u, sh, sm in lanes:
            M[p, 24 + u] += CU * (W_HU * sh - W_MU * sm)
            M[p, 32 + u] += CU * W_MU * sm
    return M, np.zeros(4)


def scores(X):
    X = np.atleast_2d(X)
    M, c = score_matrix()
    D = X @ M.T + c
    return D + BETA * X[:, 40:44]


def decide(x):
    return int(np.argmax(scores(x)[0]))
