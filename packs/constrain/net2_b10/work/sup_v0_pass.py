"""Variant 0: pure pass-through (the network's own choice; the shield applies the rules). Baseline check."""
import numpy as np


def decide(x, logits, state):
    return int(np.argmax(np.asarray(logits, dtype=float)))
