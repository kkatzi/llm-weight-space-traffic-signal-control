# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Repository paths, and resolution of the absolute paths recorded during the study.

The stored results and agent logs record absolute paths of the machine the study ran on (the pipeline under
.../weight_space_traffic/v3, the agent task folders under ~/wsp_v3_agent_runs). The records are kept as written;
resolve() maps such a path to the same file in this repository, and same_spec() compares policy specs after that
mapping, so cached evaluations are reused wherever the repository lives.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = (("/home/traffic-ntua/Documents/weight_space_traffic/v3", ROOT),
          ("/home/traffic-ntua/wsp_v3_agent_runs", os.path.join(ROOT, "packs")))


def resolve(p):
    """A recorded path, or a path relative to the repository root -> the file in this repository (anything else
    is returned unchanged)."""
    for old, new in LEGACY:
        if p == old or p.startswith(old + "/"):
            return new + p[len(old):]
    if p and not os.path.isabs(p) and os.path.exists(os.path.join(ROOT, p)):     # relative to the repository root
        return os.path.join(ROOT, p)
    return p


def norm_spec(spec):
    """A policy spec ("npz:PATH", "super:FILE|NET", ...) with every recorded path resolved."""
    head, sep, rest = spec.partition(":")
    if not sep:
        return spec
    return head + ":" + "|".join(resolve(x) for x in rest.split("|"))


def same_spec(a, b):
    return norm_spec(a) == norm_spec(b)
