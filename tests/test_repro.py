# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Fast reproducibility tests (about 2 minutes). The full audit is audit/run_audit.py.

    python -m pytest tests/ -q          # or: python tests/test_repro.py

1. Networks: L2 and L2W rebuild identically (net.net.xml compared after netconvert's dated header comment).
2. Routes: the route files of the 30 test episodes of L2 and L2W regenerate with the hashes of the files used in the study.
3. Episodes: a sample of stored test episodes (reference controllers, a PPO network, an agent patch, a rule file, a
   supervisor under the rules, a network with the failed detector) re-run with identical TTS.
4. Analysis: paper_numbers.json regenerates identically from the stored evaluations.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _body(path):
    return re.sub(r"^<\?xml[^>]*>\s*<!--.*?-->", "", open(path, encoding="utf-8").read(), count=1, flags=re.S)


def test_nets():
    from wsp import junctions as J
    tmp = tempfile.mkdtemp(prefix="wsp_nets_")
    old = J.NET_DIR
    try:
        J.NET_DIR = tmp
        for j in ("L2", "L2W"):
            J.build(j)
            for f in os.listdir(os.path.join(old, j)):
                a, b = os.path.join(old, j, f), os.path.join(tmp, j, f)
                if f == "net.net.xml":
                    assert _body(a) == _body(b), f"{j}/{f}"
                else:
                    assert open(a, "rb").read() == open(b, "rb").read(), f"{j}/{f}"
    finally:
        J.NET_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_routes():
    from wsp.env import SignalEnv
    from wsp.evaluate import SETS
    ref = json.load(open(os.path.join(ROOT, "docs", "route_hashes.json")))["routes"]
    tmp = tempfile.mkdtemp(prefix="wsp_routes_")
    try:
        for j in ("L2", "L2W"):
            for sc, s in SETS["test"]:
                wd = os.path.join(tmp, f"{j}_{sc}_{s}")
                env = SignalEnv(j, scenario=sc, seed=s, drain=True, workdir=wd)
                env.reset(seed=s)
                h = hashlib.sha256(open(os.path.join(wd, "routes.rou.xml"), "rb").read()).hexdigest()
                env.close()
                assert h == ref[f"{j}/test/{sc}_{s}"], (j, sc, s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


SAMPLE = [("L2", "fixed", 0), ("L2", "actuated", 25), ("L2", "ppo_s100", 21), ("L2", "patch_net0_b10", 12),
          ("L2", "explain_net1_small", 3), ("L2_rules", "constrain_net0_b10", 27), ("L2_fault", "patchf_net2_b0", 22),
          ("L2W", "transfer_net0", 14)]


def test_episodes():
    from wsp.evaluate import SETS, run_many, result_path
    tmp = tempfile.mkdtemp(prefix="wsp_eps_")
    try:
        jobs, ref = [], []
        for prob, lab, i in SAMPLE:
            d = json.load(open(result_path(prob, lab, "test")))
            sc, s = SETS["test"][i]
            jobs.append(dict(problem=prob, policy=d["policy"], scenario=sc, seed=s,
                             workdir=os.path.join(tmp, f"{prob}_{lab}_{i}")))
            ref.append(d["episodes"][i])
        for (prob, lab, i), r, e in zip(SAMPLE, run_many(jobs, min(8, os.cpu_count() or 1)), ref):
            for k in ("tts_veh_h", "delay", "remaining", "arrived"):
                assert abs(r[k] - e[k]) < 1e-9, (prob, lab, i, k, r[k], e[k])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_analysis():
    tmp = tempfile.mkdtemp(prefix="wsp_an_")
    try:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("models", "packs", "runs", ".git",
                                                                   "__pycache__", "paper"))
        os.symlink(os.path.join(ROOT, "packs"), os.path.join(dst, "packs"))     # read only by the analysis
        shutil.copytree(os.path.join(ROOT, "runs"), os.path.join(dst, "runs"))   # the analysis re-records held-out states here
        os.makedirs(os.path.join(dst, "models"))
        for prob in ("L2", "L2W", "L2_rules", "L2_fault"):   # curves, monitors and the target networks
            src = os.path.join(ROOT, "models", prob)
            for run in os.listdir(src):
                for f in ("curve_val.json", "monitor.csv", "final.zip"):
                    p = os.path.join(src, run, f)
                    if os.path.exists(p):
                        os.makedirs(os.path.join(dst, "models", prob, run), exist_ok=True)
                        shutil.copy(p, os.path.join(dst, "models", prob, run, f))
            if os.path.isdir(os.path.join(src, "baselines")):
                shutil.copytree(os.path.join(src, "baselines"), os.path.join(dst, "models", prob, "baselines"),
                                dirs_exist_ok=True)
        os.remove(os.path.join(dst, "results", "paper_numbers.json"))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        p = subprocess.run([sys.executable, "-m", "wsp.analysis"], cwd=dst, capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stderr[-800:]
        a = json.load(open(os.path.join(ROOT, "results", "paper_numbers.json")))
        b = json.load(open(os.path.join(dst, "results", "paper_numbers.json")))
        assert a == b
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for t in (test_nets, test_routes, test_episodes, test_analysis):
        t()
        print("ok", t.__name__, flush=True)
