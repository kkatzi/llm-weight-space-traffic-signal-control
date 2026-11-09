# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Reproducibility audit of this repository.

    python audit/run_audit.py                      # everything (about 30-45 min on 40 cores)
    python audit/run_audit.py --stages A,B,D       # a subset; --workers N; --keep to keep the working copy

Every stage runs in a fresh copy of the repository (a temporary folder elsewhere on disk, so the copy also tests that
nothing depends on where the repository lives) and compares what it regenerates with the files shipped here. The
shipped files are never modified, except that the audit writes audit/AUDIT_REPORT.md, audit/audit_results.json and
audit/MANIFEST.sha256.

    A  static     every module compiles and imports from the copy alone; no code path points outside the repository;
                  every policy recorded in a result resolves to a shipped file; every agent's evaluation log equals its
                  out-of-folder copy; every launch prompt equals the prompt archived with the run
    B  inputs     the SUMO networks rebuild byte for byte; the route files of all 222 study episodes regenerate with the
                  hashes of the files used in the study
    C  episodes   every stored test evaluation (every controller, network, agent deliverable, baseline) and the
                  fixed-time cost-set evaluations re-run from scratch, episode by episode, with identical results
    D  analysis   paper_numbers*.json, secondary.json, RESULTS*.md, SECONDARY.md and the paper's tables regenerate
                  identically from the stored evaluations; figures regenerate (text content compared)
    E  training   six PPO runs (from scratch, warm start, transfer warm start, fine-tuning, under the rules, under the
                  fault) are retrained for 100k steps; their checkpoints equal the shipped ones parameter for parameter
    F  baselines  tuning of the reference controllers on L2, one ES run per problem, mean imputation, padding, CART and
                  DAgger (Net 0) regenerate identical outputs
    G  agents     every agent task folder rebuilds identically from the generator; every deliverable re-scores to the
                  same edit size, trainability, comparisons and fidelity
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = {}
LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(cmd, cwd, timeout=None, env_extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"})
    env.update(env_extra or {})
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout[-4000:], p.stderr[-4000:]


def record(stage, name, ok, detail=""):
    RESULTS.setdefault(stage, []).append({"check": name, "ok": bool(ok), "detail": detail})
    log(f"{stage} {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def make_copy(dst):
    log(f"copying the repository to {dst}")
    shutil.copytree(REPO, dst, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".git", "audit_work*"))
    # the copy must not reuse the shipped regenerable outputs: they are deleted where a stage regenerates them


# ------------------------------------------------------------------------------------------------ A: static
def stage_A(C):
    import py_compile
    bad = []
    for f in glob.glob(f"{C}/wsp/*.py") + glob.glob(f"{C}/tests/*.py") + glob.glob(f"{C}/audit/*.py"):
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError as e:
            bad.append(f"{os.path.relpath(f, C)}: {e}")
    record("A", "all Python files compile", not bad, "; ".join(bad))
    mods = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{C}/wsp/*.py") if not f.endswith("__init__.py")
                  and not f.endswith(("pack_tools.py", "pack_eval.py")))
    code = ("import importlib, sys, wsp, os; assert os.path.realpath(wsp.__file__).startswith(os.path.realpath(sys.argv[1])), wsp.__file__\n"
            f"for m in {mods!r}: importlib.import_module('wsp.' + m)\nprint('ok')")
    rc, out, err = run(f'{sys.executable} -c "{code}" {C}', cwd=C)
    record("A", f"all {len(mods)} wsp modules import from the copy alone (no PYTHONPATH)", rc == 0 and "ok" in out, err[-300:])
    hits = []
    for f in glob.glob(f"{C}/wsp/*.py") + glob.glob(f"{C}/scripts/*") + glob.glob(f"{C}/tests/*.py"):
        if f.endswith(("paths.py", "pack_eval.py")):
            continue
        for i, ln in enumerate(open(f, encoding="utf-8"), 1):
            if "/home/traffic-ntua" in ln and not ln.strip().startswith("#"):
                hits.append(f"{os.path.relpath(f, C)}:{i}")
    record("A", "no code refers to a path on the machine the study ran on (except the recorded-path map in wsp/paths.py)", not hits,
           ", ".join(hits[:10]))
    sys.path.insert(0, C)
    from wsp.paths import norm_spec
    missing, n = [], 0
    for f in glob.glob(f"{C}/results/eval/*/*/*.json"):
        spec = json.load(open(f))["policy"]
        head, _, rest = norm_spec(spec).partition(":")
        if head in ("npz", "sb3", "rules", "super", "masked", "hold"):
            for p in rest.split("|"):
                if p.startswith("/"):
                    n += 1
                    if not os.path.exists(p):
                        missing.append(f"{os.path.relpath(f, C)} -> {p}")
    record("A", f"every recorded policy file resolves to a shipped file ({n} references)", not missing, "; ".join(missing[:5]))
    diffs, n = [], 0
    for ev in glob.glob(f"{C}/packs/*/*/eval_log.jsonl"):
        pid = json.load(open(os.path.join(os.path.dirname(ev), "spec.json")))["pack_id"]
        mirror = f"{C}/runs/agent_eval_log/{pid}.jsonl"
        n += 1
        if not os.path.exists(mirror) or open(ev).read() != open(mirror).read():
            diffs.append(pid)
    record("A", f"every agent evaluation log equals its out-of-folder copy ({n} logs)", not diffs, ", ".join(diffs))
    diffs, n = [], 0
    for rec in glob.glob(f"{C}/runs/agent_launch/*__*.json"):
        r = json.load(open(rec))
        case, pid = os.path.basename(rec)[:-5].split("__")
        lp = f"{C}/packs/{case}/{pid}/provenance/launch_prompt.txt"
        n += 1
        if not os.path.exists(lp) or open(lp, encoding="utf-8").read().strip() != r["prompt"].strip():
            diffs.append(f"{case}/{pid}")
    record("A", f"every launch prompt equals the prompt archived with its run ({n} runs)", not diffs, ", ".join(diffs[:10]))
    audits = [json.load(open(a)) for a in glob.glob(f"{C}/packs/*/*/provenance/audit.json")]
    bad = [f"{a['use']}/{a['run']}" for a in audits if not a["prompt_matches_launch"] or a["messages_received_after_launch"]
           or set(a["models"]) != {a["requested_model"]} or a["simulator_calls_logged"] != a["simulator_calls_mirror_log"]
           or a["simulator_calls_logged"] > a["simulator_budget"]]
    record("A", f"archived transcript audits: right model, exact prompt, no messages after launch, simulator calls within "
                f"budget and matching the mirror log ({len(audits)} runs)", not bad, ", ".join(bad))


# ------------------------------------------------------------------------------------------------ B: inputs
def stage_B(C, workers):
    rc, out, err = run(f"{sys.executable} -c \"from wsp import junctions as J; [J.build(j) for j in ('L2', 'L2W')]\"", cwd=C)
    import re
    body = lambda p: re.sub(r"^<\?xml[^>]*>\s*<!--.*?-->", "", open(p, encoding="utf-8").read(), count=1, flags=re.S)  # noqa: E731
    diffs, n = [], 0
    for f in sorted(glob.glob(f"{REPO}/nets/*/*")):
        g = f.replace(REPO, C, 1)
        n += 1
        same = os.path.exists(g) and (body(f) == body(g) if f.endswith("net.net.xml") else sha(f) == sha(g))
        if not same:
            diffs.append(os.path.relpath(f, REPO))
    record("B", f"the SUMO networks of L2 and L2W rebuild identically ({n} files; net.net.xml compared after netconvert's "
                f"header comment, which records the build date and file paths)", rc == 0 and not diffs,
           (err[-300:] if rc else "") + ", ".join(diffs))
    ref = json.load(open(f"{REPO}/docs/route_hashes.json"))["routes"]
    script = r'''
import hashlib, json, os, sys, tempfile
sys.path.insert(0, os.getcwd())
from concurrent.futures import ProcessPoolExecutor
from wsp.evaluate import SETS
def one(key):
    base, st, ep = key.split("/")
    sc, seed = ep.rsplit("_", 1)
    from wsp.env import SignalEnv
    wd = tempfile.mkdtemp(prefix="rt_", dir=sys.argv[1])
    env = SignalEnv(base, scenario=sc, seed=int(seed), drain=True, workdir=wd)
    env.reset(seed=int(seed))
    h = hashlib.sha256(open(os.path.join(wd, "routes.rou.xml"), "rb").read()).hexdigest()
    env.close()
    return key, h
keys = json.load(open(sys.argv[2]))
with ProcessPoolExecutor(int(sys.argv[3])) as ex:
    json.dump(dict(ex.map(one, keys)), open(os.path.join(sys.argv[1], "out.json"), "w"))
'''
    tmp = tempfile.mkdtemp(prefix="routes_", dir=C)
    json.dump(sorted(ref), open(f"{tmp}/keys.json", "w"))
    open(f"{tmp}/gen.py", "w").write(script)
    rc, out, err = run(f"{sys.executable} {tmp}/gen.py {tmp} {tmp}/keys.json {workers}", cwd=C, timeout=3600)
    got = json.load(open(f"{tmp}/out.json")) if rc == 0 else {}
    diffs = [k for k in ref if got.get(k) != ref[k]]
    record("B", f"route files of all {len(ref)} study episodes (test, dev, val, recorded, cost; L2 and L2W) regenerate "
                f"with the hashes of the files used in the study", rc == 0 and not diffs, err[-300:] + ", ".join(diffs[:10]))


# ------------------------------------------------------------------------------------------------ C: episodes
EPISODE_SCRIPT = r'''
import json, os, sys, glob
sys.path.insert(0, os.getcwd())
from concurrent.futures import ProcessPoolExecutor
from wsp.evaluate import SETS, run_episode
C, work, workers, which = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
files = sorted(glob.glob(f"{C}/results/eval/*/*/test.json"))
if which != "all":
    files = [f for f in files if any(w in f for w in which.split(","))]
files += sorted(glob.glob(f"{C}/results/eval/*/*/test_cap3600.json")) if which == "all" else []
files += sorted(glob.glob(f"{C}/results/eval/*/fixed/cost.json"))
jobs = []
for f in files:
    d = json.load(open(f))
    for i, (sc, s) in enumerate(SETS[d["set"]]):
        jobs.append(dict(problem=d["problem"], policy=d["policy"], scenario=sc, seed=s, drain_cap=d.get("drain_cap", 1800),
                         workdir=os.path.join(work, d["problem"], d["label"],
                                              d["set"] + ("" if d.get("drain_cap", 1800) == 1800 else f"_cap{d['drain_cap']}"),
                                              f"{sc}_{s}"), _file=f, _i=i))
def one(j):
    f, i = j.pop("_file"), j.pop("_i")
    try:
        r = run_episode(j)
        return f, i, {k: r[k] for k in ("tts_veh_h", "delay", "remaining", "arrived")}, None
    except Exception as e:
        return f, i, None, repr(e)
# longest first (gridlocked controllers), so the pool stays busy
jobs.sort(key=lambda j: -max(e["tts_veh_h"] for e in json.load(open(j["_file"]))["episodes"]))
out = {}
with ProcessPoolExecutor(workers) as ex:
    for f, i, r, e in ex.map(one, jobs, chunksize=1):
        out.setdefault(f, {})[i] = r if e is None else {"error": e}
json.dump(out, open(os.path.join(work, "episodes.json"), "w"))
print(len(jobs))
'''


def stage_C(C, workers, which="all"):
    work = tempfile.mkdtemp(prefix="episodes_", dir=C)
    open(f"{work}/run.py", "w").write(EPISODE_SCRIPT)
    t0 = time.time()
    rc, out, err = run(f"{sys.executable} {work}/run.py {C} {work} {workers} {which}", cwd=C, timeout=4 * 3600)
    if rc != 0:
        record("C", "re-run of stored evaluations", False, err[-500:])
        return
    got = json.load(open(f"{work}/episodes.json"))
    n_eval = n_ep = n_err = 0
    bad = []
    for f, eps in got.items():
        ref = json.load(open(f.replace(C, REPO, 1)))["episodes"]
        n_eval += 1
        for i, r in eps.items():
            n_ep += 1
            e = ref[int(i)]
            if "error" in r:
                n_err += 1
                bad.append(f"{os.path.relpath(f, C)}#{i}: {r}")
            elif any(abs(r[k] - e[k]) > 1e-9 for k in ("tts_veh_h", "delay", "remaining", "arrived")):
                bad.append(f"{os.path.relpath(f, C)}#{i}: {r} vs {({k: e[k] for k in ('tts_veh_h', 'remaining')})}")
    record("C", f"{n_eval} stored evaluations ({n_ep} episodes) re-run from scratch with identical TTS, delay, unserved and "
                f"served vehicles (tolerance 1e-9; {time.time() - t0:.0f} s)", not bad and n_ep > 0,
           (f"{len(bad)} episodes differ ({n_err} with errors): " if bad else "") + "; ".join(bad[:8]))
    RESULTS.setdefault("_counts", {})["episodes"] = n_ep
    RESULTS["_counts"]["evaluations"] = n_eval


# ------------------------------------------------------------------------------------------------ D: analysis
def stage_D(C):
    regen = ["results/paper_numbers.json", "results/paper_numbers_cap3600.json", "results/secondary.json",
             "results/RESULTS.md", "results/RESULTS_cap3600.md", "results/SECONDARY.md"]
    for f in regen:
        os.remove(f"{C}/{f}")
    shutil.rmtree(f"{C}/paper/tables")
    shutil.rmtree(f"{C}/paper/figs")
    steps = [("analysis", "-m wsp.analysis"), ("analysis, 60-min cap", "-m wsp.analysis --cap 3600"),
             ("secondary", "-m wsp.secondary"), ("figures and tables", "-m wsp.figures")]
    for name, args in steps:
        rc, out, err = run(f"{sys.executable} {args}", cwd=C, timeout=3600)
        record("D", f"`python {args}` runs", rc == 0, err[-300:] if rc else "")
    for f in regen:
        a, b = f"{REPO}/{f}", f"{C}/{f}"
        if not os.path.exists(b):
            record("D", f"{f} regenerates identically", False, "not written")
            continue
        same = (json.load(open(a)) == json.load(open(b))) if f.endswith(".json") else open(a).read() == open(b).read()
        record("D", f"{f} regenerates identically", same)
    tabs = sorted(glob.glob(f"{REPO}/paper/tables/*.tex"))
    diffs = [os.path.basename(t) for t in tabs if not os.path.exists(t.replace(REPO, C, 1))
             or open(t).read() != open(t.replace(REPO, C, 1)).read()]
    record("D", f"all {len(tabs)} paper tables regenerate identically", not diffs, ", ".join(diffs))
    figs = sorted(glob.glob(f"{REPO}/paper/figs/*.pdf"))
    diffs = []
    if shutil.which("pdftotext"):
        for f in figs:
            g = f.replace(REPO, C, 1)
            ta = subprocess.run(["pdftotext", f, "-"], capture_output=True, text=True).stdout
            tb = subprocess.run(["pdftotext", g, "-"], capture_output=True, text=True).stdout if os.path.exists(g) else None
            if ta != tb:
                diffs.append(os.path.basename(f))
        record("D", f"all {len(figs)} figures regenerate (identical text content)", not diffs, ", ".join(diffs))
    else:
        diffs = [os.path.basename(f) for f in figs if not os.path.exists(f.replace(REPO, C, 1))]
        record("D", f"all {len(figs)} figures regenerate (pdftotext not available: existence only)", not diffs, ", ".join(diffs))


# ------------------------------------------------------------------------------------------------ E: training
TRAIN = [  # the recorded commands (queue/jobs.jsonl), cut to 100k steps
    ("L2", "ppo_s100", "--seed 100", 100_000),
    ("L2", "init_r1_s400", "--seed 400 --ckpt-every 50000 --init-actor packs/init/r1/work/weights.npz --critic-warmup 50000", 100_000),
    ("L2W", "transfer_net0_s500", "--seed 500 --ckpt-every 50000 --init-actor packs/transfer/net0/work/weights.npz "
                                  "--critic-warmup 50000", 100_000),
    ("L2", "ft_net0_s600", "--seed 600 --ckpt-every 25000 --init-model models/L2/ppo_s100/final.zip", 100_000),
    ("L2_rules", "ppo_s300", "--seed 300", 100_000),
    ("L2_fault", "ft_net0_s800", "--seed 800 --ckpt-every 25000 --init-model models/L2/ppo_s100/final.zip", 100_000),
]


def stage_E(C):
    procs = []
    for prob, name, args, steps in TRAIN:
        cmd = (f"{sys.executable} -m wsp.train --problem {prob} --name audit_{name} --steps {steps} {args} "
               f"> {C}/train_audit_{name}.log 2>&1")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        procs.append((prob, name, steps, subprocess.Popen(cmd, cwd=C, shell=True, env=env)))
    import torch
    import zipfile
    import io

    def params(z):
        with zipfile.ZipFile(z) as f:
            return torch.load(io.BytesIO(f.read("policy.pth")), map_location="cpu", weights_only=False)
    for prob, name, steps, p in procs:
        p.wait()
        new = f"{C}/models/{prob}/audit_{name}/ckpt_{steps}_steps.zip"
        old = f"{REPO}/models/{prob}/{name}/ckpt_{steps}_steps.zip"
        if p.returncode != 0 or not os.path.exists(new):
            record("E", f"{prob}/{name}: retraining runs", False, open(f"{C}/train_audit_{name}.log").read()[-400:])
            continue
        a, b = params(old), params(new)
        same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        ma = open(f"{REPO}/models/{prob}/{name}/monitor.csv").read().splitlines()[2:]
        mb = open(f"{C}/models/{prob}/audit_{name}/monitor.csv").read().splitlines()[2:]
        n = min(len(ma), len(mb))
        same_ep = [x.split(",")[:2] for x in ma[:n]] == [x.split(",")[:2] for x in mb[:n]]
        record("E", f"{prob}/{name}: 100k-step retraining reproduces the shipped checkpoint parameter for parameter "
                    f"({len(a)} tensors) and the first {n} training episodes' returns and lengths", same and same_ep and n > 0)


# ------------------------------------------------------------------------------------------------ F: baselines
def stage_F(C, workers):
    os.remove(f"{C}/results/tuned/L2.json")
    rc, out, err = run(f"{sys.executable} -m wsp.tune --problem L2 --workers {workers}", cwd=C, timeout=7200)
    same = rc == 0 and json.load(open(f"{REPO}/results/tuned/L2.json")) == json.load(open(f"{C}/results/tuned/L2.json"))
    record("F", "tuning of fixed-time (270 plans) and actuated control (180 settings) on L2 reproduces "
                "results/tuned/L2.json", same, err[-300:] if rc else "")
    import numpy as np
    for prob, tag in (("L2", ""), ("L2_fault", "fault_")):
        base = f"es_{tag}net0_sig0.3_seed0"
        os.remove(f"{C}/results/baselines/{base}.json")
        os.remove(f"{C}/models/L2/baselines/{base}.npz")
        rc, out, err = run(f"{sys.executable} -c \"from wsp.baselines import es_run; es_run(0, 0.3, 0, '{prob}')\"",
                           cwd=C, timeout=7200)
        a, b = json.load(open(f"{REPO}/results/baselines/{base}.json")), (
            json.load(open(f"{C}/results/baselines/{base}.json")) if rc == 0 else {})
        keys = ("log", "submitted_call", "submitted_dev", "params_changed", "val_tts", "start")
        za, zb = np.load(f"{REPO}/models/L2/baselines/{base}.npz"), (np.load(f"{C}/models/L2/baselines/{base}.npz") if rc == 0 else None)
        same = rc == 0 and all(a[k] == b[k] for k in keys) and all(np.array_equal(za[k], zb[k]) for k in za.files)
        record("F", f"(1+1)-ES on {prob} (Net 0, sigma0 0.3, seed 0): all 10 development evaluations, the accepted steps and "
                    f"the submitted network reproduce", same, err[-300:] if rc else "")
    for stage, files in (("meanimp", [f"models/L2/baselines/meanimp_net{k}.npz" for k in range(3)]),
                         ("pad", [f"models/L2W/baselines/pad_net{k}.npz" for k in range(3)]),
                         ("cart", [f"models/L2/baselines/cart_net{k}_{n}.py" for k in range(3) for n in ("small", "mid", "full")])):
        for f in files:
            os.remove(f"{C}/{f}")
        rc, out, err = run(f"{sys.executable} -m wsp.baselines {stage} --targets 0,1,2", cwd=C, timeout=7200)
        diffs = []
        for f in files:
            if not os.path.exists(f"{C}/{f}"):
                diffs.append(f + " (missing)")
            elif f.endswith(".npz"):
                za, zb = np.load(f"{REPO}/{f}"), np.load(f"{C}/{f}")
                if not all(np.array_equal(za[k], zb[k]) for k in za.files):
                    diffs.append(f)
            elif open(f"{REPO}/{f}").read() != open(f"{C}/{f}").read():
                diffs.append(f)
        record("F", f"baseline `{stage}` regenerates its {len(files)} outputs identically", rc == 0 and not diffs,
               (err[-300:] if rc else "") + ", ".join(diffs))
    log_path = f"{C}/results/baselines/dagger.json"
    lg = json.load(open(log_path))
    files = [f"models/L2/baselines/dagger_net0_{n}.py" for n in ("small", "mid", "full")]
    for n in ("small", "mid", "full"):
        lg.pop(f"net0_{n}", None)
    json.dump(lg, open(log_path, "w"), indent=1)
    for f in files:
        os.remove(f"{C}/{f}")
    rc, out, err = run(f"{sys.executable} -m wsp.baselines dagger --targets 0", cwd=C, timeout=7200)
    diffs = [f for f in files if not os.path.exists(f"{C}/{f}") or open(f"{REPO}/{f}").read() != open(f"{C}/{f}").read()]
    same_log = rc == 0 and all(json.load(open(f"{REPO}/results/baselines/dagger.json"))[f"net0_{n}"]["selected_iter"]
                               == json.load(open(log_path))[f"net0_{n}"]["selected_iter"] for n in ("small", "mid", "full"))
    record("F", "DAgger distillation of Net 0 (3 depths x 6 iterations with simulator rollouts) regenerates the same "
                "selected rule files", rc == 0 and not diffs and same_log, (err[-300:] if rc else "") + ", ".join(diffs))


# ------------------------------------------------------------------------------------------------ G: agents
def stage_G(C):
    rebuild = os.path.join(C, "audit_packs")
    packs = sorted(glob.glob(f"{REPO}/packs/*/*/spec.json"))
    script_lines = ["import os, sys", "from wsp import packs as P"]
    targets = []
    for sp in packs:
        d = os.path.dirname(sp)
        case, pid = d.split("/")[-2:]
        build_case = "patch" if case == "patch_models" else case
        if case == "init":
            run_, budget, rep = int(pid[1]), 0, 1
        else:
            run_ = int(pid[3])
            budget = int(pid.split("_b")[1].split("_")[0]) if "_b" in pid else (10 if case == "patch_models" else 0)
            rep = int(pid.rsplit("_r", 1)[1]) if pid.rsplit("_", 1)[-1].startswith("r") and case != "init" else 1
        if case == "patch_models":
            key = (build_case, run_, budget, 1)
        else:
            key = (build_case, run_, budget, rep)
        targets.append((case, pid, key))
    keys = sorted(set(t[2] for t in targets))
    script_lines.append(f"for c, r, b, rep in {keys!r}:\n    P.build(c, r, b, rep)")
    open(f"{C}/rebuild_packs.py", "w").write("\n".join(script_lines) + "\n")
    rc, out, err = run(f"{sys.executable} {C}/rebuild_packs.py", cwd=C, timeout=7200, env_extra={"WSP_PACK_ROOT": rebuild})
    if rc != 0:
        record("G", "agent task folders rebuild", False, err[-500:])
    else:
        diffs, nfiles = [], 0
        for case, pid, (bc, r_, b_, rep) in targets:
            new_pid = (f"r{r_}" if bc == "init" else (f"net{r_}_b{b_}" if bc in ("patch", "constrain", "patch_fault") else f"net{r_}"))
            new_pid = new_pid if rep == 1 else f"{new_pid}_r{rep}"
            nd, od = f"{rebuild}/{bc}/{new_pid}", f"{REPO}/packs/{case}/{pid}"
            for f in sorted(os.listdir(nd)):
                if f in ("work", "__pycache__"):
                    continue
                nfiles += 1
                a, b = f"{od}/{f}", f"{nd}/{f}"
                if not os.path.exists(a):
                    diffs.append(f"{case}/{pid}/{f} (not shipped)")
                    continue
                if f == "spec.json":
                    ja, jb = json.load(open(a)), json.load(open(b))
                    ja.pop("pack_id"), jb.pop("pack_id")
                    same = ja == jb
                elif f == "eval.py":
                    strip = lambda s: "\n".join(l for l in s.splitlines() if "PIPELINE =" not in l)  # noqa: E731
                    same = strip(open(a).read()) == strip(open(b).read())
                elif f == "TASK.md":
                    import re
                    strip = lambda s: re.sub(r"Use `[^`]+`", "Use `<python>`", s)  # noqa: E731
                    same = strip(open(a).read()) == strip(open(b).read())
                else:
                    same = sha(a) == sha(b)
                if not same:
                    diffs.append(f"{case}/{pid}/{f}")
        record("G", f"all {len(targets)} agent task folders rebuild identically from the generator ({nfiles} files; "
                    f"pack ids, the Python interpreter path and the evaluator's pipeline path excepted)", not diffs, ", ".join(diffs[:10]))
    # re-score every deliverable (test evaluations are reused from results/eval, which stage C re-runs from scratch)
    before = {p: json.load(open(p)) for p in glob.glob(f"{REPO}/packs/*/*/scored.json")}
    shutil.copy(f"{C}/queue/jobs.jsonl", f"{C}/queue/jobs.jsonl.bak")
    paths = " ".join(os.path.dirname(p).replace(REPO, C, 1) for p in sorted(before))
    rc, out, err = run(f"{sys.executable} -m wsp.agents_eval {paths}", cwd=C, timeout=7200)
    keys = ("missing", "predictions", "trainability", "params_changed", "within_cap", "comparisons_small",
            "comparisons_full", "fidelity_small", "fidelity_full")
    diffs = []
    for p, a in before.items():
        q = p.replace(REPO, C, 1)
        b = json.load(open(q)) if os.path.exists(q) else {}
        for k in keys:
            if k in a and a[k] != b.get(k):
                diffs.append(f"{os.path.relpath(os.path.dirname(p), REPO)}:{k}")
    record("G", f"all {len(before)} agent deliverables re-score identically (edit size, cap, trainability, comparisons, "
                f"fidelity, predictions)", rc == 0 and not diffs, (err[-300:] if rc else "") + ", ".join(diffs[:10]))


# ------------------------------------------------------------------------------------------------ report
def manifest():
    if os.path.isdir(os.path.join(REPO, ".git")):
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO, text=True
        )
        rels = sorted(set(out.splitlines()))
    else:
        rels = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = sorted(d for d in dirs if d not in (".git", "__pycache__") and not d.startswith("audit_work"))
            for f in sorted(files):
                rels.append(os.path.relpath(os.path.join(root, f), REPO))
        rels.sort()
    lines = []
    for rel in rels:
        if rel in ("audit/AUDIT_REPORT.md", "audit/audit_results.json", "audit/MANIFEST.sha256"):
            continue
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            lines.append(f"{sha(p)}  {rel}")
    open(f"{REPO}/audit/MANIFEST.sha256", "w").write("\n".join(lines) + "\n")
    return len(lines)


def report(t0, stages, versions):
    checks = [(s, c) for s in sorted(k for k in RESULTS if not k.startswith("_")) for c in RESULTS[s]]
    n_ok = sum(c["ok"] for _, c in checks)
    names = {"A": "Static checks", "B": "Simulation inputs", "C": "Episodes", "D": "Analysis, tables and figures",
             "E": "Training", "F": "Baselines and tuning", "G": "Agent task folders and scoring"}
    L = ["# Reproducibility audit", "",
         f"Run {time.strftime('%Y-%m-%d %H:%M')} on {os.uname().nodename}; stages {','.join(stages)}; "
         f"{time.time() - t0:.0f} s. Software: {versions}.", "",
         f"**{n_ok} of {len(checks)} checks passed.**", ""]
    for s in sorted(k for k in RESULTS if not k.startswith("_")):
        L += [f"## {s}. {names.get(s, s)}", "", "| | Check | Detail |", "|---|---|---|"]
        for c in RESULTS[s]:
            L.append(f"| {'PASS' if c['ok'] else '**FAIL**'} | {c['check']} | {c['detail'][:300].replace('|', '/')} |")
        L.append("")
    open(f"{REPO}/audit/AUDIT_REPORT.md", "w").write("\n".join(L))
    json.dump({"results": RESULTS, "log": LOG}, open(f"{REPO}/audit/audit_results.json", "w"), indent=1)
    return n_ok, len(checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="A,B,C,D,E,F,G")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--workdir", default=None, help="parent folder of the working copy (default: system temp)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    stages = a.stages.split(",")
    t0 = time.time()
    import numpy, scipy, torch, gymnasium, stable_baselines3, matplotlib  # noqa: E401
    try:
        import importlib.metadata
        sumo = f"libsumo {importlib.metadata.version('libsumo')}"
    except Exception:
        sumo = "?"
    versions = (f"Python {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}, torch "
                f"{torch.__version__}, gymnasium {gymnasium.__version__}, stable-baselines3 {stable_baselines3.__version__}, "
                f"matplotlib {matplotlib.__version__}, {sumo}")
    C = os.path.join(tempfile.mkdtemp(prefix="wsp_audit_", dir=a.workdir), "repo")
    make_copy(C)
    try:
        for s in stages:
            log(f"=== stage {s}")
            try:
                {"A": lambda: stage_A(C), "B": lambda: stage_B(C, a.workers), "C": lambda: stage_C(C, a.workers),
                 "D": lambda: stage_D(C), "E": lambda: stage_E(C), "F": lambda: stage_F(C, a.workers),
                 "G": lambda: stage_G(C)}[s]()
            except Exception:
                record(s, f"stage {s} completed without an exception", False, traceback.format_exc()[-600:])
        n = manifest()
        log(f"manifest of {n} files written to audit/MANIFEST.sha256")
    finally:
        if not a.keep:
            shutil.rmtree(os.path.dirname(C), ignore_errors=True)
    ok, tot = report(t0, stages, versions)
    log(f"{ok}/{tot} checks passed; report in audit/AUDIT_REPORT.md")
    sys.exit(0 if ok == tot else 1)


if __name__ == "__main__":
    main()
