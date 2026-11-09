# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Core-budgeted job queue for the pipeline.

    (from the repository root)
    python -m wsp.queue run                  # the runner (start once, in the background: bash scripts/run_queue.sh)
    python -m wsp.queue add FILE.jsonl       # append jobs
    echo 30 > queue/budget                   # change the core budget live (default 36)
    echo auto > queue/budget                 # = all cores minus other users' load minus 2 (auto:4 for margin 4)
    echo <job-id or prefix*> >> queue/cancel # skip pending jobs
    touch queue/stop                         # runner exits once nothing is running

A job is one JSON line: {"id", "cmd", "cores", "after": [ids], "kind": "tune|train|eval", "steps", "run_dir",
"prio" (higher starts first, default 0)}.
State lives in queue/state.json; logs in queue/logs/<id with / -> __>.log. Commands run from the repository root.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the repository root
Q = os.path.join(ROOT, "queue")
JOBS, STATE, BUDGET, CANCEL, STOP = (os.path.join(Q, n) for n in ("jobs.jsonl", "state.json", "budget", "cancel", "stop"))
# jobs run with the interpreter that runs the queue (start it inside the environment of environment.yml)
ENV_PREFIX = ("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
              "export PATH=" + os.path.dirname(sys.executable) + ":$PATH && cd " + ROOT + " && ")


def read_jobs():
    """Jobs in scheduling order: higher "prio" first (default 0), then file order."""
    if not os.path.exists(JOBS):
        return []
    jobs = [json.loads(l) for l in open(JOBS) if l.strip()]
    return sorted(jobs, key=lambda j: -j.get("prio", 0))


def read_state():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {}


def write_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def budget():
    """Core budget. The budget file holds a number, or 'auto[:margin]' = all cores minus what other
    (non-queue) processes use minus margin (default 2), re-measured by the runner every 30 s."""
    try:
        txt = open(BUDGET).read().strip()
    except OSError:
        return 36.0
    if txt.startswith("auto"):
        try:
            return float(open(BUDGET + "_now").read().split()[0])
        except (OSError, ValueError, IndexError):
            return 24.0
    try:
        return float(txt)
    except ValueError:
        return 36.0


def measure_auto():
    """Write budget_now = cores - (cpu of processes outside this runner's tree) - margin."""
    import psutil
    txt = open(BUDGET).read().strip()
    margin = float(txt.split(":")[1]) if ":" in txt else 2.0
    ours = set()
    roots = [os.getpid()] + [v["pid"] for v in read_state().values() if v.get("status") == "running" and v.get("pid")]
    for r in roots:
        try:
            pr = psutil.Process(r)
            ours |= {pr.pid} | {c.pid for c in pr.children(recursive=True)}
        except psutil.Error:
            pass
    procs = [p for p in psutil.process_iter() if p.pid not in ours]
    for p in procs:
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    time.sleep(3)
    other = 0.0
    for p in procs:
        try:
            other += p.cpu_percent(None) / 100.0
        except psutil.Error:
            pass
    b = max(9.0, psutil.cpu_count() - other - margin)
    open(BUDGET + "_now", "w").write(f"{b:.1f} other={other:.1f} at {time.strftime('%H:%M:%S')}\n")
    return b


def cancelled(jid):
    if not os.path.exists(CANCEL):
        return False
    for p in (l.strip() for l in open(CANCEL)):
        if p and (jid == p or (p.endswith("*") and jid.startswith(p[:-1]))):
            return True
    return False


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def log_path(jid):
    return os.path.join(Q, "logs", jid.replace("/", "__") + ".log")


def adopted_rc(jid, job):
    """Exit code of a job whose process this runner did not start (runner restarted meanwhile)."""
    rcf = log_path(jid)[:-4] + ".rc"
    if os.path.exists(rcf):
        try:
            return int(open(rcf).read().strip())
        except ValueError:
            pass
    try:                                          # jobs started before rc files existed
        log = open(log_path(jid)).read()
    except OSError:
        return -999
    if "Traceback" in log:
        return -998
    if job and job["kind"] == "train":
        return 0 if os.path.exists(os.path.join(ROOT, job["run_dir"], "final.zip")) else -997
    return 0


def runner_alive():
    try:
        return alive(int(open(os.path.join(Q, "runner.pid")).read()))
    except (OSError, ValueError):
        return False


def run():
    os.makedirs(os.path.join(Q, "logs"), exist_ok=True)
    open(os.path.join(Q, "runner.pid"), "w").write(str(os.getpid()))
    st = read_state()
    procs = {}
    print(f"[{time.strftime('%H:%M:%S')}] runner up, pid {os.getpid()}", flush=True)
    last_auto = 0.0
    while True:
        jobs = read_jobs()
        by_id = {j["id"]: j for j in jobs}
        try:
            if open(BUDGET).read().strip().startswith("auto") and time.time() - last_auto > 30:
                measure_auto()
                last_auto = time.time()
        except OSError:
            pass
        # poll
        for jid, s in st.items():
            if s["status"] != "running":
                continue
            p = procs.get(jid)
            rc = p.poll() if p is not None else (None if alive(s["pid"]) else adopted_rc(jid, by_id.get(jid)))
            if rc is not None:
                s["end"] = time.time()
                s["rc"] = rc
                s["status"] = "done" if rc == 0 else "failed"
                procs.pop(jid, None)
                print(f"[{time.strftime('%H:%M:%S')}] {s['status']} {jid} rc={rc}", flush=True)
        used = sum(j["cores"] for j in jobs if st.get(j["id"], {}).get("status") == "running")
        cap = budget()
        for j in jobs:
            s = st.setdefault(j["id"], {"status": "pending"})
            if s["status"] != "pending":
                continue
            if cancelled(j["id"]):
                s["status"] = "cancelled"
                continue
            deps = [st.get(d, {}).get("status") for d in j.get("after", [])]
            if any(d in ("failed", "blocked", "cancelled") for d in deps):
                s["status"] = "blocked"
                continue
            if not all(d == "done" for d in deps):
                continue
            if used + j["cores"] > cap + 1e-9 and used > 0:
                continue
            log = open(log_path(j["id"]), "w")
            rcf = log_path(j["id"])[:-4] + ".rc"
            if os.path.exists(rcf):
                os.remove(rcf)
            wrapped = f"( {j['cmd']} ); rc=$?; echo $rc > {rcf}; exit $rc"
            p = subprocess.Popen(["bash", "-c", ENV_PREFIX + wrapped], stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            procs[j["id"]] = p
            s.update(status="running", start=time.time(), pid=p.pid)
            used += j["cores"]
            print(f"[{time.strftime('%H:%M:%S')}] start {j['id']} ({used:.1f}/{cap:.0f} cores)", flush=True)
        write_state(st)
        if os.path.exists(STOP) and not any(s["status"] == "running" for s in st.values()):
            print("stop file found; exiting", flush=True)
            return
        time.sleep(5)


def add(path):
    have = {j["id"] for j in read_jobs()}
    new = [json.loads(l) for l in open(path) if l.strip()]
    with open(JOBS, "a") as f:
        for j in new:
            if j["id"] in have:
                print("skip duplicate", j["id"])
                continue
            f.write(json.dumps(j) + "\n")
    print(f"added {len([j for j in new if j['id'] not in have])} jobs")


if __name__ == "__main__":
    os.makedirs(Q, exist_ok=True)
    if sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "add":
        add(sys.argv[2])
