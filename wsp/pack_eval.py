"""eval.py: run YOUR network (or supervisor) in the traffic simulator on the dev episodes (closed-loop arm only).

    python eval.py work/<file>.npz        (a supervisor task: python eval.py work/supervisor.py)

Each call runs the 9 dev episodes listed in TASK.md and counts against your budget. It prints the total time in
system (TTS, veh-h; lower is better) per episode, on average and per scenario, the vehicles left at the drain cap
and the phase switches per episode, next to the reference controllers, and logs the call. Once the budget is used
up it refuses to run.
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "spec.json")) as f:
    SPEC = json.load(f)
PIPELINE = "@@PIPELINE@@"          # filled in when the pack is built
CALL_S = "@@CALL_S@@"


def used_calls(logs):
    return max((sum(1 for _ in open(p)) if os.path.exists(p) else 0) for p in logs)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = os.path.abspath(sys.argv[1])
    log = os.path.join(HERE, "eval_log.jsonl")
    mirror = os.path.join(PIPELINE, "runs", "agent_eval_log", SPEC["pack_id"] + ".jsonl")
    used = used_calls([log, mirror])
    budget = SPEC["eval_budget"]
    if used >= budget:
        sys.exit(f"budget used up ({used}/{budget}); no more simulator evaluations")
    import tools                     # this folder's tools.py
    sys.path.append(PIPELINE)
    if SPEC["case"] == "constrain":
        if not path.endswith(".py"):
            sys.exit("give the supervisor file: python eval.py work/supervisor.py")
        policy = f"super:{path}|{os.path.join(HERE, 'start_weights.npz')}"
    else:
        if not path.endswith(".npz"):
            sys.exit("give a weights file: python eval.py work/<file>.npz")
        tools.save(tools.load(path), os.path.join(HERE, ".eval_check.npz"))     # shape / finiteness check
        policy = f"npz:{path}"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from wsp.evaluate import jobs_for, run_many
    label = f"agent_{SPEC['pack_id']}_{used + 1:02d}"
    rows = run_many(jobs_for(SPEC["problem"], policy, "dev", label), 1)
    tts = [r["tts_veh_h"] for r in rows]
    rec = {"call": used + 1, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": path,
           "sha1": hashlib.sha1(open(path, "rb").read()).hexdigest()[:12],
           "tts_per_episode": tts, "tts_mean": sum(tts) / len(tts),
           "remaining_vehicles": [r["remaining"] for r in rows], "switches": [r["switches"][0] for r in rows],
           "overrides": [r["overrides"][0] for r in rows] if SPEC["case"] == "constrain" else None}
    for p in (log, mirror):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"evaluation {used + 1}/{budget}: TTS per episode {[round(x, 1) for x in tts]}  mean {rec['tts_mean']:.1f} veh-h")
    scen = {}
    for r in rows:
        scen.setdefault(r["scenario"], []).append(r["tts_veh_h"])
    print("mean per scenario:", {k: round(sum(v) / len(v), 1) for k, v in scen.items()})
    print("vehicles left at the drain cap:", rec["remaining_vehicles"])
    print("phase switches per episode:", rec["switches"])
    if rec["overrides"] is not None:
        print("shield overrides per episode (requests the rules overruled):", rec["overrides"])
    print("references (mean TTS over the same 9 dev episodes):", {k: round(v, 1) for k, v in SPEC["dev_refs"].items()})


if __name__ == "__main__":
    main()
