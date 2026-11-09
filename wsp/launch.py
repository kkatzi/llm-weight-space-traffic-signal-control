# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Launch agent runs as independent headless Claude Code sessions (docs/EXPERIMENT_PLAN.md).

    (from the repository root)  nohup python -m wsp.launch --plan runs/agent_launch/plan.json --parallel 12 &

Each run is one `claude -p <launch prompt> --model <id> --allowedTools Bash Read Write Edit Glob Grep
--output-format json` process, started with its pack folder (outside the repository) as the working directory.
There is no parent session, no memory and no git context. The launch prompt is exactly
`python -m wsp.prompts --pack <pack>`.

The plan is a JSON list of {"pack": "<case>/<id>", "model": "<model id>"}, launched in order with at most --parallel
runs at a time. For every run, runs/agent_launch/<case>__<id>.json records the prompt, the model, the start and end
times, the exit code and the CLI's JSON result (session id, turns, cost, permission denials). The session
transcript is ~/.claude/projects/<slug of the pack path>/<session id>.jsonl; wsp.archive copies it into the pack's
provenance/ folder.
"""
import argparse
import glob
import json
import os
import subprocess
import time

from wsp.junctions import V3
from wsp.prompts import render

PACK_ROOT = os.path.realpath(os.environ.get("WSP_PACK_ROOT", os.path.join(V3, "packs")))
TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
LOG = os.environ.get("WSP_LAUNCH_LOG", os.path.join(V3, "runs", "agent_launch"))   # one record per run; never overwritten


def cli():
    import shutil
    c = os.environ.get("CLAUDE_CLI") or shutil.which("claude")
    if not c:
        found = sorted(glob.glob(os.path.expanduser(
            "~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude")))
        c = found[-1] if found else None
    assert c, "claude CLI not found (set CLAUDE_CLI)"
    return c


def start(run):
    pack = os.path.join(PACK_ROOT, run["pack"])
    prompt = render(pack)
    rec = {"pack": pack, "model": run["model"], "prompt": prompt, "start": time.strftime("%Y-%m-%d %H:%M:%S"),
           "cli": cli(), "tools": TOOLS}
    out = os.path.join(LOG, run["pack"].replace("/", "__") + ".json")
    assert not os.path.exists(out), f"{out} exists: this run was launched before"
    json.dump(rec, open(out, "w"), indent=1)
    stdout = open(out.replace(".json", ".stdout"), "w")
    p = subprocess.Popen([cli(), "-p", prompt, "--model", run["model"], "--allowedTools", *TOOLS,
                          "--output-format", "json"], cwd=pack, stdout=stdout, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return p, rec, out


def finish(p, rec, out):
    rec["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec["exit_code"] = p.returncode
    try:
        res = json.load(open(out.replace(".json", ".stdout")))
        rec["result"] = {k: res.get(k) for k in ("session_id", "num_turns", "total_cost_usd", "duration_ms",
                                                 "is_error", "permission_denials", "result")}
    except (ValueError, OSError) as e:
        rec["result"] = {"error": repr(e)}
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[{rec['end']}] done {rec['pack']} rc={p.returncode} session={rec['result'].get('session_id')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--parallel", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(LOG, exist_ok=True)
    plan = json.load(open(a.plan))
    running = []
    for run in plan:
        while len(running) >= a.parallel:
            for x in list(running):
                if x[0].poll() is not None:
                    finish(*x)
                    running.remove(x)
            time.sleep(5)
        running.append(start(run))
        print(f"[{time.strftime('%H:%M:%S')}] start {run['pack']} ({run['model']})", flush=True)
        time.sleep(2)
    while running:
        for x in list(running):
            if x[0].poll() is not None:
                finish(*x)
                running.remove(x)
        time.sleep(5)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
