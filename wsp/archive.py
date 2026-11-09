# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Archive and audit the agent runs from the Claude Code session store.

    (from the repository root)
    python -m wsp.archive headless                   # copy headless session transcripts, write audits
    python -m wsp.archive archive <session.jsonl>    # copy subagent transcripts, write audits
    python -m wsp.archive table                      # LaTeX summary table

For every run in packs/<use>/<run>, `archive` writes packs/<use>/<run>/provenance/:
- launch_prompt.txt  the exact text the agent was launched with;
- transcript.jsonl   the archived session transcript (redact account identifiers before public release);
- audit.json         model, timing, tool calls, simulator calls, messages the agent received after launch,
                     and automatic compliance flags.

It also writes runs/agent_runs.json, one row per run. The flags are screens for a human reviewer; each one
is adjudicated in docs/PROVENANCE.md.
"""
import datetime as dt
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

V3 = Path(__file__).resolve().parent.parent
ROOT = V3.parent
PAPER_EXP = V3 / "packs"
PACK_RE = re.compile(r"v3/packs/([a-z0-9_]+)/([a-z0-9_]+)")
HANDBACK_REMINDER = "Your final report is delivered through SubagentHandback"
# Screens for rule 4 of TASK.md (no fitting, no automated search). Hits are reviewed by hand.
FIT_RE = re.compile(r"lstsq|polyfit|curve_fit|scipy\.optimize|minimize\(|sklearn|import torch|\.backward\(|"
                    r"optim\.|LinearRegression|DecisionTree|np\.linalg\.(?:solve|pinv)|itertools\.product")
SIM_RE = re.compile(r"\b(?:sumo|traci|libsumo|TSCEnv)\b", re.I)
GAP_S = 600


def text_of(content):
    if isinstance(content, str):
        return [content]
    return [x.get("text", "") for x in content or [] if isinstance(x, dict) and x.get("type") == "text"]


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def harness_usage(session):
    """subagent_tokens, tool_uses and duration_ms that the harness reported for each agent id."""
    out = {}
    pat = re.compile(r'<agent-message from=\\?"(a[0-9a-f]+)\\?".*?<subagent_tokens>(\d+)</subagent_tokens>'
                     r"<tool_uses>(\d+)</tool_uses><duration_ms>(\d+)</duration_ms>", re.S)
    for line in open(session):
        if "<subagent_tokens>" not in line:
            continue
        for m in pat.finditer(line):
            out[m.group(1)] = {"subagent_tokens": int(m.group(2)), "tool_uses": int(m.group(3)),
                               "duration_ms": int(m.group(4))}
    return out


def audit_transcript(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    launch, models, calls, seen, msg_ids = None, Counter(), [], set(), set()
    out_tokens, received, times = 0, [], []
    for r in recs:
        if r.get("type") not in ("user", "assistant"):
            continue          # attachments, api-request records (blobs repeat earlier content)
        if r.get("timestamp"):
            times.append(ts(r["timestamp"]))
        m = r.get("message") or {}
        if r["type"] == "user":
            for t in text_of(m.get("content")):
                if not t:
                    continue
                if launch is None:
                    launch = t
                else:
                    kind = "harness_handback_reminder" if HANDBACK_REMINDER in t else "other"
                    received.append({"time": r.get("timestamp"), "kind": kind, "text": t})
        else:
            if m.get("model"):
                models[m["model"]] += 1
            if m.get("id") and m["id"] not in msg_ids:
                msg_ids.add(m["id"])
                out_tokens += (m.get("usage") or {}).get("output_tokens", 0)
            for x in m.get("content") or []:
                if isinstance(x, dict) and x.get("type") == "tool_use" and x["id"] not in seen:
                    seen.add(x["id"])
                    calls.append(x)
    times.sort()
    gaps = [(a, b) for a, b in zip(times, times[1:]) if (b - a).total_seconds() > GAP_S]
    active = (times[-1] - times[0]).total_seconds() - sum((b - a).total_seconds() for a, b in gaps)
    return launch, models, calls, received, times, gaps, active, out_tokens


def flags_for(calls, pack):
    pack_s, flags = str(pack), {}

    def add(key, val):
        flags.setdefault(key, [])
        if val not in flags[key]:
            flags[key].append(val)

    for x in calls:
        inp = x["input"]
        text = inp.get("command") or inp.get("content") or inp.get("new_string") or ""
        target = inp.get("file_path") or ""
        for p in re.findall(r"(/home/[^\s'\"`;|)>]+)", text + " " + target):
            if not p.startswith(pack_s) and "/anaconda3/" not in p and "/miniconda3/" not in p:
                add("path_outside_pack", p[:160])
        if re.search(r"(?<![\w.])\.\./\.\./", text):
            add("relative_path_two_levels_up", x["id"])
        if x["name"] in ("Write", "Edit") and not target.startswith(pack_s + "/work"):
            add("write_outside_work", target)
        if x["name"] not in ("Bash", "Read", "Write", "Edit", "SubagentHandback", "ToolSearch"):
            add("other_tool", x["name"])
        m = FIT_RE.search(text)
        if m:
            add("fit_or_search_keyword", {"tool_use_id": x["id"], "match": m.group(0)})
        if x["name"] == "Bash" and SIM_RE.search(inp.get("command", "")) and "eval.py" not in inp.get("command", ""):
            add("simulator_keyword_outside_eval", x["id"])
    return flags


def archive(session):
    session = Path(session)
    sub = session.with_suffix("") / "subagents"
    usage = harness_usage(session)
    out_path = PAPER_EXP / "agent_runs.json"
    old = json.loads(out_path.read_text()) if out_path.exists() else []
    rows = []
    for path in sorted(sub.glob("agent-*.jsonl")):
        launch, models, calls, received, times, gaps, active, out_tokens = audit_transcript(path)
        m = PACK_RE.search(launch or "")
        if not m:
            continue
        use, run = m.groups()
        pack = PAPER_EXP / use / run
        prov = pack / "provenance"
        prov.mkdir(exist_ok=True)
        shutil.copy2(path, prov / "transcript.jsonl")
        (prov / "launch_prompt.txt").write_text(launch)
        agent_id = path.stem.removeprefix("agent-")
        eval_log = pack / "eval_log.jsonl"
        sim_calls = sum(1 for line in open(eval_log) if line.strip()) if eval_log.exists() else 0
        spec = json.loads((pack / "spec.json").read_text())
        mirror = V3 / "runs" / "agent_eval_log" / f"{spec['pack_id']}.jsonl"
        sim_mirror = sum(1 for line in open(mirror) if line.strip()) if mirror.exists() else 0
        audit = {
            "use": use, "run": run, "agent_id": agent_id, "source": str(path),
            "models": dict(models),
            "start": times[0].isoformat(), "end": times[-1].isoformat(),
            "active_minutes": round(active / 60, 1),
            "gaps_over_10_min": [[a.isoformat(), b.isoformat()] for a, b in gaps],
            "tool_calls": dict(Counter(c["name"] for c in calls)),
            "simulator_calls_logged": sim_calls,
            "simulator_calls_mirror_log": sim_mirror,
            "simulator_budget": spec["eval_budget"],
            "output_tokens": out_tokens,
            "harness_usage": usage.get(agent_id),
            "messages_received_after_launch": received,
            "flags": flags_for(calls, pack),
        }
        (prov / "audit.json").write_text(json.dumps(audit, indent=1))
        rows.append({k: audit[k] for k in ("use", "run", "agent_id", "models", "start", "active_minutes",
                                            "tool_calls", "simulator_calls_logged", "output_tokens",
                                            "harness_usage")}
                    | {"n_messages_other": sum(1 for r in received if r["kind"] != "harness_handback_reminder"),
                       "flags": sorted(audit["flags"])})
        print(f"{use}/{run}: {dict(models)} tools={sum(audit['tool_calls'].values())} sim={sim_calls} "
              f"active={audit['active_minutes']}min flags={sorted(audit['flags'])}")
    new = {(r["use"], r["run"]) for r in rows}
    rows += [r for r in old if (r["use"], r["run"]) not in new]
    order = {"init": 0, "patch": 1, "patch_models": 2, "transfer": 3, "explain": 4, "constrain": 5}
    rows.sort(key=lambda r: (order.get(r["use"], 9), r["run"]))
    out_path.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} runs archived")


def archive_headless():
    """Runs launched as independent headless sessions (wsp.launch): one transcript per run in the session store."""
    store = Path.home() / ".claude" / "projects"
    rows = []
    for rec_path in sorted((V3 / "runs" / "agent_launch").glob("*__*.json")):
        rec = json.loads(rec_path.read_text())
        pack = Path(rec["pack"])
        use, run = pack.parent.name, pack.name
        sid = (rec.get("result") or {}).get("session_id")
        slug = str(pack).replace("/", "-").replace("_", "-")
        tr = store / slug / f"{sid}.jsonl"
        if not sid or not tr.exists():
            print("MISSING transcript", use, run, sid)
            continue
        launch, models, calls, received, times, gaps, active, out_tokens = audit_transcript(tr)
        prov = pack / "provenance"
        prov.mkdir(exist_ok=True)
        shutil.copy2(tr, prov / "transcript.jsonl")
        sub = store / slug / sid / "subagents"
        if sub.exists():
            shutil.copytree(sub, prov / "subagents", dirs_exist_ok=True)
        (prov / "launch_prompt.txt").write_text(rec["prompt"])
        spec = json.loads((pack / "spec.json").read_text())
        log = pack / "eval_log.jsonl"
        mirror = V3 / "runs" / "agent_eval_log" / f"{spec['pack_id']}.jsonl"
        n_log = sum(1 for l in open(log) if l.strip()) if log.exists() else 0
        n_mir = sum(1 for l in open(mirror) if l.strip()) if mirror.exists() else 0
        audit = {"use": use, "run": run, "session_id": sid, "source": str(tr), "requested_model": rec["model"],
                 "models": dict(models), "prompt_matches_launch": (launch or "").strip() == rec["prompt"].strip(),
                 "start": times[0].isoformat(), "end": times[-1].isoformat(), "active_minutes": round(active / 60, 1),
                 "gaps_over_10_min": [[a.isoformat(), b.isoformat()] for a, b in gaps],
                 "tool_calls": dict(Counter(c["name"] for c in calls)), "simulator_calls_logged": n_log,
                 "simulator_calls_mirror_log": n_mir, "simulator_budget": spec["eval_budget"],
                 "output_tokens": out_tokens, "cost_usd": rec["result"].get("total_cost_usd"),
                 "turns": rec["result"].get("num_turns"), "permission_denials": rec["result"].get("permission_denials"),
                 "subagents": sub.exists(), "messages_received_after_launch": received,
                 "flags": flags_for(calls, pack)}
        (prov / "audit.json").write_text(json.dumps(audit, indent=1))
        rows.append({k: audit[k] for k in ("use", "run", "session_id", "requested_model", "models",
                                            "prompt_matches_launch", "start", "active_minutes", "tool_calls",
                                            "simulator_calls_logged", "simulator_calls_mirror_log", "simulator_budget",
                                            "output_tokens", "cost_usd", "turns", "subagents")}
                    | {"n_messages_after_launch": len(received), "flags": sorted(audit["flags"])})
        print(f"{use}/{run}: {dict(models)} prompt_ok={audit['prompt_matches_launch']} tools={sum(audit['tool_calls'].values())} "
              f"sim={n_log}/{n_mir}/{spec['eval_budget']} active={audit['active_minutes']}min flags={sorted(audit['flags'])}")
    order = {"init": 0, "patch": 1, "patch_models": 2, "transfer": 3, "explain": 4, "constrain": 5}
    rows.sort(key=lambda r: (order.get(r["use"], 9), r["run"]))
    (V3 / "runs" / "agent_runs.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} runs archived")


def table():
    rows = json.loads((PAPER_EXP / "agent_runs.json").read_text())
    names = {"init": "Initialise", "patch": "Patch", "patch_models": "Patch, other writers",
             "transfer": "Transfer", "explain": "Explain", "constrain": "Constrain"}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Use & Run & Tool calls & Feedback rounds & Active min & Harness min & Tokens (k) \\", r"\midrule"]
    prev = None
    for r in rows:
        if prev and r["use"] != prev:
            lines.append(r"\midrule")
        u = r["harness_usage"] or {}
        lines.append(f"{names[r['use']] if r['use'] != prev else ''} & {r['run'].replace('_', ' ')} & "
                     f"{sum(r['tool_calls'].values())} & {r['simulator_calls_logged'] or '--'} & "
                     f"{r['active_minutes']:.0f} & {u.get('duration_ms', 0) / 60000:.0f} & "
                     f"{u.get('subagent_tokens', 0) / 1000:.0f} \\\\")
        prev = r["use"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = V3 / "paper" / "tables" / "agent_runs.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    if sys.argv[1] == "archive":
        archive(sys.argv[2])
    elif sys.argv[1] == "headless":
        archive_headless()
    elif sys.argv[1] == "table":
        table()
    else:
        sys.exit(__doc__)
