# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""One-line summary of a scored agent run against its reference (used while runs come in).

    python -m wsp.brief packs/<case>/<id>
"""
import json
import os
import sys

from wsp.stats import load, paired


def brief(pack):
    pack = os.path.realpath(pack)
    case, pid = os.path.basename(os.path.dirname(pack)), os.path.basename(pack)
    sc = json.load(open(os.path.join(pack, "scored.json")))
    launch = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "agent_launch",
                          f"{case}__{pid}.json")
    meta = json.load(open(launch))["result"] if os.path.exists(launch) else {}
    k = int(pid[3]) if pid.startswith("net") else None
    out = []
    if case in ("patch", "patch_models"):
        lab = ("patchm_" if case == "patch_models" else "patch_") + pid
        r = paired(load("L2", lab)["tts"], load("L2", f"ppo_s{100 + k}")["tts"])
        out.append(f"vs target {r['pct']:+.2f}% [{r['pct_lo']:+.2f}, {r['pct_hi']:+.2f}] wins {r['wins']}/30; "
                   f"params {sc.get('params_changed')} (cap ok: {sc.get('within_cap')}); unserved {load('L2', lab)['unserved'].sum()}")
    elif case == "patch_fault":
        x, f, h = load("L2_fault", f"patchf_{pid}")["tts"], load("L2_fault", f"faulty_net{k}")["tts"], load("L2", f"ppo_s{100 + k}")["tts"]
        r = paired(x, f)
        rec = (f.mean() - x.mean()) / (f.mean() - h.mean())
        out.append(f"vs faulty net {r['pct']:+.2f}% [{r['pct_lo']:+.2f}, {r['pct_hi']:+.2f}] wins {r['wins']}/30; "
                   f"recovered {100 * rec:.0f}% of the fault's cost (faulty {f.mean():.1f}, healthy {h.mean():.1f}, patched {x.mean():.1f}); "
                   f"params {sc.get('params_changed')}; unserved {load('L2_fault', f'patchf_{pid}')['unserved'].sum()}")
    elif case == "init":
        x = load("L2", f"init_r{pid[1:]}")
        for b in ("fixed", "actuated"):
            r = paired(x["tts"], load("L2", b)["tts"])
            out.append(f"vs {b} {r['pct']:+.1f}% [{r['pct_lo']:+.1f}, {r['pct_hi']:+.1f}]")
        out.append(f"TTS {x['tts'].mean():.2f}; trainable {sc.get('meets_trainability')}")
    elif case == "transfer":
        x = load("L2W", f"transfer_net{k}")
        for b in ("fixed", "actuated", f"pad_net{k}"):
            r = paired(x["tts"], load("L2W", b)["tts"])
            out.append(f"vs {b} {r['pct']:+.1f}%")
        out.append(f"TTS {x['tts'].mean():.2f}; frozen ok {sc.get('frozen_layers_equal')}")
    elif case == "explain":
        net = load("L2", f"ppo_s{100 + k}")["tts"]
        for n in ("small", "full"):
            if f"fidelity_{n}" in sc:
                r = paired(load("L2", f"explain_net{k}_{n}")["tts"], net)
                out.append(f"{n}: {sc[f'comparisons_{n}']} comparisons, fidelity {100 * sc[f'fidelity_{n}']:.1f}%, "
                           f"TTS vs network {r['pct']:+.1f}%")
    elif case == "constrain":
        x = load("L2_rules", f"constrain_{pid}")
        r = paired(x["tts"], load("L2_rules", f"naive_net{k}")["tts"])
        out.append(f"vs naive shield {r['pct']:+.2f}% [{r['pct_lo']:+.2f}, {r['pct_hi']:+.2f}] wins {r['wins']}/30")
    out.append(f"turns {meta.get('num_turns')}, ${meta.get('total_cost_usd', 0):.2f}, denials {len(meta.get('permission_denials') or [])}")
    print(f"{case}/{pid}: " + " | ".join(out))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        brief(p)
