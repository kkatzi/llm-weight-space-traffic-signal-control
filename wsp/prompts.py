# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""The launch prompts of the agent runs: one template, task bullets and summary lists.

    (from the repository root)  python -m wsp.prompts --pack packs/patch/net0_b10      # the exact prompt for one pack
                python -m wsp.prompts --all                                            # all launch prompts

Arms of the same task differ only in the simulator bullet and the summary items about simulator calls.
"""
import argparse
import json
import os

from wsp.junctions import V3

TEMPLATE = """You are taking part in a research experiment on {topic}.

Your task pack is the folder {pack}

Start by reading TASK.md in that folder, then follow it exactly. In particular:
- Read only files inside the pack.
- Write only to its work/ subfolder.
- {sim}
- {nofit}
- {task}
- Use the Python interpreter that TASK.md names.

When you finish, make sure {files} exist. Then reply with a short summary (under 200 words) covering:
{summary}"""

TOPIC = {"init": "programming neural-network weights by hand", "patch": "programming neural-network weights by hand",
         "patch_fault": "programming neural-network weights by hand",
         "transfer": "programming neural-network weights by hand", "explain": "reading neural-network weights",
         "constrain": "adapting trained neural-network controllers"}
NOFIT = {"init": "Do not fit weights to data with any optimiser.", "patch": "Do not fit weights to data with any optimiser.",
         "patch_fault": "Do not fit weights to data with any optimiser.",
         "transfer": "Do not fit weights to data with any optimiser.",
         "explain": "Do not fit any model and do not run automated searches over rules or thresholds.",
         "constrain": "Do not fit any model and do not run automated searches."}
FILES = {"init": "work/weights.npz, work/NOTES.md and work/predictions.json",
         "patch_fault": "work/weights.npz, work/NOTES.md and work/predictions.json",
         "patch": "work/weights.npz, work/NOTES.md and work/predictions.json",
         "transfer": "work/weights.npz, work/NOTES.md and work/predictions.json",
         "explain": "work/rules_small.py, work/rules_full.py, work/EXPLANATION.md, work/NOTES.md and work/predictions.json",
         "constrain": "work/supervisor.py, work/NOTES.md and work/predictions.json"}


def task_bullet(case, spec):
    if case in ("patch", "patch_fault"):
        total = 64 * spec["obs_dim"] + 64 + 64 * 64 + 64 + 4 * 64 + 4
        return f"Change at most {int(0.05 * total)} parameters."
    return {"init": "Meet the trainability requirements.",
            "transfer": "Keep W2, b2, W3 and b3 exactly equal to the source network's.",
            "explain": "Keep rules_small.py within 8 comparisons.",
            "constrain": "Use the network through its logits only."}[case]


def summary(case, loop):
    if case == "patch_fault":
        case = "patch"
    items = {"init": ["what you built", "its trainability numbers", "the checks you ran", "your test predictions"],
             "patch": (["what you changed and why", "how many parameters you changed",
                        "how many simulator calls you used and what they showed", "the dev TTS of your final weights",
                        "your test predictions"] if loop else
                       ["what you changed and why", "how many parameters you changed", "the checks you ran",
                        "your test predictions"]),
             "transfer": ["how you mapped L2W's inputs onto the L2 network and why", "the checks you ran",
                          "the trainability numbers", "your test predictions"],
             "explain": ["the policy in a few sentences",
                         "the fidelity and comparison count of both rule files on the recorded states", "your predictions"],
             "constrain": (["what your supervisor does", "when it overrides the network",
                            "your simulator calls and what they showed", "the dev TTS of your final supervisor",
                            "your predictions"] if loop else
                           ["what your supervisor does", "when it overrides the network", "the checks you ran",
                            "your predictions"])}[case]
    return "\n".join(f"- {x};" if i < len(items) - 1 else f"- {x}." for i, x in enumerate(items))


def render(pack):
    pack = os.path.realpath(pack)          # the pack's real location (outside the repository), never the packs link
    spec = json.load(open(os.path.join(pack, "spec.json")))
    case, b = spec["case"], spec["eval_budget"]
    sim = ("Do not run any simulator. You get one shot." if b == 0 else
           f"Use the simulator only through the pack's eval.py, within its budget of {b} calls. Run eval.py from "
           "inside the pack folder; each call takes about 10 s.")
    return TEMPLATE.format(topic=TOPIC[case], pack=pack, sim=sim, nofit=NOFIT[case], task=task_bullet(case, spec),
                           files=FILES[case], summary=summary(case, b > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.pack:
        print(render(a.pack))
        return
    seen = set()
    root = os.path.join(V3, "packs")
    print("The launch prompts, verbatim. {PACK} stands for the absolute path of the run's task folder.\n")
    for case in ("init", "patch", "patch_fault", "transfer", "explain", "constrain"):
        for pid in sorted(os.listdir(os.path.join(root, case))) if os.path.isdir(os.path.join(root, case)) else []:
            p = os.path.join(root, case, pid)
            txt = render(p).replace(os.path.realpath(p), "{PACK}")
            if txt in seen:
                continue
            seen.add(txt)
            b = json.load(open(os.path.join(p, "spec.json")))["eval_budget"]
            print(f"=== {case}, {'one shot' if b == 0 else f'{b} simulator calls'} ===\n{txt}\n")


if __name__ == "__main__":
    main()
