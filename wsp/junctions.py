# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""The paper's junctions: network geometry, demand and scenarios.

    L2    one signalised junction J0, four approaches (N, E, S, W), 300 m, 50 km/h. On each approach a two-lane
          road widens 40 m before the stop line (node centres; 22.4 m of lane after the junction area) into three
          lanes: 0 right + through, 1 through, 2 left-turn pocket (about 3 cars). Upstream lane 0 feeds lane 0,
          upstream lane 1 feeds lanes 1 and 2. Exits have two lanes.
          A left-turner that finds the pocket full waits in upstream lane 1 and blocks the through traffic behind it.
    L2W   L2 with a wider E-W road: 3 upstream lanes widen to 4 (r + s, s, s, l), 3-lane E-W exits, busier E-W demand.

Demand: origin-destination flows with per-second insertion probability, four 15-min intervals. A scenario is
    scale    multiplier on the base entry flows (main = E/W entries, other = N/S entries)
    bias     directional log-bias b: E/W entries x e^b, N/S entries x e^-b
    profile  4 multipliers, one per 15-min interval
Training family: scale U(0.7, 1.25), bias U(-0.2, 0.3), profile entries U(0.6, 1.25).
Held-out scenarios: low (x0.8), nominal (x1.0), high (x1.15); bias 0, profile 0.75 / 1.0 / 1.15 / 0.9.

The nets and the route files are deterministic; tests/test_repro.py checks them byte for byte against the
released nets and the route files used in the study.

    python -m wsp.junctions          # (from the repository root) build nets/<junction>/{net.net.xml, meta.json}
"""
import json
import math
import os
import subprocess
import xml.etree.ElementTree as ET

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(V3)
NET_DIR = os.path.join(V3, "nets")
APPROACHES = ["N", "E", "S", "W"]
VEC = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
PHASES = [("NS_through", {"N", "S"}, {"s", "r"}), ("NS_left", {"N", "S"}, {"l"}),
          ("EW_through", {"E", "W"}, {"s", "r"}), ("EW_left", {"E", "W"}, {"l"})]
DIRMAP = {"s": "s", "r": "r", "R": "r", "l": "l", "L": "l"}
SPEED = 13.89
NOMINAL_PROFILE = [0.75, 1.0, 1.15, 0.9]
SCENARIO_SCALE = {"low": 0.8, "nominal": 1.0, "high": 1.15}

JUNCTIONS = {
    "L2": dict(lanes=dict(N=3, E=3, S=3, W=3), up=dict(N=2, E=2, S=2, W=2), out=dict(N=2, E=2, S=2, W=2),
               length=300.0, pocket=40.0, base=(750, 450), turn=dict(s=5.5, l=3.0, r=1.5)),
    "L2W": dict(lanes=dict(N=3, E=4, S=3, W=4), up=dict(N=2, E=3, S=2, W=3), out=dict(N=2, E=3, S=2, W=3),
                length=300.0, pocket=40.0, base=(1050, 450), turn=dict(s=5.5, l=3.0, r=1.5)),
}


def base_of(problem):
    """'L2_rules' -> 'L2', 'L2_fault' -> 'L2'."""
    for suf in ("_rules", "_fault"):
        if problem.endswith(suf):
            return problem[:-len(suf)]
    return problem


def net_dir(j):
    return os.path.join(NET_DIR, base_of(j))


def net_path(j):
    return os.path.join(net_dir(j), "net.net.xml")


def load_meta(j):
    with open(os.path.join(net_dir(j), "meta.json")) as f:
        m = json.load(f)
    _augment(m, net_path(j))
    return m


# ---------------------------------------------------------------- scenarios
def scenario(j, name):
    return dict(scale=SCENARIO_SCALE[name], bias=0.0, profile=list(NOMINAL_PROFILE))


def sample_train(j, rng):
    return dict(scale=float(rng.uniform(0.7, 1.25)), bias=float(rng.uniform(-0.2, 0.3)),
                profile=[float(x) for x in rng.uniform(0.6, 1.25, 4)])


# ---------------------------------------------------------------- geometry
def _layout(j):
    g = JUNCTIONS[base_of(j)]
    nodes = {"J0": (0.0, 0.0, "traffic_light")}
    edges = []
    for a in APPROACHES:
        dx, dy = VEC[a]
        pl, L = g["pocket"], g["length"]
        nodes[f"U{a}"] = (dx * pl, dy * pl, "priority")
        nodes[f"B{a}"] = (dx * L, dy * L, "priority")
        edges += [(f"B{a}", f"U{a}", g["up"][a]), (f"U{a}", "J0", g["lanes"][a]),
                  ("J0", f"U{a}", g["out"][a]), (f"U{a}", f"B{a}", g["out"][a])]
    return nodes, edges


def _turn(nodes, frm, j, to):
    h = math.atan2(nodes[j][1] - nodes[frm][1], nodes[j][0] - nodes[frm][0])
    o = math.atan2(nodes[to][1] - nodes[j][1], nodes[to][0] - nodes[j][0])
    d = math.degrees((o - h + math.pi) % (2 * math.pi) - math.pi)
    if abs(d) < 30:
        return "s"
    if abs(d) > 150:
        return "t"
    return "l" if d > 0 else "r"


def _approach(nodes, frm, j):
    dx, dy = nodes[frm][0] - nodes[j][0], nodes[frm][1] - nodes[j][1]
    if abs(dy) >= abs(dx):
        return "N" if dy > 0 else "S"
    return "E" if dx > 0 else "W"


def _connections(nodes, edges):
    nl = {(e[0], e[1]): e[2] for e in edges}
    cons = []
    for j in nodes:
        if j.startswith("J"):
            ins = [a for a, b, _ in edges if b == j]
            outs = [b for a, b, _ in edges if a == j]
            for frm in ins:
                n_in = nl[(frm, j)]
                for to in outs:
                    t, n_out = _turn(nodes, frm, j, to), nl[(j, to)]
                    if t == "r":
                        cons.append((f"{frm}_{j}", f"{j}_{to}", 0, 0))
                    elif t == "s":
                        for k in range(n_in - 1):
                            cons.append((f"{frm}_{j}", f"{j}_{to}", k, min(k, n_out - 1)))
                    elif t == "l":
                        cons.append((f"{frm}_{j}", f"{j}_{to}", n_in - 1, n_out - 1))
        elif j[0] == "U":                             # pocket start: straight through, the leftmost lane widens
            for a, b, n in edges:
                if b != j:
                    continue
                to = next(y for x, y, _ in edges if x == j and y != a)
                n_out = nl[(j, to)]
                for k in range(n):
                    cons.append((f"{a}_{j}", f"{j}_{to}", k, min(k, n_out - 1)))
                for k in range(n, n_out):
                    cons.append((f"{a}_{j}", f"{j}_{to}", n - 1, k))
    return cons


def build(j):
    out = net_dir(j)
    os.makedirs(out, exist_ok=True)
    nodes, edges = _layout(j)
    with open(os.path.join(out, "nodes.nod.xml"), "w") as f:
        f.write("<nodes>\n")
        for nid, (x, y, typ) in nodes.items():
            f.write(f'    <node id="{nid}" x="{x}" y="{y}" type="{typ}"/>\n')
        f.write("</nodes>\n")
    with open(os.path.join(out, "edges.edg.xml"), "w") as f:
        f.write("<edges>\n")
        for a, b, n in edges:
            f.write(f'    <edge id="{a}_{b}" from="{a}" to="{b}" numLanes="{n}" speed="{SPEED}"/>\n')
        f.write("</edges>\n")
    with open(os.path.join(out, "cons.con.xml"), "w") as f:
        f.write("<connections>\n")
        for fe, te, fl, tl in _connections(nodes, edges):
            f.write(f'    <connection from="{fe}" to="{te}" fromLane="{fl}" toLane="{tl}"/>\n')
        f.write("</connections>\n")
    net = os.path.join(out, "net.net.xml")
    subprocess.run(["netconvert", "--node-files", os.path.join(out, "nodes.nod.xml"),
                    "--edge-files", os.path.join(out, "edges.edg.xml"),
                    "--connection-files", os.path.join(out, "cons.con.xml"),
                    "--no-turnarounds", "true", "--offset.disable-normalization", "true",
                    "--tls.default-type", "static", "--no-warnings", "true", "-o", net], check=True)
    meta = _meta(j, nodes, net)
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def _meta(j, nodes, net):
    root = ET.parse(net).getroot()
    lane_len = {ln.get("id"): float(ln.get("length")) for e in root.iter("edge")
                if e.get("function") != "internal" for ln in e.iter("lane")}
    links = []
    for c in root.iter("connection"):
        if c.get("tl") != "J0":
            continue
        frm = c.get("from")
        links.append({"index": int(c.get("linkIndex")), "in_lane": f"{frm}_{c.get('fromLane')}",
                      "out_lane": f"{c.get('to')}_{c.get('toLane')}", "dir": DIRMAP[c.get("dir")],
                      "approach": _approach(nodes, frm.split("_")[0], "J0")})
    links.sort(key=lambda d: d["index"])
    assert [d["index"] for d in links] == list(range(len(links)))
    app = {d["in_lane"]: d["approach"] for d in links}
    in_lanes = sorted(app, key=lambda ln: (APPROACHES.index(app[ln]), int(ln.rsplit("_", 1)[1])))
    green, yellow = [], []
    for _, apps, dirs in PHASES:
        on = [d["approach"] in apps and d["dir"] in dirs for d in links]
        green.append("".join("G" if x else "r" for x in on))
        yellow.append("".join("y" if x else "r" for x in on))
    m = {"problem": j, "phases": [p[0] for p in PHASES], "aux_tls": [],
         "tls": [{"id": "J0", "xy": [0.0, 0.0], "in_lanes": in_lanes, "lane_approach": [app[ln] for ln in in_lanes],
                  "lane_dirs": ["".join(sorted({d["dir"] for d in links if d["in_lane"] == ln})) for ln in in_lanes],
                  "lane_length": [lane_len[ln] for ln in in_lanes], "links": links, "green": green,
                  "yellow": yellow}]}
    _augment(m, net)
    return m


def _augment(m, net):
    """Exit lanes, upstream lanes (entry edges feeding a pocket edge) with their feeds, and a lane table."""
    root = ET.parse(net).getroot()
    lanes = {}
    for e in root.iter("edge"):
        if e.get("function") == "internal":
            continue
        for ln in e.iter("lane"):
            lanes[ln.get("id")] = {"length": float(ln.get("length")), "speed": float(ln.get("speed")),
                                   "edge": e.get("id")}
    xy = {jn.get("id"): (float(jn.get("x")), float(jn.get("y"))) for jn in root.iter("junction")}
    to_node = {e.get("id"): e.get("to") for e in root.iter("edge") if e.get("function") != "internal"}
    for t in m["tls"]:
        J = t["id"]

        def compass(ln):
            x, y = xy[to_node[ln.rsplit("_", 1)[0]]]
            dx, dy = x - xy[J][0], y - xy[J][1]
            return ("N" if dy > 0 else "S") if abs(dy) >= abs(dx) else ("E" if dx > 0 else "W")
        outs = sorted({d["out_lane"] for d in t["links"]},
                      key=lambda ln: (APPROACHES.index(compass(ln)), int(ln.rsplit("_", 1)[1])))
        t["out_lanes"] = outs
        t["out_length"] = [lanes[ln]["length"] for ln in outs]
        in_edges = sorted({ln.rsplit("_", 1)[0] for ln in t["in_lanes"]})
        feeds = {}
        for c in root.iter("connection"):
            fr, to = c.get("from"), c.get("to")
            if to in in_edges and fr.startswith("B") and not to_node[fr].startswith("J"):
                feeds.setdefault(f"{fr}_{c.get('fromLane')}", set()).add(f"{to}_{c.get('toLane')}")
        ups = sorted(feeds, key=lambda ln: (APPROACHES.index(ln[1]), int(ln.rsplit("_", 1)[1])))
        t["up_lanes"] = ups
        t["up_length"] = [lanes[ln]["length"] for ln in ups]
        t["feeds"] = {k: sorted(v) for k, v in feeds.items()}
    m["lanes"] = lanes


# ---------------------------------------------------------------- demand
def _ods(j):
    """-> list of (entry_edge, exit_edge, share, entry_approach, is_main)."""
    nodes, edges = _layout(j)
    turn = JUNCTIONS[base_of(j)]["turn"]
    out = []
    for a in APPROACHES:
        ent = next(f"{x}_{y}" for x, y, _ in edges if x == f"B{a}")
        first = ent.split("_")[1]
        w = {}
        for b in APPROACHES:
            if b == a:
                continue
            ext = next(f"{x}_{y}" for x, y, _ in edges if y == f"B{b}")
            last = ext.split("_")[0]
            t = _turn(nodes, first if first != "J0" else f"B{a}", "J0", last if last != "J0" else f"B{b}")
            w[ext] = turn[t]
        tot = sum(w.values())
        for ext, v in w.items():
            out.append((ent, ext, v / tot, a, a in "EW"))
    return out


def write_routes(j, sc, path):
    """Route file for scenario dict sc. Returns the expected number of vehicles.
    Flow ids are '<entry approach>_<k>', so a pending vehicle's id names its approach."""
    main, other = JUNCTIONS[base_of(j)]["base"]
    flows, total = [], 0.0
    for ent, ext, share, a, is_main in _ods(j):
        R = (main * math.exp(sc["bias"])) if is_main else (other * math.exp(-sc["bias"]))
        R *= sc["scale"] * share
        cuts = [0.0, 900.0, 1800.0, 2700.0, 3600.0]
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            rate = R * sc["profile"][min(int(t0 // 900), 3)]
            if rate <= 0:
                continue
            p = min(rate / 3600.0, 0.95)
            total += p * (t1 - t0)
            flows.append((t0, t1, ent, ext, p, a))
    flows.sort(key=lambda r: r[0])
    lines = ['<routes>',
             '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>']
    for i, (t0, t1, ent, ext, p, a) in enumerate(flows):
        lines.append(f'    <flow id="{a}_{i}" type="car" from="{ent}" to="{ext}" begin="{t0:.1f}" end="{t1:.1f}" '
                     f'probability="{p:.6f}" departLane="best" departSpeed="max"/>')
    lines.append('</routes>')
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return total


def pending_approach(j):
    """vehicle id -> entry approach letter."""
    return lambda vid: vid[0]


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["L2", "L2W"]):
        mt = build(name)
        t = mt["tls"][0]
        print(name, "stop-line lanes", len(t["in_lanes"]), "upstream lanes", len(t["up_lanes"]),
              "exit lanes", len(t["out_lanes"]))
