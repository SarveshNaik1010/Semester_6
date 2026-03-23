"""
Water Jug Problem — DFS with Backtracking Tree
================================================
pip install matplotlib
python dfs_water_jug.py
"""

import sys, uuid, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

# ── INPUT ──────────────────────────────────────────────
print("=" * 42)
print("   WATER JUG — DFS Solver")
print("=" * 42)
cap_a      = int(input("  Jug A capacity : "))
cap_b      = int(input("  Jug B capacity : "))
target     = int(input("  Target amount  : "))
jug_choice = input("  Target in Jug? (A/B): ").strip().upper()
while jug_choice not in ("A","B"):
    jug_choice = input("  Enter A or B: ").strip().upper()

def is_goal(s):
    return s[0] == target if jug_choice == "A" else s[1] == target

# ── SUCCESSORS ─────────────────────────────────────────
def successors(state):
    a, b = state
    moves = []
    if a < cap_a: moves.append(((cap_a, b), "Fill A"))
    if b < cap_b: moves.append(((a, cap_b), "Fill B"))
    if a > 0:     moves.append(((0, b),     "Empty A"))
    if b > 0:     moves.append(((a, 0),     "Empty B"))
    if a > 0 and b < cap_b:
        p = min(a, cap_b-b); moves.append(((a-p, b+p), "A→B"))
    if b > 0 and a < cap_a:
        p = min(b, cap_a-a); moves.append(((a+p, b-p), "B→A"))
    return [(s, l) for s, l in moves if s != state]

# ── BUILD TREE ─────────────────────────────────────────
# This builds a proper TREE (not a graph) where:
#   - Each node is a UNIQUE tree position (state can repeat = pruned node)
#   - "pruned" nodes = state already visited on this path = DFS BACKTRACKS here
#   - Tree stops as soon as goal is first found (as real DFS would)
#
# This is the standard textbook representation of a DFS search tree.

sys.setrecursionlimit(50000)

nodes   = {}  # node_id -> dict
root_id = "root"
nodes[root_id] = {
    "state": (0, 0), "depth": 0, "parent": None,
    "label": "", "status": "start", "children": []
}
goal_id = [None]
stop    = [False]
visit_counter = [0]

def build_tree(node_id, visited, depth=0):
    if stop[0]:
        return
    state = nodes[node_id]["state"]
    visited = visited | {state}   # copy — each path has its own visited set

    for nxt, lbl in successors(state):
        if stop[0]:
            return
        cid = str(uuid.uuid4())[:8]

        if nxt in visited:
            # Already on this path → dead end / pruned → DFS backtracks here
            nodes[cid] = {
                "state": nxt, "depth": depth+1, "parent": node_id,
                "label": lbl, "status": "pruned", "children": [],
                "visit_num": None
            }
            nodes[node_id]["children"].append(cid)
        else:
            visit_counter[0] += 1
            status = "goal" if is_goal(nxt) else "new"
            nodes[cid] = {
                "state": nxt, "depth": depth+1, "parent": node_id,
                "label": lbl, "status": status, "children": [],
                "visit_num": visit_counter[0]
            }
            nodes[node_id]["children"].append(cid)
            if status == "goal":
                goal_id[0] = cid
                stop[0]    = True
                return
            build_tree(cid, visited, depth+1)

nodes[root_id]["visit_num"] = 0
build_tree(root_id, set(), 0)

if goal_id[0] is None:
    print("No solution found."); sys.exit()

# Mark the solution path from goal back to root
sol_ids = set()
nid = goal_id[0]
while nid is not None:
    sol_ids.add(nid)
    nid = nodes[nid]["parent"]

for nid in sol_ids:
    if nodes[nid]["status"] not in ("start", "goal"):
        nodes[nid]["status"] = "solution"

goal_depth = nodes[goal_id[0]]["depth"]
total_nodes = len(nodes)
pruned_count = sum(1 for n in nodes.values() if n["status"] == "pruned")

print(f"\n  Goal found at depth : {goal_depth}")
print(f"  Tree nodes total    : {total_nodes}")
print(f"  Pruned (backtracks) : {pruned_count}")

# ── LAYOUT ─────────────────────────────────────────────
# Post-order: leaves get sequential x, parents centred above children

def assign_x(nid, counter):
    ch = nodes[nid]["children"]
    if not ch:
        nodes[nid]["x"] = counter[0] * 1.0
        counter[0] += 1
        return
    for c in ch:
        assign_x(c, counter)
    xs = [nodes[c]["x"] for c in ch]
    nodes[nid]["x"] = (min(xs) + max(xs)) / 2.0

assign_x(root_id, [0])

# y = -depth
for nid, nd in nodes.items():
    nd["y"] = -nd["depth"] * 1.8

# Scale x a bit for readability
all_x = [nd["x"] for nd in nodes.values()]
x_span = max(all_x) - min(all_x) + 1
for nd in nodes.values():
    nd["x"] = nd["x"] * max(1.5, 12.0 / x_span)

# ── DRAW ───────────────────────────────────────────────
all_x2 = [nd["x"] for nd in nodes.values()]
all_y2 = [nd["y"] for nd in nodes.values()]
W = max(18, (max(all_x2) - min(all_x2) + 4))
H = max(14, abs(min(all_y2)) + 4)

fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor("#0f0f1a")
ax.set_facecolor("#0f0f1a")

# Colours
C_START  = "#FF8C00"   # orange
C_SOL    = "#FFD700"   # gold  — solution path
C_GOAL   = "#00E676"   # green
C_NEW    = "#4A6FA5"   # blue  — explored (not on solution path)
C_PRUNED = "#B71C1C"   # dark red — pruned/backtracked
C_SOLEDGE  = "#FFD700"
C_NEWEDGE  = "#4A6FA5"
C_PRUNEDGE = "#B71C1C"
C_BT_ARROW = "#FF4081"   # bright pink — backtrack arrows

NODE_R = 0.45   # node radius for positioning arrow endpoints

def node_color(status):
    return {
        "start":    C_START,
        "solution": C_SOL,
        "goal":     C_GOAL,
        "new":      C_NEW,
        "pruned":   C_PRUNED,
    }.get(status, C_NEW)

# ── 1. Draw edges ───────────────────────────────────────
for nid, nd in nodes.items():
    if nd["parent"] is None:
        continue
    par = nodes[nd["parent"]]
    x0, y0 = par["x"], par["y"]
    x1, y1 = nd["x"],  nd["y"]

    is_sol    = (nid in sol_ids and nd["parent"] in sol_ids)
    is_pruned = nd["status"] == "pruned"

    color = C_SOLEDGE if is_sol else (C_PRUNEDGE if is_pruned else C_NEWEDGE)
    lw    = 2.5 if is_sol else 1.2
    ls    = "solid" if (is_sol or not is_pruned) else (0, (6, 3))

    ax.annotate("",
        xy=(x1, y1 + NODE_R), xytext=(x0, y0 - NODE_R),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=lw,
            linestyle=ls, mutation_scale=14,
            connectionstyle="arc3,rad=0.0"),
        zorder=2)

    # Operation label on edge
    ex = (x0 + x1) / 2 + 0.08
    ey = (y0 + y1) / 2
    ax.text(ex, ey, nd["label"], fontsize=7, color=color,
            ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15",
                      fc="#0f0f1a", ec=color, lw=0.6, alpha=0.9))

# ── 2. Backtrack arrows ─────────────────────────────────
# For every pruned node: draw a curved red arrow from that pruned
# node back UP to its parent to show "DFS came here, can't go further,
# so it backtracks to parent and tries the next sibling"
#
# The arrow is drawn as a curve that goes LEFT of the edge so it
# doesn't overlap with the downward tree edge.

for nid, nd in nodes.items():
    if nd["status"] != "pruned":
        continue
    par = nodes[nd["parent"]]
    x0, y0 = nd["x"],   nd["y"]   # FROM pruned child
    x1, y1 = par["x"],  par["y"]  # TO   parent

    ax.annotate("",
        xy=(x1, y1 - NODE_R),      # arrive at parent (bottom)
        xytext=(x0, y0 + NODE_R),  # depart pruned node (top)
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_BT_ARROW,
            lw=1.8,
            linestyle=(0, (5, 3)),
            mutation_scale=13,
            alpha=0.9,
            connectionstyle="arc3,rad=0.5"),
        zorder=5)

# ── 3. Draw nodes ───────────────────────────────────────
for nid, nd in nodes.items():
    x, y   = nd["x"], nd["y"]
    status = nd["status"]
    col    = node_color(status)
    sz     = (1800 if status in ("goal", "start") else
              1500 if status == "solution" else
              1100 if status == "new" else 900)

    # Glow for important nodes
    if status in ("start", "solution", "goal"):
        ax.scatter(x, y, s=sz+700, c=col, alpha=0.15, zorder=3)

    ax.scatter(x, y, s=sz, c=col, zorder=6,
               edgecolors="#ffffff25", linewidths=0.9)

    # State label
    tc = "#000000" if status in ("start","solution","goal") else "#FFFFFF"
    ax.text(x, y, f"({nd['state'][0]},{nd['state'][1]})",
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=tc, zorder=7)

    # Visit-order number for non-pruned nodes
    vnum = nd.get("visit_num")
    if vnum is not None:
        vc = (C_GOAL  if status == "goal"     else
              C_SOL   if status == "solution"  else
              C_START if status == "start"     else
              "#90CAF9")
        ax.text(x - 0.58, y + 0.58, f"#{vnum}",
                fontsize=9.5, fontweight="bold", color=vc,
                ha="center", va="center", zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5,
                                             foreground="#0f0f1a")])

    # ✖ on pruned nodes
    if status == "pruned":
        ax.text(x + 0.52, y + 0.52, "✖",
                fontsize=11, color=C_PRUNED,
                ha="center", va="center", fontweight="bold", zorder=8,
                path_effects=[pe.withStroke(linewidth=2,
                                             foreground="#0f0f1a")])

# START / GOAL top badges
for nid, txt in [(root_id, "START"), (goal_id[0], "GOAL ★")]:
    nd  = nodes[nid]
    bg  = C_START if txt == "START" else C_GOAL
    ax.text(nd["x"], nd["y"] + 0.75, txt,
            ha="center", va="center",
            fontsize=9, color="#000000", fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.35",
                      fc=bg, ec=bg, lw=1.8))

# ── LEGEND ─────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_START,   label="Start node  (0,0)"),
    mpatches.Patch(color=C_SOL,     label="Solution path  (DFS found this route)"),
    mpatches.Patch(color=C_GOAL,    label=f"Goal  —  {target}L in Jug {jug_choice}"),
    mpatches.Patch(color=C_NEW,     label="Explored node  (off solution path)"),
    mpatches.Patch(color=C_PRUNED,  label="✖  Already visited  →  DFS must backtrack"),
    mpatches.Patch(color=C_SOLEDGE, label="━━  Solution path edge"),
    mpatches.Patch(color=C_NEWEDGE, label="──  DFS forward edge"),
    mpatches.Patch(color=C_BT_ARROW,label="↩   Backtrack  (DFS retreats to parent)"),
]
ax.legend(handles=legend_handles, loc="lower left",
          facecolor="#1a1a2e", edgecolor="#444466",
          labelcolor="white", fontsize=9.5,
          framealpha=0.97, borderpad=1.0, handlelength=1.6)

# ── TITLE ──────────────────────────────────────────────
ax.set_title(
    f"DFS — Depth First Search  │  Water Jug Problem\n"
    f"Jug A = {cap_a}L    Jug B = {cap_b}L    Target = {target}L in Jug {jug_choice}\n"
    f"DFS dives deep into each branch first  •  "
    f"When it hits ✖ (already visited state), it backtracks ↩ to parent and tries the next child  •  "
    f"Solution found at depth {goal_depth}  •  {pruned_count} backtracks",
    color="white", fontsize=12, fontweight="bold",
    loc="left", pad=16, linespacing=1.75)

ax.axis("off")
plt.savefig("dfs_tree.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  Saved → dfs_tree.png")
plt.show()