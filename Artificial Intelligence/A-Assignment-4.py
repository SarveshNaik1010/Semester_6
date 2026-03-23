"""
8-Puzzle — Best First Search with Full Tree Visualization
==========================================================
Heuristic : Manhattan Distance
pip install matplotlib
python bfs_8puzzle.py
"""

import heapq, uuid, sys, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── CONFIGURATION ──────────────────────────────────────────────────
GOAL_STATE  = (1, 2, 3, 8, 0, 4, 7, 6, 5)
GOAL_POS    = {val: idx for idx, val in enumerate(GOAL_STATE)}

# Initial state  →  change here to test others
# (2,8,3, 1,6,4, 7,0,5) is a classic 5-move puzzle
INIT_STATE  = (2, 8, 3,
               1, 6, 4,
               7, 0, 5)

# ── HEURISTIC ──────────────────────────────────────────────────────
def manhattan(state):
    d = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        gi = GOAL_POS[tile]
        d += abs(idx//3 - gi//3) + abs(idx%3 - gi%3)
    return d

# ── SUCCESSORS ─────────────────────────────────────────────────────
MOVE_NAMES = {(-1,0):"↑ Up", (1,0):"↓ Down", (0,-1):"← Left", (0,1):"→ Right"}

def successors(state):
    blank = state.index(0)
    r, c  = divmod(blank, 3)
    result = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            nb = nr*3 + nc
            s  = list(state)
            s[blank], s[nb] = s[nb], s[blank]
            result.append((tuple(s), MOVE_NAMES[(dr,dc)]))
    return result

# ── BEST FIRST SEARCH + TREE BUILD ─────────────────────────────────
# nodes dict: id -> {state, depth, parent, label, status, h, children}

nodes        = {}
visit_order  = [0]
goal_id      = [None]

root_id = "root"
nodes[root_id] = {
    "state": INIT_STATE, "depth": 0, "parent": None,
    "label": "", "status": "start", "h": manhattan(INIT_STATE),
    "children": [], "visit_num": 0
}

# BFS-style: expand best h(n) first, record ALL nodes explored
# We also add "rejected" (already visited) siblings to the tree for visualization

open_heap  = []           # (h, counter, node_id)
visited    = {}           # state -> node_id  (first time we expand this state)
counter    = [0]

def push(nid):
    counter[0] += 1
    heapq.heappush(open_heap, (nodes[nid]["h"], counter[0], nid))

push(root_id)
visited[INIT_STATE] = root_id

while open_heap and goal_id[0] is None:
    h_val, _, nid = heapq.heappop(open_heap)
    nd = nodes[nid]

    if nd["status"] == "pruned":
        continue                # duplicate in heap (lazy deletion)
    if nd["state"] in visited and visited[nd["state"]] != nid:
        nd["status"] = "pruned"
        continue

    if nd["status"] not in ("start", "solution", "goal"):
        visit_order[0] += 1
        nd["visit_num"] = visit_order[0]

    if manhattan(nd["state"]) == 0:          # goal check
        nd["status"] = "goal"
        goal_id[0]   = nid
        break

    for nxt_state, lbl in successors(nd["state"]):
        cid = str(uuid.uuid4())[:8]
        h   = manhattan(nxt_state)
        if nxt_state in visited:
            # Already visited — show as pruned leaf in tree
            nodes[cid] = {
                "state": nxt_state, "depth": nd["depth"]+1,
                "parent": nid, "label": lbl, "status": "pruned",
                "h": h, "children": [], "visit_num": None
            }
            nd["children"].append(cid)
        else:
            visit_order[0] += 1
            nodes[cid] = {
                "state": nxt_state, "depth": nd["depth"]+1,
                "parent": nid, "label": lbl, "status": "new",
                "h": h, "children": [], "visit_num": visit_order[0]
            }
            nd["children"].append(cid)
            visited[nxt_state] = cid
            push(cid)

if goal_id[0] is None:
    print("No solution found."); sys.exit()

# Mark solution path
sol_ids = set()
nid = goal_id[0]
while nid:
    sol_ids.add(nid)
    nid = nodes[nid]["parent"]
for nid in sol_ids:
    if nodes[nid]["status"] not in ("start","goal"):
        nodes[nid]["status"] = "solution"

goal_depth   = nodes[goal_id[0]]["depth"]
total_nodes  = len(nodes)
pruned_count = sum(1 for n in nodes.values() if n["status"] == "pruned")

# Print solution path
print("\n  8-PUZZLE — Best First Search Solution")
print("  " + "─"*45)
path_ids = []
nid = goal_id[0]
while nid:
    path_ids.append(nid)
    nid = nodes[nid]["parent"]
path_ids.reverse()
for step, nid in enumerate(path_ids):
    s = nodes[nid]["state"]
    h = nodes[nid]["h"]
    lbl = nodes[nid]["label"]
    print(f"  Step {step}  h={h}  {lbl}")
    for row in range(3):
        print("    " + " ".join(str(s[row*3+c]) if s[row*3+c] else "_" for c in range(3)))
print(f"\n  Solved in {goal_depth} moves  |  Nodes in tree: {total_nodes}  |  Pruned: {pruned_count}")

# ── LAYOUT ──────────────────────────────────────────────────────────
sys.setrecursionlimit(50000)

def assign_x(nid, counter):
    ch = nodes[nid]["children"]
    if not ch:
        nodes[nid]["x"] = counter[0] * 1.0
        counter[0] += 1
        return
    for c in ch:
        assign_x(c, counter)
    xs = [nodes[c]["x"] for c in ch]
    nodes[nid]["x"] = (min(xs)+max(xs))/2.0

assign_x(root_id, [0])
for nid, nd in nodes.items():
    nd["y"] = -nd["depth"] * 2.2

all_x = [nd["x"] for nd in nodes.values()]
x_span = max(all_x) - min(all_x) + 1
scale  = max(1.8, 14.0 / x_span)
for nd in nodes.values():
    nd["x"] *= scale

# ── COLORS ──────────────────────────────────────────────────────────
C_BG      = "#0a0a1a"
C_START   = "#FF8C00"
C_SOL     = "#FFD700"
C_GOAL    = "#00E676"
C_NEW     = "#4A6FA5"
C_PRUNED  = "#B71C1C"
C_SOLEDGE = "#FFD700"
C_NEWEDGE = "#4A6FA5"
C_PRUNEDGE= "#B71C1C"
C_BT      = "#FF4081"

def node_color(s):
    return {"start":C_START,"solution":C_SOL,"goal":C_GOAL,
            "new":C_NEW,"pruned":C_PRUNED}.get(s, C_NEW)

# ── FIGURE ──────────────────────────────────────────────────────────
all_x2 = [nd["x"] for nd in nodes.values()]
all_y2 = [nd["y"] for nd in nodes.values()]
W = max(20, (max(all_x2)-min(all_x2)+5))
H = max(16, abs(min(all_y2))+5)

fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

NODE_R = 0.48

# ── DRAW EDGES ──────────────────────────────────────────────────────
for nid, nd in nodes.items():
    if nd["parent"] is None: continue
    par = nodes[nd["parent"]]
    x0,y0 = par["x"], par["y"]
    x1,y1 = nd["x"],  nd["y"]
    is_sol    = (nid in sol_ids and nd["parent"] in sol_ids)
    is_pruned = nd["status"] == "pruned"
    col = C_SOLEDGE if is_sol else (C_PRUNEDGE if is_pruned else C_NEWEDGE)
    lw  = 2.8 if is_sol else 1.2
    ls  = "solid" if (is_sol or not is_pruned) else (0,(6,3))
    ax.annotate("", xy=(x1, y1+NODE_R), xytext=(x0, y0-NODE_R),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                        linestyle=ls, mutation_scale=14,
                        connectionstyle="arc3,rad=0.0"), zorder=2)
    ex = (x0+x1)/2 + 0.1
    ey = (y0+y1)/2
    ax.text(ex, ey, nd["label"], fontsize=7, color=col,
            ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc=C_BG, ec=col, lw=0.6, alpha=0.9))

# ── BACKTRACK ARROWS ────────────────────────────────────────────────
for nid, nd in nodes.items():
    if nd["status"] != "pruned": continue
    par = nodes[nd["parent"]]
    ax.annotate("",
        xy=(par["x"], par["y"]-NODE_R),
        xytext=(nd["x"], nd["y"]+NODE_R),
        arrowprops=dict(arrowstyle="-|>", color=C_BT, lw=1.6,
                        linestyle=(0,(5,3)), mutation_scale=13,
                        alpha=0.85, connectionstyle="arc3,rad=0.5"), zorder=5)

# ── DRAW NODES ──────────────────────────────────────────────────────
for nid, nd in nodes.items():
    x, y   = nd["x"], nd["y"]
    status = nd["status"]
    col    = node_color(status)
    sz     = 2200 if status in ("start","goal") else \
             1800 if status == "solution" else \
             1200 if status == "new" else 900

    if status in ("start","solution","goal"):
        ax.scatter(x, y, s=sz+900, c=col, alpha=0.12, zorder=3)
    ax.scatter(x, y, s=sz, c=col, zorder=6, edgecolors="#ffffff20", linewidths=0.9)

    # State label: draw mini 3x3 grid INSIDE the node
    s   = nd["state"]
    tc  = "#000000" if status in ("start","solution","goal") else "#FFFFFF"
    # Compact representation: row0 / row1 / row2
    r0  = "".join(str(v) if v else "_" for v in s[0:3])
    r1  = "".join(str(v) if v else "_" for v in s[3:6])
    r2  = "".join(str(v) if v else "_" for v in s[6:9])
    ax.text(x, y+0.30, r0, ha="center", va="center", fontsize=6.5,
            fontweight="bold", color=tc, zorder=7, fontfamily="monospace")
    ax.text(x, y,      r1, ha="center", va="center", fontsize=6.5,
            fontweight="bold", color=tc, zorder=7, fontfamily="monospace")
    ax.text(x, y-0.30, r2, ha="center", va="center", fontsize=6.5,
            fontweight="bold", color=tc, zorder=7, fontfamily="monospace")

    # h(n) label below node
    ax.text(x, y-0.68, f"h={nd['h']}", ha="center", va="center",
            fontsize=7, color="#AAAACC", zorder=7)

    # Visit order badge
    vnum = nd.get("visit_num")
    if vnum is not None:
        vc = C_GOAL if status=="goal" else C_SOL if status=="solution" \
             else C_START if status=="start" else "#90CAF9"
        ax.text(x-0.62, y+0.62, f"#{vnum}", fontsize=9, fontweight="bold",
                color=vc, ha="center", va="center", zorder=8,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=C_BG)])

    # ✖ on pruned
    if status == "pruned":
        ax.text(x+0.55, y+0.55, "✖", fontsize=11, color=C_PRUNED,
                ha="center", va="center", fontweight="bold", zorder=8,
                path_effects=[pe.withStroke(linewidth=2, foreground=C_BG)])

# START / GOAL badges
for nid, txt in [(root_id,"START"),(goal_id[0],"GOAL ★")]:
    nd  = nodes[nid]
    bg  = C_START if txt=="START" else C_GOAL
    ax.text(nd["x"], nd["y"]+0.85, txt, ha="center", va="center",
            fontsize=9, color="#000", fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.35", fc=bg, ec=bg, lw=1.8))

# ── LEGEND ──────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_START,    label="Start node"),
    mpatches.Patch(color=C_SOL,      label="Solution path  (Best-h route)"),
    mpatches.Patch(color=C_GOAL,     label="Goal  (Manhattan h=0)"),
    mpatches.Patch(color=C_NEW,      label="Explored node  (off solution path)"),
    mpatches.Patch(color=C_PRUNED,   label="✖  Already visited  →  skipped"),
    mpatches.Patch(color=C_SOLEDGE,  label="━━  Solution path edge"),
    mpatches.Patch(color=C_NEWEDGE,  label="──  BFS forward edge"),
    mpatches.Patch(color=C_BT,       label="↩  Skip arrow  (already visited)"),
]
ax.legend(handles=legend_handles, loc="lower left",
          facecolor="#1a1a2e", edgecolor="#444466",
          labelcolor="white", fontsize=9.5,
          framealpha=0.97, borderpad=1.0, handlelength=1.6)

# ── TITLE ───────────────────────────────────────────────────────────
goal_state_str = "1 2 3 / 8 _ 4 / 7 6 5"
ax.set_title(
    f"Best First Search  │  8-Puzzle Problem\n"
    f"Heuristic = Manhattan Distance  h(n)  │  Goal: [{goal_state_str}]\n"
    f"BFS always expands the node with lowest h(n)  •  "
    f"Solution found at depth {goal_depth}  •  {total_nodes} tree nodes  •  {pruned_count} pruned",
    color="white", fontsize=12, fontweight="bold",
    loc="left", pad=16, linespacing=1.75)

ax.axis("off")
plt.tight_layout()
plt.savefig("bfs_8puzzle.png",
            dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("  Saved → bfs_8puzzle.png")
plt.show()