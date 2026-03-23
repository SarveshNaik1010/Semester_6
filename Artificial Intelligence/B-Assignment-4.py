"""
Robot Navigation — Best First Search with Grid + Search Tree Visualization
===========================================================================
Heuristic : Euclidean Distance to goal
pip install matplotlib
python bfs_robot.py
"""

import heapq, uuid, sys, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

# ── GRID ───────────────────────────────────────────────────────────
# '.' = free, '#' = obstacle, 'S' = start, 'G' = goal
GRID = [
    ['S', '.', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '#', '.', '.'],
    ['#', '.', '.', '.', '.'],
    ['.', '.', '#', '.', 'G'],
]

ROWS = len(GRID)
COLS = len(GRID[0])

def find_cell(ch):
    for r, row in enumerate(GRID):
        for c, cell in enumerate(row):
            if cell == ch:
                return (r, c)

START = find_cell('S')
GOAL  = find_cell('G')

MOVES = [(-1,0,"↑"),(1,0,"↓"),(0,-1,"←"),(0,1,"→")]

def euclidean(state):
    return math.sqrt((state[0]-GOAL[0])**2 + (state[1]-GOAL[1])**2)

def get_neighbors(state):
    r, c = state
    result = []
    for dr, dc, lbl in MOVES:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] != '#':
            result.append(((nr, nc), lbl))
    return result

# ── BEST FIRST SEARCH + TREE BUILD ─────────────────────────────────
nodes       = {}
visit_order = [0]
goal_id     = [None]
counter     = [0]

root_id = "root"
nodes[root_id] = {
    "state": START, "depth": 0, "parent": None,
    "label": "", "status": "start", "h": euclidean(START),
    "children": [], "visit_num": 0
}

open_heap = []
visited   = {}

def push(nid):
    counter[0] += 1
    heapq.heappush(open_heap, (nodes[nid]["h"], counter[0], nid))

push(root_id)
visited[START] = root_id

while open_heap and goal_id[0] is None:
    h_val, _, nid = heapq.heappop(open_heap)
    nd = nodes[nid]

    if nd["state"] in visited and visited[nd["state"]] != nid:
        nd["status"] = "pruned"
        continue
    if nd["status"] == "pruned":
        continue

    if nd["state"] == GOAL:
        nd["status"] = "goal"
        goal_id[0] = nid
        break

    for nxt_state, lbl in get_neighbors(nd["state"]):
        cid = str(uuid.uuid4())[:8]
        h   = euclidean(nxt_state)
        if nxt_state in visited:
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
    print("No path found."); sys.exit()

# Mark solution path
sol_ids = set()
nid = goal_id[0]
while nid:
    sol_ids.add(nid)
    nid = nodes[nid]["parent"]
for nid in sol_ids:
    if nodes[nid]["status"] not in ("start","goal"):
        nodes[nid]["status"] = "solution"

sol_path  = []
nid = goal_id[0]
while nid:
    sol_path.append(nodes[nid]["state"])
    nid = nodes[nid]["parent"]
sol_path.reverse()

goal_depth   = nodes[goal_id[0]]["depth"]
total_nodes  = len(nodes)
pruned_count = sum(1 for n in nodes.values() if n["status"] == "pruned")

print(f"\n  Robot Navigation — Best First Search")
print(f"  Path: {' → '.join(str(p) for p in sol_path)}")
print(f"  Steps: {goal_depth}  |  Nodes: {total_nodes}  |  Pruned: {pruned_count}")

# ── TREE LAYOUT ─────────────────────────────────────────────────────
sys.setrecursionlimit(50000)

def assign_x(nid, ctr):
    ch = nodes[nid]["children"]
    if not ch:
        nodes[nid]["x"] = ctr[0] * 1.0
        ctr[0] += 1
        return
    for c in ch:
        assign_x(c, ctr)
    xs = [nodes[c]["x"] for c in ch]
    nodes[nid]["x"] = (min(xs)+max(xs))/2.0

assign_x(root_id, [0])
for nid, nd in nodes.items():
    nd["y"] = -nd["depth"] * 2.0

all_x = [nd["x"] for nd in nodes.values()]
x_span = max(all_x)-min(all_x)+1
scale  = max(1.6, 10.0/x_span)
for nd in nodes.values():
    nd["x"] *= scale

# ── COLORS ──────────────────────────────────────────────────────────
C_BG       = "#0a0a1a"
C_START    = "#FF8C00"
C_SOL      = "#FFD700"
C_GOAL     = "#00E676"
C_NEW      = "#4A6FA5"
C_PRUNED   = "#B71C1C"
C_SOLEDGE  = "#FFD700"
C_NEWEDGE  = "#4A6FA5"
C_PRUNEDGE = "#B71C1C"
C_BT       = "#FF4081"

def node_color(s):
    return {"start":C_START,"solution":C_SOL,"goal":C_GOAL,
            "new":C_NEW,"pruned":C_PRUNED}.get(s, C_NEW)

# ── FIGURE: left=grid, right=tree ───────────────────────────────────
all_x2 = [nd["x"] for nd in nodes.values()]
all_y2 = [nd["y"] for nd in nodes.values()]
tree_W = max(14, (max(all_x2)-min(all_x2)+4))
tree_H = max(14, abs(min(all_y2))+4)

fig = plt.figure(figsize=(tree_W+10, max(tree_H, 10)))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, tree_W/8], figure=fig, wspace=0.04)

ax_grid = fig.add_subplot(gs[0])
ax_tree = fig.add_subplot(gs[1])
ax_grid.set_facecolor(C_BG)
ax_tree.set_facecolor(C_BG)

# ── LEFT: GRID VISUALIZATION ────────────────────────────────────────
sol_set = set(sol_path)

for r in range(ROWS):
    for c in range(COLS):
        cell = GRID[r][c]
        pos  = (r, c)
        if cell == '#':
            fc, ec, txt_col = "#333355", "#555577", "#888899"
            lbl = "▪"
        elif pos == START:
            fc, ec, txt_col = C_START, "#FFAA00", "#000"
            lbl = "S"
        elif pos == GOAL:
            fc, ec, txt_col = C_GOAL, "#00FF88", "#000"
            lbl = "G"
        elif pos in sol_set:
            fc, ec, txt_col = "#2a2a5a", C_SOL, C_SOL
            lbl = "★"
        elif pos in visited:
            fc, ec, txt_col = "#1a2a3a", C_NEW, C_NEW
            lbl = "·"
        else:
            fc, ec, txt_col = "#111128", "#333355", "#444466"
            lbl = "."

        rect = plt.Rectangle((c-0.5, ROWS-1-r-0.5), 1, 1,
                              fc=fc, ec=ec, lw=1.5, zorder=1)
        ax_grid.add_patch(rect)
        ax_grid.text(c, ROWS-1-r, lbl, ha="center", va="center",
                     fontsize=14, color=txt_col, fontweight="bold", zorder=3)

        # Show grid coordinates
        ax_grid.text(c, ROWS-1-r-0.35, f"({r},{c})", ha="center", va="center",
                     fontsize=6.5, color="#666688", zorder=3)

        # Show h(n) on visited cells
        if pos in visited and pos not in (START, GOAL) and cell != '#':
            h = round(euclidean(pos), 2)
            ax_grid.text(c, ROWS-1-r+0.35, f"h={h}", ha="center", va="center",
                         fontsize=6, color="#AAAACC", zorder=3)

# Draw solution path arrows on grid
for i in range(len(sol_path)-1):
    r0,c0 = sol_path[i]
    r1,c1 = sol_path[i+1]
    ax_grid.annotate("",
        xy=(c1, ROWS-1-r1), xytext=(c0, ROWS-1-r0),
        arrowprops=dict(arrowstyle="-|>", color=C_SOL,
                        lw=2.5, mutation_scale=20,
                        connectionstyle="arc3,rad=0.0"), zorder=4)

ax_grid.set_xlim(-0.5, COLS-0.5)
ax_grid.set_ylim(-0.5, ROWS-0.5)
ax_grid.set_aspect('equal')
ax_grid.set_title("Grid Map  │  ★ = Solution Path",
                  color="white", fontsize=11, fontweight="bold", pad=10)
ax_grid.axis("off")

# Grid legend
lg = [
    mpatches.Patch(color=C_START,  label="S = Start"),
    mpatches.Patch(color=C_GOAL,   label="G = Goal"),
    mpatches.Patch(color=C_SOL,    label="★ = Solution path"),
    mpatches.Patch(color=C_NEW,    label="· = Explored cell"),
    mpatches.Patch(color="#333355",label="▪ = Obstacle"),
]
ax_grid.legend(handles=lg, loc="lower left",
               facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="white", fontsize=8.5,
               framealpha=0.97, borderpad=0.8)

# ── RIGHT: SEARCH TREE ───────────────────────────────────────────────
NODE_R = 0.45

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
    ax_tree.annotate("", xy=(x1, y1+NODE_R), xytext=(x0, y0-NODE_R),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                        linestyle=ls, mutation_scale=14,
                        connectionstyle="arc3,rad=0.0"), zorder=2)
    ex = (x0+x1)/2 + 0.1
    ey = (y0+y1)/2
    ax_tree.text(ex, ey, nd["label"], fontsize=7, color=col,
                 ha="center", va="center", zorder=4,
                 bbox=dict(boxstyle="round,pad=0.15", fc=C_BG, ec=col, lw=0.6, alpha=0.9))

for nid, nd in nodes.items():
    if nd["status"] != "pruned": continue
    par = nodes[nd["parent"]]
    ax_tree.annotate("",
        xy=(par["x"], par["y"]-NODE_R),
        xytext=(nd["x"], nd["y"]+NODE_R),
        arrowprops=dict(arrowstyle="-|>", color=C_BT, lw=1.6,
                        linestyle=(0,(5,3)), mutation_scale=13,
                        alpha=0.85, connectionstyle="arc3,rad=0.5"), zorder=5)

for nid, nd in nodes.items():
    x, y   = nd["x"], nd["y"]
    status = nd["status"]
    col    = node_color(status)
    sz     = 2000 if status in ("start","goal") else \
             1600 if status == "solution" else \
             1100 if status == "new" else 850

    if status in ("start","solution","goal"):
        ax_tree.scatter(x, y, s=sz+800, c=col, alpha=0.12, zorder=3)
    ax_tree.scatter(x, y, s=sz, c=col, zorder=6, edgecolors="#ffffff20", linewidths=0.9)

    tc = "#000000" if status in ("start","solution","goal") else "#FFFFFF"
    ax_tree.text(x, y+0.18, str(nd["state"]), ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color=tc, zorder=7)
    ax_tree.text(x, y-0.25, f"h={nd['h']:.2f}", ha="center", va="center",
                 fontsize=6.5, color="#AAAACC", zorder=7)

    vnum = nd.get("visit_num")
    if vnum is not None:
        vc = C_GOAL if status=="goal" else C_SOL if status=="solution" \
             else C_START if status=="start" else "#90CAF9"
        ax_tree.text(x-0.6, y+0.62, f"#{vnum}", fontsize=9, fontweight="bold",
                     color=vc, ha="center", va="center", zorder=8,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground=C_BG)])

    if status == "pruned":
        ax_tree.text(x+0.55, y+0.55, "✖", fontsize=11, color=C_PRUNED,
                     ha="center", va="center", fontweight="bold", zorder=8,
                     path_effects=[pe.withStroke(linewidth=2, foreground=C_BG)])

for nid, txt in [(root_id,"START"),(goal_id[0],"GOAL ★")]:
    nd  = nodes[nid]
    bg  = C_START if txt=="START" else C_GOAL
    ax_tree.text(nd["x"], nd["y"]+0.82, txt, ha="center", va="center",
                 fontsize=9, color="#000", fontweight="bold", zorder=9,
                 bbox=dict(boxstyle="round,pad=0.35", fc=bg, ec=bg, lw=1.8))

legend_handles = [
    mpatches.Patch(color=C_START,   label="Start node"),
    mpatches.Patch(color=C_SOL,     label="Solution path"),
    mpatches.Patch(color=C_GOAL,    label="Goal node"),
    mpatches.Patch(color=C_NEW,     label="Explored node"),
    mpatches.Patch(color=C_PRUNED,  label="✖  Already visited"),
    mpatches.Patch(color=C_BT,      label="↩  Skip (already visited)"),
]
ax_tree.legend(handles=legend_handles, loc="lower left",
               facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="white", fontsize=8.5,
               framealpha=0.97, borderpad=0.8, handlelength=1.4)

ax_tree.set_title("Search Tree  │  h(n) = Euclidean Distance",
                  color="white", fontsize=11, fontweight="bold", pad=10)
ax_tree.axis("off")

# ── MAIN TITLE ───────────────────────────────────────────────────────
fig.suptitle(
    f"Best First Search  │  Robot Navigation Problem\n"
    f"Grid {ROWS}×{COLS}  │  Start={START}  →  Goal={GOAL}  │  "
    f"Path length={goal_depth} steps  │  {total_nodes} tree nodes  │  {pruned_count} pruned",
    color="white", fontsize=13, fontweight="bold", y=1.01, linespacing=1.7)

plt.savefig("bfs_robot.png",
            dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("  Saved → bfs_robot.png")
plt.show()