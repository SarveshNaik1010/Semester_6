"""
Cities Distance — Best First Search with Road Map + Search Tree Visualization
==============================================================================
Romania road map (classic AI textbook — Russell & Norvig)
Heuristic: Straight-line (Euclidean) distance to Bucharest
pip install matplotlib
python bfs_cities.py
"""

import heapq, uuid, sys, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

# ── GRAPH DATA ─────────────────────────────────────────────────────
GRAPH = {
    'Arad':       [('Zerind',75),('Sibiu',140),('Timisoara',118)],
    'Zerind':     [('Arad',75),('Oradea',71)],
    'Oradea':     [('Zerind',71),('Sibiu',151)],
    'Sibiu':      [('Arad',140),('Oradea',151),('Fagaras',99),('Rimnicu',80)],
    'Timisoara':  [('Arad',118),('Lugoj',111)],
    'Lugoj':      [('Timisoara',111),('Mehadia',70)],
    'Mehadia':    [('Lugoj',70),('Drobeta',75)],
    'Drobeta':    [('Mehadia',75),('Craiova',120)],
    'Craiova':    [('Drobeta',120),('Pitesti',138),('Rimnicu',146)],
    'Rimnicu':    [('Sibiu',80),('Craiova',146),('Pitesti',97)],
    'Fagaras':    [('Sibiu',99),('Bucharest',211)],
    'Pitesti':    [('Rimnicu',97),('Craiova',138),('Bucharest',101)],
    'Bucharest':  [('Fagaras',211),('Pitesti',101),('Giurgiu',90),('Urziceni',85)],
    'Giurgiu':    [('Bucharest',90)],
    'Urziceni':   [('Bucharest',85),('Hirsova',98),('Vaslui',142)],
    'Hirsova':    [('Urziceni',98),('Eforie',86)],
    'Eforie':     [('Hirsova',86)],
    'Vaslui':     [('Urziceni',142),('Iasi',92)],
    'Iasi':       [('Vaslui',92),('Neamt',87)],
    'Neamt':      [('Iasi',87)],
}

# Map coordinates (pixel-style x,y for visualization)
COORDS = {
    'Arad':       (91, 492),  'Zerind':     (75, 449),
    'Oradea':     (71, 410),  'Sibiu':      (207,457),
    'Timisoara':  (94, 410),  'Lugoj':      (165,379),
    'Mehadia':    (168,339),  'Drobeta':    (165,299),
    'Craiova':    (253,288),  'Rimnicu':    (233,410),
    'Fagaras':    (305,449),  'Pitesti':    (320,368),
    'Bucharest':  (400,327),  'Giurgiu':    (375,270),
    'Urziceni':   (456,350),  'Hirsova':    (515,350),
    'Eforie':     (562,293),  'Vaslui':     (540,437),
    'Iasi':       (473,506),  'Neamt':      (406,537),
}

START = 'Arad'
GOAL  = 'Bucharest'

def straight_line(city):
    x1,y1 = COORDS[city]
    x2,y2 = COORDS[GOAL]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

# ── BEST FIRST SEARCH + TREE BUILD ─────────────────────────────────
nodes       = {}
visit_order = [0]
goal_id     = [None]
counter     = [0]

root_id = "root"
nodes[root_id] = {
    "state": START, "depth": 0, "parent": None,
    "label": "", "status": "start", "h": straight_line(START),
    "children": [], "visit_num": 0, "cost": 0
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
        nd["status"] = "pruned"; continue
    if nd["status"] == "pruned":
        continue

    if nd["state"] == GOAL:
        nd["status"] = "goal"
        goal_id[0] = nid
        break

    for neighbor, dist in GRAPH.get(nd["state"], []):
        cid = str(uuid.uuid4())[:8]
        h   = straight_line(neighbor)
        lbl = f"{dist}km"
        if neighbor in visited:
            nodes[cid] = {
                "state": neighbor, "depth": nd["depth"]+1,
                "parent": nid, "label": lbl, "status": "pruned",
                "h": h, "children": [], "visit_num": None, "cost": nd["cost"]+dist
            }
            nd["children"].append(cid)
        else:
            visit_order[0] += 1
            nodes[cid] = {
                "state": neighbor, "depth": nd["depth"]+1,
                "parent": nid, "label": lbl, "status": "new",
                "h": h, "children": [], "visit_num": visit_order[0], "cost": nd["cost"]+dist
            }
            nd["children"].append(cid)
            visited[neighbor] = cid
            push(cid)

if goal_id[0] is None:
    print("No route found."); sys.exit()

# Mark solution path
sol_ids = set()
nid = goal_id[0]
while nid:
    sol_ids.add(nid)
    nid = nodes[nid]["parent"]
for nid in sol_ids:
    if nodes[nid]["status"] not in ("start","goal"):
        nodes[nid]["status"] = "solution"

sol_path = []
nid = goal_id[0]
while nid:
    sol_path.append(nodes[nid]["state"])
    nid = nodes[nid]["parent"]
sol_path.reverse()
total_dist = nodes[goal_id[0]]["cost"]

goal_depth   = nodes[goal_id[0]]["depth"]
total_nodes  = len(nodes)
pruned_count = sum(1 for n in nodes.values() if n["status"] == "pruned")

print(f"\n  Cities — Best First Search")
print(f"  Route: {' → '.join(sol_path)}")
print(f"  Distance: {total_dist} km  |  Nodes: {total_nodes}  |  Pruned: {pruned_count}")

# ── TREE LAYOUT ─────────────────────────────────────────────────────
sys.setrecursionlimit(50000)

def assign_x(nid, ctr):
    ch = nodes[nid]["children"]
    if not ch:
        nodes[nid]["x"] = ctr[0]*1.0; ctr[0]+=1; return
    for c in ch: assign_x(c, ctr)
    xs = [nodes[c]["x"] for c in ch]
    nodes[nid]["x"] = (min(xs)+max(xs))/2.0

assign_x(root_id, [0])
for nid, nd in nodes.items():
    nd["y"] = -nd["depth"]*2.2

all_x = [nd["x"] for nd in nodes.values()]
x_span = max(all_x)-min(all_x)+1
scale  = max(1.8, 12.0/x_span)
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
C_ROAD     = "#334455"
C_SOLROAD  = "#FFD700"

def node_color(s):
    return {"start":C_START,"solution":C_SOL,"goal":C_GOAL,
            "new":C_NEW,"pruned":C_PRUNED}.get(s, C_NEW)

# ── FIGURE: top=map, bottom=tree ────────────────────────────────────
all_x2 = [nd["x"] for nd in nodes.values()]
all_y2 = [nd["y"] for nd in nodes.values()]
tree_W = max(18, (max(all_x2)-min(all_x2)+4))
tree_H = max(12, abs(min(all_y2))+4)

fig = plt.figure(figsize=(tree_W, tree_H+10))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, tree_H/8], figure=fig, hspace=0.08)

ax_map  = fig.add_subplot(gs[0])
ax_tree = fig.add_subplot(gs[1])
ax_map.set_facecolor("#0d1117")
ax_tree.set_facecolor(C_BG)

# ── TOP: ROAD MAP ────────────────────────────────────────────────────
sol_set = set(sol_path)

# Normalize coordinates for map display
xs_all = [v[0] for v in COORDS.values()]
ys_all = [v[1] for v in COORDS.values()]
x_min, x_max = min(xs_all), max(xs_all)
y_min, y_max = min(ys_all), max(ys_all)

def norm(city):
    x, y = COORDS[city]
    return ((x-x_min)/(x_max-x_min)*16 + 0.5,
            (y-y_min)/(y_max-y_min)*7  + 0.5)

# Draw all roads
drawn_roads = set()
for city, neighbors in GRAPH.items():
    for nb, dist in neighbors:
        key = tuple(sorted([city, nb]))
        if key in drawn_roads: continue
        drawn_roads.add(key)
        x0,y0 = norm(city)
        x1,y1 = norm(nb)
        is_sol_road = (city in sol_set and nb in sol_set and
                       abs(sol_path.index(city)-sol_path.index(nb))==1
                       if city in sol_set and nb in sol_set else False)
        col = C_SOLROAD if is_sol_road else C_ROAD
        lw  = 3.0 if is_sol_road else 1.0
        ax_map.plot([x0,x1],[y0,y1], color=col, lw=lw, zorder=1,
                    solid_capstyle='round')
        # Distance label on road
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax_map.text(mx, my, str(dist), fontsize=6.5, color="#AAAACC",
                    ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#0d1117",
                              ec="#333355", lw=0.5, alpha=0.8))

# Draw cities
for city, (cx, cy) in COORDS.items():
    nx, ny = norm(city)
    if city == START:
        col, sz, tc = C_START, 180, "#000"
    elif city == GOAL:
        col, sz, tc = C_GOAL, 180, "#000"
    elif city in sol_set:
        col, sz, tc = C_SOL, 140, "#000"
    elif city in visited:
        col, sz, tc = C_NEW, 110, "#fff"
    else:
        col, sz, tc = "#334466", 90, "#888"

    ax_map.scatter(nx, ny, s=sz, c=col, zorder=5, edgecolors="#ffffff30", lw=0.8)
    h_val = round(straight_line(city), 1)
    # City name
    ax_map.text(nx, ny+0.18, city, fontsize=7.5, color=col,
                ha="center", va="bottom", fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="#0d1117")])
    # h(n)
    ax_map.text(nx, ny-0.22, f"h={h_val:.0f}", fontsize=6, color="#8899BB",
                ha="center", va="top", zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

ax_map.set_xlim(0, 17)
ax_map.set_ylim(0, 8)
ax_map.set_title(
    f"Romania Road Map  │  Best First Search: {START} → {GOAL}  │  "
    f"Route: {' → '.join(sol_path)}  ({total_dist} km)  │  "
    f"Gold = solution path  •  Blue = explored cities",
    color="white", fontsize=11, fontweight="bold", pad=10)
ax_map.axis("off")

map_lg = [
    mpatches.Patch(color=C_START,  label=f"Start: {START}"),
    mpatches.Patch(color=C_GOAL,   label=f"Goal: {GOAL}"),
    mpatches.Patch(color=C_SOL,    label="Solution path city"),
    mpatches.Patch(color=C_NEW,    label="Explored (off path)"),
    mpatches.Patch(color=C_SOLROAD,label="━━ Solution road"),
    mpatches.Patch(color=C_ROAD,   label="── Other road"),
]
ax_map.legend(handles=map_lg, loc="lower left",
              facecolor="#1a1a2e", edgecolor="#444466",
              labelcolor="white", fontsize=8.5,
              framealpha=0.97, borderpad=0.8)

# ── BOTTOM: SEARCH TREE ──────────────────────────────────────────────
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
    ex = (x0+x1)/2+0.1; ey = (y0+y1)/2
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
    sz     = 2200 if status in ("start","goal") else \
             1800 if status == "solution" else \
             1200 if status == "new" else 900

    if status in ("start","solution","goal"):
        ax_tree.scatter(x, y, s=sz+900, c=col, alpha=0.12, zorder=3)
    ax_tree.scatter(x, y, s=sz, c=col, zorder=6, edgecolors="#ffffff20", linewidths=0.9)

    tc = "#000000" if status in ("start","solution","goal") else "#FFFFFF"
    # City name in node
    city_short = nd["state"][:7]  # truncate long names
    ax_tree.text(x, y+0.18, city_short, ha="center", va="center",
                 fontsize=7, fontweight="bold", color=tc, zorder=7)
    ax_tree.text(x, y-0.28, f"h={nd['h']:.0f}", ha="center", va="center",
                 fontsize=6.5, color="#AAAACC" if status not in ("start","solution","goal") else tc,
                 zorder=7)

    vnum = nd.get("visit_num")
    if vnum is not None:
        vc = C_GOAL if status=="goal" else C_SOL if status=="solution" \
             else C_START if status=="start" else "#90CAF9"
        ax_tree.text(x-0.65, y+0.65, f"#{vnum}", fontsize=9, fontweight="bold",
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
    mpatches.Patch(color=C_START,   label=f"Start: {START}"),
    mpatches.Patch(color=C_SOL,     label="Solution path cities"),
    mpatches.Patch(color=C_GOAL,    label=f"Goal: {GOAL}"),
    mpatches.Patch(color=C_NEW,     label="Explored city (off path)"),
    mpatches.Patch(color=C_PRUNED,  label="✖  Already visited"),
    mpatches.Patch(color=C_BT,      label="↩  Skip (already visited)"),
]
ax_tree.legend(handles=legend_handles, loc="lower left",
               facecolor="#1a1a2e", edgecolor="#444466",
               labelcolor="white", fontsize=8.5,
               framealpha=0.97, borderpad=0.8, handlelength=1.4)

ax_tree.set_title(
    f"Search Tree  │  h(n) = Straight-line distance to {GOAL}  │  "
    f"Best First Search always expands city with smallest h(n)",
    color="white", fontsize=11, fontweight="bold", pad=10)
ax_tree.axis("off")

fig.suptitle(
    f"Best First Search  │  Cities Distance Problem  │  Romania Map\n"
    f"Start: {START}  →  Goal: {GOAL}  │  Path: {' → '.join(sol_path)}  │  "
    f"Distance: {total_dist} km  │  {total_nodes} tree nodes  │  {pruned_count} pruned",
    color="white", fontsize=13, fontweight="bold", y=1.005, linespacing=1.7)

plt.savefig("bfs_cities.png",
            dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("  Saved → bfs_cities.png")
plt.show()