"""
╔══════════════════════════════════════════════════════╗
║   WATER JUG PROBLEM — BFS (Breadth First Search)     ║
║   Visual Storytelling Edition                        ║
╚══════════════════════════════════════════════════════╝

BFS spreads like RIPPLES IN A POND — it checks every
state at distance 1 before distance 2, guaranteeing
the SHORTEST solution path.

Output: bfs_tree.png

Requirements:  pip install networkx matplotlib
"""

from collections import deque, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────────────────
# 1.  USER INPUT
# ─────────────────────────────────────────────────────────
print("╔══════════════════════════════════════════════╗")
print("║   WATER JUG PROBLEM  —  BFS Solver           ║")
print("╚══════════════════════════════════════════════╝")
cap_a      = int(input("  Jug A capacity : "))
cap_b      = int(input("  Jug B capacity : "))
target     = int(input("  Target amount  : "))
jug_choice = input("  Target in which jug? (A/B): ").strip().upper()
while jug_choice not in ("A", "B"):
    jug_choice = input("  Please enter A or B: ").strip().upper()

if jug_choice == "A":
    def is_goal(s): return s[0] == target
else:
    def is_goal(s): return s[1] == target


# ─────────────────────────────────────────────────────────
# 2.  BFS CORE
# ─────────────────────────────────────────────────────────
def get_successors(state):
    a, b = state
    moves = []
    if a < cap_a:            moves.append(((cap_a, b),       "Fill A"))
    if b < cap_b:            moves.append(((a, cap_b),       "Fill B"))
    if a > 0:                moves.append(((0, b),           "Empty A"))
    if b > 0:                moves.append(((a, 0),           "Empty B"))
    if a > 0 and b < cap_b:
        p = min(a, cap_b-b); moves.append(((a-p, b+p),      "A→B"))
    if b > 0 and a < cap_a:
        p = min(b, cap_a-a); moves.append(((a+p, b-p),      "B→A"))
    return [(s, l) for s, l in moves if s != state]


def bfs():
    """Standard BFS — returns tree info + BFS visit order."""
    start   = (0, 0)
    visited = {start}
    parent  = {start: None}
    elabel  = {}
    depth   = {start: 0}
    order   = [start]           # BFS discovery order
    queue   = deque([start])
    goal    = None

    while queue:
        state = queue.popleft()
        if is_goal(state):
            goal = state
            break
        for nxt, lbl in get_successors(state):
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt]           = state
                elabel[(state, nxt)]  = lbl
                depth[nxt]            = depth[state] + 1
                order.append(nxt)
                queue.append(nxt)

    return parent, elabel, depth, goal, order


parent, elabel, depth_map, goal_state, visit_order = bfs()

if goal_state is None:
    print("\n  ❌ No solution found. Try different values.")
    exit()

# Reconstruct solution path
path, node = [], goal_state
while node is not None:
    path.append(node); node = parent[node]
path.reverse()
path_set   = set(path)
path_edges = set(zip(path, path[1:]))
steps      = len(path) - 1

print(f"\n  ✅ BFS Solution: {steps} steps (guaranteed SHORTEST!)")
for i, s in enumerate(path): print(f"     Step {i}: {s}")


# ─────────────────────────────────────────────────────────
# 3.  BUILD GRAPH + WIDE LEVEL LAYOUT
# ─────────────────────────────────────────────────────────
G = nx.DiGraph()
for child, par in parent.items():
    G.add_node(child)
    if par is not None:
        G.add_edge(par, child, label=elabel.get((par, child), ""))

level_groups = defaultdict(list)
for n, d in depth_map.items():
    level_groups[d].append(n)

pos = {}
for d, nodes in level_groups.items():
    nodes_sorted = sorted(nodes)
    for i, n in enumerate(nodes_sorted):
        x = (i - (len(nodes_sorted)-1) / 2) * 3.2
        y = -d * 2.8
        pos[n] = (x, y)

max_depth = max(depth_map.values())
max_width = max(len(v) for v in level_groups.values())

# ─────────────────────────────────────────────────────────
# 4.  FIGURE — wide canvas + info panel on right
# ─────────────────────────────────────────────────────────
fig_w = max(32, max_width * 3.6)
fig_h = max(22, (max_depth + 1) * 3.2 + 4)

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#07090e")
gs  = gridspec.GridSpec(1, 2, width_ratios=[3.2, 1], wspace=0.02)
ax  = fig.add_subplot(gs[0])   # tree
ap  = fig.add_subplot(gs[1])   # info panel
ax.set_facecolor("#07090e")
ap.set_facecolor("#0a1520")
ap.set_xlim(0,1); ap.set_ylim(0,1); ap.axis("off")

# --- Colour palette (OCEAN / WATER theme — BFS = ripples in water)
C_BG     = "#07090e"
C_START  = "#00b4d8"   # bright cyan       — start
C_EXPL   = "#0d3047"   # dark teal         — explored off-path
C_PATH   = "#ffd166"   # warm gold         — solution path nodes
C_GOAL   = "#06d6a0"   # mint green        — goal
C_PEDGE  = "#ef476f"   # rose              — solution path edges
C_OEDGE  = "#0d2030"   # very dim navy     — other edges
C_WAVE   = "#0077b6"   # deep ocean blue   — wave band


# ─────────────────────────────────────────────────────────
# 5.  HORIZONTAL WAVE BANDS  ◄ THE KEY BFS VISUAL
#     Every level = one "wave ring" expanding outward
# ─────────────────────────────────────────────────────────
all_x = [p[0] for p in pos.values()]
xmin, xmax = min(all_x)-2.5, max(all_x)+2.5

for d in range(max_depth+1):
    y    = -d * 2.8
    alpha = 0.10 + 0.06*(d % 2)
    # Shaded horizontal band
    ax.fill_between([xmin, xmax], [y-1.35, y-1.35], [y+1.35, y+1.35],
                    color=C_WAVE, alpha=alpha, zorder=0)
    ax.axhline(y - 1.35, color=C_WAVE, lw=0.5, ls=":", alpha=0.3)

    # Wave label LEFT
    label = f"≋  Wave 0  (Start)" if d == 0 else f"≋  Wave {d}"
    ax.text(xmin - 0.3, y, label,
            color=C_WAVE, fontsize=9, va="center", alpha=0.65,
            style="italic", ha="right",
            path_effects=[pe.withStroke(linewidth=2, foreground="#07090e")])

    # Node count RIGHT
    n_wave = len(level_groups[d])
    ax.text(xmax + 0.3, y,
            f"{n_wave} node{'s' if n_wave!=1 else ''}",
            color=C_WAVE, fontsize=8.5, va="center", alpha=0.6,
            style="italic", ha="left")


# ─────────────────────────────────────────────────────────
# 6.  DRAW EDGES
# ─────────────────────────────────────────────────────────
for u, v in G.edges():
    on_path = (u, v) in path_edges
    ec = C_PEDGE if on_path else C_OEDGE
    lw = 2.8     if on_path else 0.8
    ax.annotate("", xy=pos[v], xytext=pos[u],
                arrowprops=dict(arrowstyle="-|>", color=ec, lw=lw,
                                mutation_scale=18,
                                connectionstyle="arc3,rad=0.06"),
                zorder=2)

# Edge labels
for (u, v), lbl in elabel.items():
    if (u, v) not in G.edges(): continue
    mx = pos[u][0]*0.45 + pos[v][0]*0.55
    my = pos[u][1]*0.45 + pos[v][1]*0.55
    on_path = (u, v) in path_edges
    fc2  = "#2a0014" if on_path else "#071525"
    ec2  = C_PEDGE   if on_path else "#1e3d5c"
    ax.text(mx + 0.08, my, lbl, fontsize=7, color=ec2,
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.22", fc=fc2, ec=ec2, lw=0.8, alpha=0.92))


# ─────────────────────────────────────────────────────────
# 7.  DRAW NODES
# ─────────────────────────────────────────────────────────
visit_idx = {n: i for i, n in enumerate(visit_order)}

for n in G.nodes():
    x, y = pos[n]
    if n == goal_state:
        color, size = C_GOAL,  2000
    elif n in path_set:
        color, size = C_PATH,  1500
    elif n == (0, 0):
        color, size = C_START, 1500
    else:
        color, size = C_EXPL,  1000

    # Glow halo for important nodes
    if n in path_set or n == goal_state or n == (0, 0):
        ax.scatter(x, y, s=size+700, c=color, alpha=0.15, zorder=3)

    ax.scatter(x, y, s=size, c=color, zorder=4,
               edgecolors="#ffffff25", linewidths=0.9)

    # State text (A, B)
    tc = "#000000" if n in (path_set | {goal_state}) else "#8fc9df"
    ax.text(x, y, f"({n[0]},{n[1]})", ha="center", va="center",
            fontsize=8, fontweight="bold", color=tc, zorder=7)

    # Visit-order badge on off-path nodes
    if n not in path_set and n != goal_state:
        ax.text(x, y + 0.62, f"#{visit_idx.get(n,'?')}",
                ha="center", va="center",
                fontsize=5.5, color="#4da6c8", zorder=8, alpha=0.75)

    # Mini jug fill bars below solution-path nodes
    if n in path_set or n == goal_state:
        bar_ox = x - 0.52
        for ji, (jval, jcol, jcap) in enumerate([
                (n[0], "#00b4d8", cap_a),   # Jug A — blue
                (n[1], "#ff9f1c", cap_b)]):  # Jug B — orange
            fill_h = 0.55 * (jval/jcap) if jcap > 0 else 0
            ax.add_patch(plt.Rectangle(
                (bar_ox + ji*0.58, y-1.1), 0.44, 0.58,
                fc="#112233", ec="#223344", lw=0.6, zorder=6))
            if fill_h > 0:
                ax.add_patch(plt.Rectangle(
                    (bar_ox + ji*0.58, y-1.1), 0.44, fill_h,
                    fc=jcol, alpha=0.75, zorder=7))
            ax.text(bar_ox + ji*0.58 + 0.22, y-1.12 + fill_h + 0.07,
                    f"{'A' if ji==0 else 'B'}={jval}",
                    ha="center", va="bottom", fontsize=5.5,
                    color=jcol, fontweight="bold", zorder=8)


# ─────────────────────────────────────────────────────────
# 8.  START / GOAL BADGES
# ─────────────────────────────────────────────────────────
def badge(ax, x, y, txt, bg, fg):
    ax.text(x, y+0.95, txt, ha="center", va="center", fontsize=8.5,
            color=fg, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc=bg, ec=fg, lw=1.6))

badge(ax, pos[(0,0)][0],      pos[(0,0)][1],      "► START",  C_START, "#000")
badge(ax, pos[goal_state][0], pos[goal_state][1],  "★ GOAL!",  C_GOAL,  "#000")


# ─────────────────────────────────────────────────────────
# 9.  TITLE
# ─────────────────────────────────────────────────────────
ax.set_title(
    "BFS — Breadth First Search\n"
    "\"Like ripples on water: every state 1 step away is checked before any state 2 steps away\"\n"
    f"Jug A = {cap_a}L  |  Jug B = {cap_b}L  |  Target = {target}L in Jug {jug_choice}  |  "
    f"Solution = {steps} steps  (SHORTEST guaranteed ✓)",
    color="white", fontsize=13, fontweight="bold",
    loc="left", pad=16, linespacing=1.6)
ax.axis("off")


# ─────────────────────────────────────────────────────────
# 10.  SIDE PANEL
# ─────────────────────────────────────────────────────────
def pt(y, txt, sz=9, col="#cde8f5", bold=False, mono=False):
    ap.text(0.07, y, txt, transform=ap.transAxes,
            fontsize=sz, color=col, fontweight="bold" if bold else "normal",
            va="top", ha="left",
            family="monospace" if mono else "DejaVu Sans", clip_on=False)

# Panel border
ap.add_patch(plt.Rectangle((0.03, 0.01), 0.94, 0.97, transform=ap.transAxes,
             fc="#081420", ec=C_WAVE, lw=1.5, alpha=0.9, zorder=0))

y = 0.97
pt(y, "HOW BFS WORKS", 12, C_START, True); y -= 0.05
for line in ["Drop a stone in a calm pond.",
             "Ripples spread outward, ring",
             "by ring, in all directions.",
             "", "BFS does the same thing:"]:
    pt(y, line, 8.5, "#8fc9e0"); y -= 0.028
for line in [" 1. Check ALL states 1 op away",
             " 2. Then ALL 2 ops away",
             " 3. Then 3 ops... and so on."]:
    pt(y, line, 8, "#c0dde8"); y -= 0.026
y -= 0.005
pt(y, "→ First goal found is the", 8.5, "#8fc9e0"); y -= 0.026
pt(y, "  SHORTEST PATH POSSIBLE!", 9.5, C_GOAL, True); y -= 0.042

pt(y, "TREE SHAPE", 10, C_PATH, True); y -= 0.030
for line in ["WIDE & FLAT", "Many nodes per level.",
             "Each shaded band = one BFS wave."]:
    pt(y, line, 8.5, "#8fc9e0"); y -= 0.026
y -= 0.012

pt(y, "COLOUR GUIDE", 10, C_START, True); y -= 0.032
legend_items = [
    (C_START, "Start state  (0, 0)"),
    (C_PATH,  "Solution path node"),
    (C_GOAL,  f"GOAL  ({target}L in Jug {jug_choice})"),
    (C_EXPL,  "Explored, off-path"),
    (C_PEDGE, "Solution path edge"),
    (C_OEDGE, "Other tree edge"),
]
for bg, desc in legend_items:
    ap.add_patch(plt.Rectangle((0.07, y-0.021), 0.11, 0.027,
                 transform=ap.transAxes, fc=bg, ec="#ffffff33", lw=0.7, zorder=5))
    pt(y, f"               {desc}", 8, "#cde8f5"); y -= 0.032
y -= 0.008

pt(y, "SOLUTION PATH", 10, C_GOAL, True); y -= 0.030
pt(y, f"{steps} steps — provably optimal", 8.5, C_PATH); y -= 0.030

for i, (a, b) in enumerate(path):
    if y < 0.07: break
    bar_a = "█"*a + "░"*(cap_a-a)
    bar_b = "█"*b + "░"*(cap_b-b)
    col   = C_GOAL if (a,b)==goal_state else (C_START if i==0 else C_PATH)
    op    = ""
    if i < len(path)-1:
        op = "  ← " + elabel.get((path[i], path[i+1]), "")
    pt(y, f" {i}) A[{bar_a}] B[{bar_b}]{op}", 7.2, col, mono=True)
    y -= 0.028
y -= 0.012

pt(y, "STATS", 10, C_START, True);                     y -= 0.028
pt(y, f"Nodes in tree : {len(visit_order)}",   8.5, "#8fc9e0"); y -= 0.024
pt(y, f"Solution depth: {steps}",              8.5, "#8fc9e0"); y -= 0.024
pt(y, f"Tree levels   : {max_depth+1}",        8.5, "#8fc9e0"); y -= 0.024
pt(y, f"Max nodes/wave: {max_width}",          8.5, "#8fc9e0")

# Footer
fig.text(0.5, 0.003,
    "BFS visits every node at depth D before any node at depth D+1 "
    "— this is why it ALWAYS finds the SHORTEST path",
    ha="center", fontsize=9.5, color=C_WAVE, style="italic")

plt.savefig("bfs_tree.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n  📊 Saved → bfs_tree.png")
plt.show()