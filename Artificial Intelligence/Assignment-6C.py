"""
=============================================================
  Constraint Satisfaction Problem (CSP)
  Problem : MAP COLORING
  Classic: Australia / Custom Region Map
  Technique: Backtracking + Forward Checking
=============================================================
"""

import os
import copy
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "csp_map_coloring.png")

# ── Colour palette ──
CLR_BG     = "#0C0C14"
CLR_PANEL  = "#131320"
CLR_ACCENT = "#F59E0B"
CLR_GREEN  = "#10B981"
CLR_RED    = "#EF4444"
CLR_TEXT   = "#F1F5F9"
CLR_DIM    = "#64748B"
CLR_BORDER = "#1E1E30"
CLR_HEADER = "#F59E0B"

# Map region fill colors (CSP domain)
DOMAIN_COLORS = {
    "Red":    "#EF4444",
    "Green":  "#10B981",
    "Blue":   "#3B82F6",
    "Yellow": "#F59E0B",
}

# Region polygon positions for plotting (centroid x, y)
AUSTRALIA_POS = {
    "WA":  (1.2, 3.5),
    "NT":  (3.0, 4.2),
    "SA":  (3.2, 2.8),
    "QLD": (5.0, 4.2),
    "NSW": (5.2, 2.8),
    "VIC": (4.6, 1.8),
    "TAS": (4.8, 0.7),
}

AUSTRALIA_ADJ = {
    "WA":  ["NT", "SA"],
    "NT":  ["WA", "SA", "QLD"],
    "SA":  ["WA", "NT", "QLD", "NSW", "VIC"],
    "QLD": ["NT", "SA", "NSW"],
    "NSW": ["QLD", "SA", "VIC"],
    "VIC": ["SA", "NSW"],
    "TAS": [],
}


# =====================================================================
#  CSP SOLVER
# =====================================================================

def is_valid(region, color, assignment, adjacency):
    for neighbor in adjacency.get(region, []):
        if assignment.get(neighbor) == color:
            return False
    return True


def solve_map_coloring(regions, adjacency, colors):
    """
    Backtracking CSP solver with Forward Checking.
    Returns assignment dict and search steps.
    """
    assignment = {}
    steps      = []   # (region, color_tried, ok, assignment_snapshot)
    domains    = {r: list(colors) for r in regions}

    def forward_check(region, color, domains):
        new_domains = copy.deepcopy(domains)
        for neighbor in adjacency.get(region, []):
            if neighbor not in assignment:
                if color in new_domains[neighbor]:
                    new_domains[neighbor].remove(color)
                if not new_domains[neighbor]:
                    return None  # Domain wipeout
        return new_domains

    def backtrack(assign, unassigned, domains):
        if not unassigned:
            return True

        # MRV (Minimum Remaining Values) heuristic
        region = min(unassigned, key=lambda r: len(domains[r]))

        for color in domains[region]:
            if is_valid(region, color, assign, adjacency):
                assign[region] = color
                steps.append((region, color, True, dict(assign)))

                new_domains = forward_check(region, color, domains)
                if new_domains is not None:
                    remaining = [r for r in unassigned if r != region]
                    if backtrack(assign, remaining, new_domains):
                        return True

                steps.append((region, color, False, dict(assign)))
                del assign[region]

        return False

    success = backtrack(assignment, list(regions), domains)
    return assignment if success else None, steps


# =====================================================================
#  TERMINAL OUTPUT
# =====================================================================

def print_results(regions, adjacency, colors, assignment, steps):
    print()
    print("+" + "=" * 62 + "+")
    print("|   CSP -- MAP COLORING SOLVER                                 |")
    print("+" + "=" * 62 + "+")
    print(f"|   Regions    : {len(regions):<46}|")
    print(f"|   Colors     : {len(colors):<46}|")
    print(f"|   Steps      : {len(steps):<46}|")
    print(f"|   Status     : {'SOLVED ✓' if assignment else 'NO SOLUTION ✗':<46}|")
    print("+" + "=" * 62 + "+")

    if assignment:
        print(f"\n  {'Region':<10} {'Color':<10} {'Neighbors':<30} {'Conflict?'}")
        print("  " + "=" * 60)
        for region in sorted(assignment.keys()):
            color = assignment[region]
            neighbors = adjacency.get(region, [])
            neighbor_colors = [f"{n}={assignment.get(n,'?')}" for n in neighbors]
            conflict = any(assignment.get(n) == color for n in neighbors)
            nc_str = ", ".join(neighbor_colors) if neighbor_colors else "None"
            status = "✗ CONFLICT" if conflict else "✓ OK"
            print(f"  {region:<10} {color:<10} {nc_str:<30} {status}")

    print(f"\n  Step-by-step (first 20):")
    print(f"  {'Step':<6} {'Region':<8} {'Color':<10} {'Result'}")
    print("  " + "-" * 38)
    for i, (reg, col, ok, _) in enumerate(steps[:20], 1):
        print(f"  {i:<6} {reg:<8} {col:<10} {'✓ Placed' if ok else '✗ Backtrack'}")
    print()


# =====================================================================
#  VISUALIZATION
# =====================================================================

def draw_graph(ax, regions, adjacency, assignment, pos, title="Map Graph", colors_map=None):
    ax.set_facecolor(CLR_PANEL)
    ax.axis("off")
    ax.set_title(title, color=CLR_ACCENT, fontsize=9, fontfamily="monospace", pad=4)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw edges
    drawn = set()
    for region, neighbors in adjacency.items():
        for nb in neighbors:
            key = tuple(sorted([region, nb]))
            if key not in drawn:
                drawn.add(key)
                x1, y1 = pos[region]
                x2, y2 = pos[nb]
                ax.plot([x1, x2], [y1, y2], color=CLR_BORDER, lw=1.5, zorder=1)

    # Draw nodes
    for region in regions:
        x, y = pos[region]
        color = assignment.get(region) if assignment else None
        fill  = DOMAIN_COLORS.get(color, CLR_PANEL) if color else CLR_PANEL
        edge  = DOMAIN_COLORS.get(color, CLR_ACCENT) if color else CLR_DIM

        circle = plt.Circle((x, y), 0.35, color=fill, ec=edge, lw=2.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, region, ha="center", va="center",
                fontsize=9, fontweight="bold", color=CLR_TEXT,
                fontfamily="monospace", zorder=4)


def visualize(regions, adjacency, colors, assignment, steps):
    fig = plt.figure(figsize=(18, 12), facecolor=CLR_BG)
    fig.suptitle(
        "CSP  —  Map Coloring Problem (Australia)\n"
        "Technique: Backtracking + Forward Checking + MRV Heuristic",
        color=CLR_HEADER, fontsize=15, fontweight="bold",
        fontfamily="monospace", y=0.98
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.5, wspace=0.4,
                           left=0.03, right=0.97, top=0.90, bottom=0.06)

    # ── Panel 1: Uncolored Graph ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    draw_graph(ax1, regions, adjacency, {}, AUSTRALIA_POS, "Constraint Graph (Uncolored)")

    # ── Panel 2: Colored Graph ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2:4])
    draw_graph(ax2, regions, adjacency, assignment, AUSTRALIA_POS, "Solved Map Coloring ✓")

    # Add color legend to solved graph
    legend_handles = [
        mpatches.Patch(facecolor=v, edgecolor=CLR_BG, label=k)
        for k, v in DOMAIN_COLORS.items()
        if k in (assignment or {}).values()
    ]
    ax2.legend(
        handles=legend_handles,
        loc="lower right", facecolor=CLR_BG, edgecolor=CLR_BORDER,
        labelcolor=CLR_TEXT, fontsize=8
    )

    # ── Panel 3: Assignment Table ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(CLR_PANEL)
    ax3.axis("off")
    ax3.set_title("Final Assignment", color=CLR_ACCENT, fontsize=9,
                  fontfamily="monospace", pad=4)

    if assignment:
        table_data = [[r, assignment[r]] for r in sorted(assignment.keys())]
        tbl = ax3.table(
            cellText=table_data,
            colLabels=["Region", "Color"],
            cellLoc="center", loc="center",
            bbox=[0.05, 0.05, 0.9, 0.85]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(CLR_BG if r == 0 else CLR_PANEL)
            cell.set_edgecolor(CLR_BORDER)
            if r > 0 and c == 1:
                color_name = table_data[r - 1][1]
                cell.set_facecolor(DOMAIN_COLORS.get(color_name, CLR_PANEL) + "44")
            cell.set_text_props(
                color=CLR_ACCENT if r == 0 else CLR_TEXT,
                fontfamily="monospace",
                fontweight="bold" if r == 0 else "normal"
            )

    # ── Panel 4: Adjacency / Constraint Table ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CLR_PANEL)
    ax4.axis("off")
    ax4.set_title("Adjacency Constraints", color=CLR_ACCENT, fontsize=9,
                  fontfamily="monospace", pad=4)

    table_data2 = []
    for region in sorted(regions):
        nbrs = ", ".join(adjacency.get(region, [])) or "None"
        table_data2.append([region, nbrs])

    tbl2 = ax4.table(
        cellText=table_data2,
        colLabels=["Region", "Neighbors"],
        cellLoc="left", loc="center",
        bbox=[0.0, 0.05, 1.0, 0.85]
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8.5)
    for (r, c), cell in tbl2.get_celld().items():
        cell.set_facecolor(CLR_BG if r == 0 else CLR_PANEL)
        cell.set_edgecolor(CLR_BORDER)
        cell.set_text_props(
            color=CLR_ACCENT if r == 0 else CLR_TEXT,
            fontfamily="monospace"
        )

    # ── Panel 5: Steps Timeline ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.set_facecolor(CLR_PANEL)
    ax5.set_title("Backtracking Search Steps (Assign vs Backtrack per Region)",
                  color=CLR_ACCENT, fontsize=9, fontfamily="monospace", pad=4)

    assigns_per_region  = {r: 0 for r in regions}
    backtracks_per_region = {r: 0 for r in regions}
    for reg, col, ok, _ in steps:
        if ok:
            assigns_per_region[reg]    += 1
        else:
            backtracks_per_region[reg] += 1

    x_vals = list(range(len(regions)))
    region_list = sorted(regions)
    a_vals = [assigns_per_region[r] for r in region_list]
    b_vals = [backtracks_per_region[r] for r in region_list]

    ax5.bar([i - 0.2 for i in x_vals], a_vals, width=0.38,
            color=CLR_GREEN, label="Assigned", alpha=0.9)
    ax5.bar([i + 0.2 for i in x_vals], b_vals, width=0.38,
            color=CLR_RED, label="Backtrack", alpha=0.9)

    ax5.set_xticks(x_vals)
    ax5.set_xticklabels(region_list, fontsize=9, color=CLR_TEXT, fontfamily="monospace")
    ax5.set_ylabel("Steps", color=CLR_DIM, fontsize=8, fontfamily="monospace")
    ax5.set_facecolor(CLR_PANEL)
    ax5.spines[:].set_color(CLR_BORDER)
    ax5.tick_params(colors=CLR_DIM)
    for label in ax5.get_yticklabels():
        label.set_color(CLR_DIM)
        label.set_fontfamily("monospace")
    ax5.legend(facecolor=CLR_BG, edgecolor=CLR_BORDER, labelcolor=CLR_TEXT, fontsize=8)

    # Color domain swatches
    swatch_x = 0.01
    for color, hex_c in DOMAIN_COLORS.items():
        ax5.add_patch(mpatches.FancyBboxPatch(
            (swatch_x, -0.5), 0.6, 0.4,
            boxstyle="round,pad=0.05", transform=ax5.transData,
            facecolor=hex_c, edgecolor=CLR_BG, linewidth=1, zorder=5
        ))
        ax5.text(swatch_x + 0.3, -0.28, color[0], ha="center", va="center",
                 fontsize=7, fontweight="bold", color="white",
                 fontfamily="monospace", zorder=6)
        swatch_x += 0.8

    # Status banner
    stat_clr = CLR_GREEN if assignment else CLR_RED
    stat_txt = (f"✓  MAP COLORED  —  {len(assignment)} regions  |  "
                f"{len(colors)} colors used  |  {len(steps)} backtracking steps"
                if assignment else f"✗  NO VALID COLORING FOUND  —  {len(steps)} steps")
    fig.text(0.5, 0.013, stat_txt, ha="center", va="bottom",
             fontsize=10, fontweight="bold", fontfamily="monospace",
             color=stat_clr,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#0D2818" if assignment else "#2D1B1B",
                       edgecolor=stat_clr, lw=1.5))

    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight", facecolor=CLR_BG)
    print(f"\n  [INFO] Visualization saved  ->  {OUTPUT_FILE}")
    plt.show()


# =====================================================================
#  USER INPUT
# =====================================================================

def get_input():
    print()
    print("+" + "=" * 62 + "+")
    print("|   CSP  --  Map Coloring Solver                               |")
    print("|   Default: Australia Map (7 regions, 3 colors)               |")
    print("+" + "=" * 62 + "+")
    print()
    print("  Press Enter to use Australia map")
    choice = input("  or type  custom  to add extra colors/regions : ").strip().lower()

    if choice == "custom":
        n_colors = input("\n  Number of colors to use [2-4] (default 3): ").strip()
        try:
            n_colors = max(2, min(4, int(n_colors)))
        except ValueError:
            n_colors = 3
        colors = list(DOMAIN_COLORS.keys())[:n_colors]
        return list(AUSTRALIA_ADJ.keys()), AUSTRALIA_ADJ, colors

    return list(AUSTRALIA_ADJ.keys()), AUSTRALIA_ADJ, ["Red", "Green", "Blue"]


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print()
    print("+" + "=" * 62 + "+")
    print("|      CSP -- Map Coloring Problem Solver                      |")
    print("|      Technique : Backtracking + Forward Checking + MRV      |")
    print("+" + "=" * 62 + "+")

    regions, adjacency, colors = get_input()

    print(f"\n  Regions : {', '.join(regions)}")
    print(f"  Colors  : {', '.join(colors)}")
    print(f"\n  Solving ...\n")

    assignment, steps = solve_map_coloring(regions, adjacency, colors)

    print_results(regions, adjacency, colors, assignment, steps)
    visualize(regions, adjacency, colors, assignment, steps)


if __name__ == "__main__":
    main()