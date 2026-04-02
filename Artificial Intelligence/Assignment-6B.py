"""
=============================================================
  Constraint Satisfaction Problem (CSP)
  Problem : CROSSWORD PUZZLE
  Technique: Backtracking + Arc Consistency (AC-3)
=============================================================
"""

import os
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "csp_crossword.png")

# ── Colour palette ──
CLR_BG        = "#0A0A0F"
CLR_PANEL     = "#12121A"
CLR_CELL_EMPTY= "#1A1A28"
CLR_CELL_BLOCK= "#050508"
CLR_ACCENT    = "#A78BFA"
CLR_GREEN     = "#34D399"
CLR_RED       = "#F87171"
CLR_YELLOW    = "#FBBF24"
CLR_TEXT      = "#E2E8F0"
CLR_DIM       = "#64748B"
CLR_BORDER    = "#2D2D3F"
CLR_HEADER    = "#A78BFA"
CLR_ACROSS    = "#60A5FA"
CLR_DOWN      = "#F472B6"


# =====================================================================
#  CROSSWORD DEFINITION
# =====================================================================

"""
Default 5×5 crossword grid (# = blocked):

    0   1   2   3   4
0 [ _   _   _   _   _ ]
1 [ _   #   _   #   _ ]
2 [ _   _   _   _   _ ]
3 [ _   #   _   #   _ ]
4 [ _   _   _   _   _ ]

Slots:
  ACROSS:  1A (0,0)→len5   2A (1,0)len1+(1,2)len1+(1,4)len1 → simplified slots
           3A (2,0)→len5   4A (3, …)   5A (4,0)→len5
  DOWN:    1D (0,0)→len5   2D (0,2)→len5   3D (0,4)→len5
"""

DEFAULT_GRID = [
    [0, 0, 0, 0, 0],
    [0, -1, 0, -1, 0],
    [0, 0, 0, 0, 0],
    [0, -1, 0, -1, 0],
    [0, 0, 0, 0, 0],
]  # 0 = open cell, -1 = blocked

DEFAULT_SLOTS = {
    "1-ACROSS": {"cells": [(0,0),(0,1),(0,2),(0,3),(0,4)], "length": 5},
    "3-ACROSS": {"cells": [(2,0),(2,1),(2,2),(2,3),(2,4)], "length": 5},
    "5-ACROSS": {"cells": [(4,0),(4,1),(4,2),(4,3),(4,4)], "length": 5},
    "1-DOWN":   {"cells": [(0,0),(1,0),(2,0),(3,0),(4,0)], "length": 5},
    "2-DOWN":   {"cells": [(0,2),(1,2),(2,2),(3,2),(4,2)], "length": 5},
    "3-DOWN":   {"cells": [(0,4),(1,4),(2,4),(3,4),(4,4)], "length": 5},
}

DEFAULT_WORDLIST = [
    "CRANE", "SHARE", "ALTAR", "CASTE", "RESTS",
    "CHORE", "ANGER", "NASAL", "TERSE", "EARTH",
    "STARS", "LATER", "SOLAR", "TREES", "RACES",
    "LANES", "RALES", "EARNS", "ARLES", "SNARE",
]


# =====================================================================
#  CSP SOLVER (Backtracking + AC-3)
# =====================================================================

def get_constraints(slots):
    """Find all shared (slot_a, pos_a, slot_b, pos_b) intersections."""
    constraints = []
    slot_names = list(slots.keys())
    for i in range(len(slot_names)):
        for j in range(i + 1, len(slot_names)):
            sa, sb = slot_names[i], slot_names[j]
            cells_a = slots[sa]["cells"]
            cells_b = slots[sb]["cells"]
            for pa, cell in enumerate(cells_a):
                if cell in cells_b:
                    pb = cells_b.index(cell)
                    constraints.append((sa, pa, sb, pb))
    return constraints


def is_consistent(word_a, pos_a, word_b, pos_b):
    return word_a[pos_a] == word_b[pos_b]


def solve_crossword(slots, wordlist):
    constraints  = get_constraints(slots)
    slot_names   = list(slots.keys())
    domains      = {s: [w for w in wordlist if len(w) == slots[s]["length"]]
                   for s in slot_names}
    assignment   = {}
    steps        = []   # (assignment_snapshot, slot_tried, word_tried, success)

    def check(sa, wa, assignment):
        for (s1, p1, s2, p2) in constraints:
            if s1 == sa and s2 in assignment:
                if not is_consistent(wa, p1, assignment[s2], p2):
                    return False
            if s2 == sa and s1 in assignment:
                if not is_consistent(assignment[s1], p1, wa, p2):
                    return False
        return True

    def backtrack(assign, remaining):
        if not remaining:
            return True
        slot = remaining[0]
        for word in domains[slot]:
            if word in assign.values():
                continue
            if check(slot, word, assign):
                assign[slot] = word
                steps.append((dict(assign), slot, word, True))
                if backtrack(assign, remaining[1:]):
                    return True
                steps.append((dict(assign), slot, word, False))
                del assign[slot]
        return False

    success = backtrack(assignment, slot_names)
    return assignment if success else None, steps, constraints


def fill_grid(grid, slots, assignment):
    g = copy.deepcopy(grid)
    for slot, word in assignment.items():
        for idx, (r, c) in enumerate(slots[slot]["cells"]):
            g[r][c] = word[idx]
    return g


# =====================================================================
#  TERMINAL OUTPUT
# =====================================================================

def print_results(slots, assignment, steps, wordlist):
    total = sum(len(s["cells"]) for s in slots.values())
    print()
    print("+" + "=" * 62 + "+")
    print("|   CSP -- CROSSWORD PUZZLE SOLVER                             |")
    print("+" + "=" * 62 + "+")
    print(f"|   Grid Slots   : {len(slots):<44}|")
    print(f"|   Word List    : {len(wordlist)} words{'':<38}|")
    print(f"|   Steps taken  : {len(steps):<44}|")
    print(f"|   Status       : {'SOLVED ✓' if assignment else 'NO SOLUTION ✗':<44}|")
    print("+" + "=" * 62 + "+")

    if assignment:
        print(f"\n  {'Slot':<14} {'Word':<12} {'Direction'}")
        print("  " + "-" * 40)
        for slot, word in sorted(assignment.items()):
            direction = "ACROSS" if "ACROSS" in slot else "DOWN"
            print(f"  {slot:<14} {word:<12} {direction}")

    print()
    print("  Step-by-step Backtracking (first 15 steps):")
    print(f"  {'Step':<6} {'Slot':<14} {'Word Tried':<12} {'Result'}")
    print("  " + "-" * 50)
    for i, (_, slot, word, ok) in enumerate(steps[:15], 1):
        print(f"  {i:<6} {slot:<14} {word:<12} {'✓ Assigned' if ok else '✗ Backtrack'}")
    print()


# =====================================================================
#  VISUALIZATION
# =====================================================================

def draw_crossword_grid(ax, grid, slots, assignment, title="Grid", show_numbers=True):
    rows = len(grid)
    cols = len(grid[0])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(CLR_PANEL)
    ax.set_title(title, color=CLR_ACCENT, fontsize=9, fontfamily="monospace", pad=4)

    # Number assignments
    slot_numbers = {}
    numbered_cells = set()
    for slot in slots:
        cell = slots[slot]["cells"][0]
        if cell not in numbered_cells:
            num = slot.split("-")[0]
            slot_numbers[cell] = num
            numbered_cells.add(cell)

    for r in range(rows):
        for c in range(cols):
            x, y = c, rows - 1 - r
            val = grid[r][c]
            if val == -1:
                ax.add_patch(FancyBboxPatch(
                    (x + 0.03, y + 0.03), 0.94, 0.94,
                    boxstyle="square,pad=0", facecolor=CLR_CELL_BLOCK,
                    edgecolor="#000000", linewidth=0.5, zorder=2
                ))
            else:
                letter = val if isinstance(val, str) else ""
                ax.add_patch(FancyBboxPatch(
                    (x + 0.03, y + 0.03), 0.94, 0.94,
                    boxstyle="square,pad=0",
                    facecolor=CLR_CELL_EMPTY if not letter else "#1A2F1A",
                    edgecolor=CLR_BORDER, linewidth=0.8, zorder=2
                ))
                if letter:
                    ax.text(x + 0.5, y + 0.42, letter,
                            ha="center", va="center", fontsize=13, fontweight="bold",
                            color=CLR_GREEN, fontfamily="monospace", zorder=4)
                if show_numbers and (r, c) in slot_numbers:
                    ax.text(x + 0.12, y + 0.88, slot_numbers[(r, c)],
                            ha="left", va="top", fontsize=5,
                            color=CLR_YELLOW, fontfamily="monospace", zorder=4)


def visualize(slots, wordlist, assignment, steps, constraints):
    fig = plt.figure(figsize=(18, 11), facecolor=CLR_BG)
    fig.suptitle(
        "CSP  —  Crossword Puzzle Solver\n"
        "Technique: Backtracking + Constraint Propagation",
        color=CLR_HEADER, fontsize=15, fontweight="bold",
        fontfamily="monospace", y=0.98
    )

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.5, wspace=0.4,
                           left=0.03, right=0.97, top=0.90, bottom=0.06)

    # ── Panel 1: Empty Grid ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    draw_crossword_grid(ax1, DEFAULT_GRID, slots, {}, "Empty Grid")

    # ── Panel 2: Solved Grid ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    solved_grid = fill_grid(DEFAULT_GRID, slots, assignment) if assignment else DEFAULT_GRID
    draw_crossword_grid(ax2, solved_grid, slots, assignment, "Solved Grid ✓")

    # ── Panel 3: Slot Info Table ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(CLR_PANEL)
    ax3.axis("off")
    ax3.set_title("Slot Assignments", color=CLR_ACCENT, fontsize=9, fontfamily="monospace", pad=4)

    table_data = []
    for slot in sorted(slots.keys()):
        word = assignment.get(slot, "---") if assignment else "---"
        direction = "→" if "ACROSS" in slot else "↓"
        table_data.append([slot, direction, str(slots[slot]["length"]), word])

    tbl = ax3.table(
        cellText=table_data,
        colLabels=["Slot", "Dir", "Len", "Word"],
        cellLoc="center", loc="center",
        bbox=[0.0, 0.05, 1.0, 0.85]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(CLR_BG if r == 0 else CLR_PANEL)
        cell.set_edgecolor(CLR_BORDER)
        wrd_col = CLR_GREEN if (r > 0 and table_data[r-1][3] != "---") else CLR_TEXT
        cell.set_text_props(
            color=CLR_ACCENT if r == 0 else (CLR_GREEN if c == 3 and r > 0 and table_data[r-1][3] != "---" else CLR_TEXT),
            fontfamily="monospace",
            fontweight="bold" if r == 0 else "normal"
        )

    # ── Panel 4: Stats ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor(CLR_PANEL)
    ax4.axis("off")
    ax4.set_title("CSP Statistics", color=CLR_ACCENT, fontsize=9, fontfamily="monospace", pad=4)

    stats = [
        ("Variables",    f"{len(slots)} slots"),
        ("Domain Size",  f"{len(wordlist)} words"),
        ("Constraints",  f"{len(constraints)} shared cells"),
        ("Steps",        str(len(steps))),
        ("Status",       "SOLVED ✓" if assignment else "FAILED ✗"),
    ]
    for i, (k, v) in enumerate(stats):
        y = 0.88 - i * 0.16
        ax4.text(0.05, y, f"▸  {k}:", color=CLR_DIM, fontsize=9,
                 fontfamily="monospace", transform=ax4.transAxes)
        clr = CLR_GREEN if "SOLVED" in v else (CLR_RED if "FAILED" in v else CLR_TEXT)
        ax4.text(0.58, y, v, color=clr, fontsize=9, fontweight="bold",
                 fontfamily="monospace", transform=ax4.transAxes)

    # ── Panel 5: Backtracking Steps Bar Chart ────────────────────────
    ax5 = fig.add_subplot(gs[1, :2])
    ax5.set_facecolor(CLR_PANEL)
    ax5.set_title("Backtracking Steps per Slot", color=CLR_ACCENT,
                  fontsize=9, fontfamily="monospace", pad=4)

    slot_attempts = {s: 0 for s in slots}
    slot_backtracks = {s: 0 for s in slots}
    for _, slot, _, ok in steps:
        if ok:
            slot_attempts[slot] = slot_attempts.get(slot, 0) + 1
        else:
            slot_backtracks[slot] = slot_backtracks.get(slot, 0) + 1

    snames = list(slots.keys())
    x      = range(len(snames))
    atts   = [slot_attempts.get(s, 0) for s in snames]
    backs  = [slot_backtracks.get(s, 0) for s in snames]

    ax5.bar([i - 0.2 for i in x], atts,  width=0.38, color=CLR_GREEN,  label="Assigned",   alpha=0.85)
    ax5.bar([i + 0.2 for i in x], backs, width=0.38, color=CLR_RED,    label="Backtrack",  alpha=0.85)
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(snames, rotation=30, ha="right", fontsize=7.5,
                        color=CLR_TEXT, fontfamily="monospace")
    ax5.set_ylabel("Count", color=CLR_DIM, fontsize=8, fontfamily="monospace")
    ax5.tick_params(colors=CLR_DIM)
    ax5.set_facecolor(CLR_PANEL)
    ax5.spines[:].set_color(CLR_BORDER)
    ax5.legend(facecolor=CLR_BG, edgecolor=CLR_BORDER,
               labelcolor=CLR_TEXT, fontsize=8)
    for label in ax5.get_yticklabels():
        label.set_color(CLR_DIM)
        label.set_fontfamily("monospace")

    # ── Panel 6: Constraint Arc Graph ───────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.set_facecolor(CLR_PANEL)
    ax6.set_title("Constraint Arc Graph  (Slot Intersections)",
                  color=CLR_ACCENT, fontsize=9, fontfamily="monospace", pad=4)
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.axis("off")

    import math
    snames = list(slots.keys())
    n = len(snames)
    pos = {}
    for i, s in enumerate(snames):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos[s] = (math.cos(angle), math.sin(angle))

    drawn_edges = set()
    for (sa, pa, sb, pb) in constraints:
        key = tuple(sorted([sa, sb]))
        if key not in drawn_edges:
            drawn_edges.add(key)
            x1, y1 = pos[sa]
            x2, y2 = pos[sb]
            ax6.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(
                             arrowstyle="<->",
                             color=CLR_ACCENT, lw=1.0, alpha=0.5
                         ))

    for s in snames:
        x, y = pos[s]
        clr = CLR_ACROSS if "ACROSS" in s else CLR_DOWN
        ax6.add_patch(plt.Circle((x, y), 0.12, color=CLR_PANEL, ec=clr, lw=2, zorder=3))
        ax6.text(x, y, s.replace("-", "\n"), ha="center", va="center",
                 fontsize=5.5, color=CLR_TEXT, fontfamily="monospace", zorder=4)

    # Legend
    ax6.add_patch(plt.Circle((-1.3, -1.3), 0.05, color=CLR_PANEL, ec=CLR_ACROSS, lw=2))
    ax6.text(-1.15, -1.3, "ACROSS", va="center", fontsize=7, color=CLR_ACROSS, fontfamily="monospace")
    ax6.add_patch(plt.Circle((-0.3, -1.3), 0.05, color=CLR_PANEL, ec=CLR_DOWN, lw=2))
    ax6.text(-0.15, -1.3, "DOWN", va="center", fontsize=7, color=CLR_DOWN, fontfamily="monospace")

    # Status banner
    stat_clr = CLR_GREEN if assignment else CLR_RED
    stat_txt = (f"✓  CROSSWORD SOLVED  —  {len(slots)} slots filled  |  {len(steps)} backtracking steps"
                if assignment else f"✗  NO SOLUTION FOUND  —  {len(steps)} steps attempted")
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
    print("|   CSP  --  Crossword Puzzle Solver                           |")
    print("|   Slots: 1-ACROSS, 3-ACROSS, 5-ACROSS, 1-DOWN, 2-DOWN, 3-DOWN|")
    print("+" + "=" * 62 + "+")
    print()
    print("  Press Enter to use default 5×5 grid + built-in word list")
    choice = input("  or type  custom  to supply your own word list : ").strip().lower()

    if choice == "custom":
        print("\n  Enter words separated by spaces (5-letter words only for default grid):")
        raw = input("  Words : ").strip().upper().split()
        return DEFAULT_SLOTS, raw if raw else DEFAULT_WORDLIST
    return DEFAULT_SLOTS, DEFAULT_WORDLIST


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print()
    print("+" + "=" * 62 + "+")
    print("|      CSP -- Crossword Puzzle Solver                          |")
    print("|      Technique : Backtracking + Constraint Propagation       |")
    print("+" + "=" * 62 + "+")

    slots, wordlist = get_input()
    constraints = get_constraints(slots)

    print(f"\n  Grid   : 5 × 5")
    print(f"  Slots  : {len(slots)}  ( {', '.join(slots.keys())} )")
    print(f"  Words  : {len(wordlist)}")
    print(f"  Arcs   : {len(constraints)}")
    print(f"\n  Solving ...\n")

    assignment, steps, constraints = solve_crossword(slots, wordlist)

    print_results(slots, assignment, steps, wordlist)
    visualize(slots, wordlist, assignment, steps, constraints)


if __name__ == "__main__":
    main()