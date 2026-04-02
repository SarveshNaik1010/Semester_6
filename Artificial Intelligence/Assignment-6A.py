"""
=============================================================
  Constraint Satisfaction Problem (CSP)
  Problem : CRYPTARITHMETIC
  Example : SEND + MORE = MONEY
=============================================================
"""

import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "csp_cryptarithmetic.png")

# ── Colour palette ──
CLR_BG        = "#0D1117"
CLR_PANEL     = "#161B22"
CLR_ACCENT    = "#58A6FF"
CLR_GREEN     = "#3FB950"
CLR_RED       = "#F85149"
CLR_YELLOW    = "#D29922"
CLR_TEXT      = "#C9D1D9"
CLR_DIM       = "#8B949E"
CLR_BORDER    = "#30363D"
CLR_HEADER    = "#58A6FF"


# =====================================================================
#  CRYPTARITHMETIC SOLVER (CSP with Backtracking)
# =====================================================================

def solve_cryptarithmetic(word1, word2, word3):
    """
    Solve: word1 + word2 = word3
    Returns all solutions and the search tree nodes (for visualization).
    """
    letters = list(set(word1 + word2 + word3))
    leading = {word1[0], word2[0], word3[0]}
    digits  = list(range(10))
    solutions   = []
    tree_nodes  = []   # (assignment_so_far, depth, result)
    node_count  = [0]

    def word_val(word, assign):
        val = 0
        for ch in word:
            val = val * 10 + assign[ch]
        return val

    def backtrack(assign, remaining, depth):
        node_count[0] += 1
        snapshot = dict(assign)
        tree_nodes.append((snapshot, depth, None))  # None = pending

        if not remaining:
            v1 = word_val(word1, assign)
            v2 = word_val(word2, assign)
            v3 = word_val(word3, assign)
            ok = (v1 + v2 == v3)
            tree_nodes[-1] = (snapshot, depth, ok)
            if ok:
                solutions.append((dict(assign), v1, v2, v3))
            return

        letter = remaining[0]
        used   = set(assign.values())

        for d in digits:
            if d in used:
                continue
            if d == 0 and letter in leading:
                continue
            assign[letter] = d
            backtrack(assign, remaining[1:], depth + 1)
            del assign[letter]

    backtrack({}, letters, 0)
    return solutions, tree_nodes, node_count[0]


# =====================================================================
#  TERMINAL OUTPUT
# =====================================================================

def print_results(word1, word2, word3, solutions, node_count):
    # Fixed-width box — wide enough for any input
    W = 62  # inner content width (between the | pipes)

    def pad(text):
        """Left-align text and pad to exactly W chars."""
        return text[:W].ljust(W)

    def bar(c="="):
        return "+" + c * W + "+"

    problem_line  = f"Problem :  {word1} + {word2} = {word3}"
    vars_line     = f"Variables  : {', '.join(sorted(set(word1+word2+word3)))}"
    constr_line   = f"Constraint : {word1} + {word2} = {word3}  (no leading zeros)"
    nodes_line    = f"Nodes explored : {node_count}"
    solns_line    = f"Solutions found: {len(solutions)}"
    status_line   = f"Status         : {'SOLVED ✓' if solutions else 'NO SOLUTION FOUND ✗'}"

    print()
    print(bar())
    print(f"|{'CSP -- CRYPTARITHMETIC SOLVER':^{W}}|")
    print(bar())
    print(f"|  {pad(problem_line)}|")
    print(bar("-"))
    print(f"|  {pad(vars_line)}|")
    print(f"|  {pad(constr_line)}|")
    print(bar("-"))
    print(f"|  {pad(nodes_line)}|")
    print(f"|  {pad(solns_line)}|")
    print(f"|  {pad(status_line)}|")
    print(bar())

    if not solutions:
        return

    print(f"\n  {'Step':<6} {'Assignment':<40} {'Equation':<30} {'Status'}")
    print("  " + "=" * 90)
    for i, (assign, v1, v2, v3) in enumerate(solutions, 1):
        asgn_str = "  ".join(f"{k}={v}" for k, v in sorted(assign.items()))
        eq_str   = f"{v1} + {v2} = {v3}"
        print(f"  {i:<6} {asgn_str:<40} {eq_str:<30} ✓ VALID")
    print()


# =====================================================================
#  VISUALIZATION
# =====================================================================

def draw_letter_box(ax, x, y, letter, digit, size=0.7, highlight=False):
    color = CLR_GREEN if highlight else CLR_PANEL
    edge  = CLR_GREEN if highlight else CLR_ACCENT
    ax.add_patch(FancyBboxPatch(
        (x, y), size, size,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=edge, linewidth=1.5, zorder=3
    ))
    ax.text(x + size/2, y + size*0.72, letter,
            ha="center", va="center", fontsize=9,
            color=CLR_DIM, fontfamily="monospace", zorder=4)
    ax.text(x + size/2, y + size*0.28, str(digit) if digit is not None else "?",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=CLR_TEXT if digit is not None else CLR_DIM,
            fontfamily="monospace", zorder=4)


def draw_word(ax, word, assignment, start_x, y, label=None):
    for i, ch in enumerate(word):
        d = assignment.get(ch)
        draw_letter_box(ax, start_x + i * 0.85, y, ch, d)
    if label:
        ax.text(start_x - 0.5, y + 0.35, label,
                ha="center", va="center", fontsize=11,
                color=CLR_ACCENT, fontfamily="monospace", fontweight="bold")


def visualize(word1, word2, word3, solutions, tree_nodes, node_count):
    fig = plt.figure(figsize=(18, 12), facecolor=CLR_BG)
    fig.suptitle(
        f"CSP  —  Cryptarithmetic   [{word1} + {word2} = {word3}]\n"
        "Technique: Backtracking with Constraint Propagation",
        color=CLR_HEADER, fontsize=15, fontweight="bold",
        fontfamily="monospace", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.04, right=0.97, top=0.90, bottom=0.06)

    # ── Panel 1: Problem Statement ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(CLR_PANEL)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 4)
    ax1.axis("off")

    best = solutions[0] if solutions else {}
    assign = best[0] if best else {}

    max_len = max(len(word1), len(word2), len(word3))
    ox = (5 - max_len * 0.85) / 2

    draw_word(ax1, word1, assign, ox + (max_len - len(word1)) * 0.85, 3.0)
    draw_word(ax1, word2, assign, ox + (max_len - len(word2)) * 0.85, 2.1)
    ax1.plot([ox - 0.1, ox + max_len * 0.85], [1.9, 1.9], color=CLR_ACCENT, lw=1.5)
    ax1.text(ox - 0.3, 2.1, "+", color=CLR_ACCENT, fontsize=14, fontweight="bold",
             fontfamily="monospace")
    draw_word(ax1, word3, assign, ox + (max_len - len(word3)) * 0.85, 1.0)
    ax1.set_title("Problem Statement", color=CLR_ACCENT, fontsize=10,
                  fontfamily="monospace", pad=6)

    if solutions:
        v1, v2, v3 = solutions[0][1], solutions[0][2], solutions[0][3]
        ax1.text(2.5, 0.3, f"{v1} + {v2} = {v3}", ha="center",
                 color=CLR_GREEN, fontsize=11, fontfamily="monospace", fontweight="bold")

    # ── Panel 2: Variable – Digit Assignment Table ───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(CLR_PANEL)
    ax2.axis("off")
    ax2.set_title("Variable → Digit Assignment", color=CLR_ACCENT,
                  fontsize=10, fontfamily="monospace", pad=6)

    if solutions:
        assign_map = solutions[0][0]
        letters_sorted = sorted(assign_map.keys())
        col_labels = ["Letter", "Digit", "Role"]
        leading = {word1[0], word2[0], word3[0]}

        table_data = []
        for ltr in letters_sorted:
            role = "Leading" if ltr in leading else "Middle/End"
            table_data.append([ltr, str(assign_map[ltr]), role])

        tbl = ax2.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center", loc="center",
            bbox=[0.05, 0.05, 0.9, 0.85]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor(CLR_BG if r == 0 else CLR_PANEL)
            cell.set_edgecolor(CLR_BORDER)
            cell.set_text_props(
                color=CLR_ACCENT if r == 0 else CLR_TEXT,
                fontfamily="monospace",
                fontweight="bold" if r == 0 else "normal"
            )
    else:
        ax2.text(0.5, 0.5, "No Solution Found", ha="center", va="center",
                 color=CLR_RED, fontsize=12, fontfamily="monospace")

    # ── Panel 3: Stats & Constraints ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(CLR_PANEL)
    ax3.axis("off")
    ax3.set_title("CSP Statistics", color=CLR_ACCENT,
                  fontsize=10, fontfamily="monospace", pad=6)

    all_letters = sorted(set(word1 + word2 + word3))
    stats = [
        ("Variables",     str(len(all_letters))),
        ("Domain",        "{ 0 – 9 }"),
        ("Constraints",   "Alldiff + Sum"),
        ("Nodes Explored", str(node_count)),
        ("Solutions",     str(len(solutions))),
        ("Status",        "SOLVED ✓" if solutions else "NO SOL ✗"),
    ]

    for i, (label, val) in enumerate(stats):
        y_pos = 0.88 - i * 0.14
        color = CLR_GREEN if "SOLVED" in val else (CLR_RED if "NO SOL" in val else CLR_TEXT)
        ax3.text(0.05, y_pos, f"▸  {label}:", color=CLR_DIM,
                 fontsize=9, fontfamily="monospace", transform=ax3.transAxes)
        ax3.text(0.60, y_pos, val, color=color,
                 fontsize=9, fontfamily="monospace", fontweight="bold",
                 transform=ax3.transAxes)

    # ── Panel 4: Search Tree (partial, first 60 nodes) ──────────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(CLR_BG)
    ax4.axis("off")
    ax4.set_title("Backtracking Search Tree  (first nodes per depth level)",
                  color=CLR_ACCENT, fontsize=10, fontfamily="monospace", pad=6)

    # Group nodes by depth — show up to 8 per depth, up to depth 5
    depth_groups = {}
    for snap, depth, result in tree_nodes:
        if depth not in depth_groups:
            depth_groups[depth] = []
        if len(depth_groups[depth]) < 8:
            depth_groups[depth].append((snap, result))

    max_depth = min(max(depth_groups.keys()) if depth_groups else 0, 5)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.1, 1.1)

    node_positions = {}
    for depth in range(max_depth + 1):
        nodes = depth_groups.get(depth, [])
        n = len(nodes)
        for j, (snap, result) in enumerate(nodes):
            x = (j + 0.5) / max(n, 1)
            y = 1.0 - depth * (1.0 / max(max_depth, 1))

            if result is True:
                clr, ec = CLR_GREEN, CLR_GREEN
            elif result is False:
                clr, ec = "#2D1B1B", CLR_RED
            else:
                clr, ec = CLR_PANEL, CLR_ACCENT

            key = f"{depth}-{j}"
            node_positions[key] = (x, y)

            circle = plt.Circle((x, y), 0.025, color=clr, ec=ec, lw=1.2, zorder=3)
            ax4.add_patch(circle)

            label = ""
            if snap:
                last_key = list(snap.keys())[-1] if snap else ""
                last_val = snap.get(last_key, "")
                label = f"{last_key}={last_val}" if last_key else ""
            ax4.text(x, y - 0.055, label, ha="center", va="top",
                     fontsize=6, color=CLR_DIM, fontfamily="monospace")

            # Connect to parent
            if depth > 0 and depth_groups.get(depth - 1):
                parent_n = len(depth_groups[depth - 1])
                parent_j = min(j, parent_n - 1)
                px = (parent_j + 0.5) / max(parent_n, 1)
                py = 1.0 - (depth - 1) * (1.0 / max(max_depth, 1))
                ax4.plot([px, x], [py, y], color=CLR_BORDER, lw=0.6, zorder=1)

    # Depth labels
    for depth in range(max_depth + 1):
        y = 1.0 - depth * (1.0 / max(max_depth, 1))
        ax4.text(0.01, y, f"D{depth}", ha="left", va="center",
                 fontsize=7, color=CLR_DIM, fontfamily="monospace")

    # Legend
    for lbl, clr in [("Solution", CLR_GREEN), ("Failure", CLR_RED), ("Pending", CLR_ACCENT)]:
        circle = plt.Circle((0, 0), 0.01, color=clr)
    ax4.text(0.78, -0.07, "● Solution  ", color=CLR_GREEN, fontsize=8, fontfamily="monospace")
    ax4.text(0.87, -0.07, "● Failure  ", color=CLR_RED, fontsize=8, fontfamily="monospace")
    ax4.text(0.95, -0.07, "● Node", color=CLR_ACCENT, fontsize=8, fontfamily="monospace")

    # Status banner
    stat_clr = CLR_GREEN if solutions else CLR_RED
    stat_txt = (f"✓  SOLVED  —  {len(solutions)} solution(s) found   |   "
                f"{node_count} nodes explored via Backtracking")
    if not solutions:
        stat_txt = f"✗  NO SOLUTION FOUND  —  {node_count} nodes explored"
    fig.text(0.5, 0.013, stat_txt, ha="center", va="bottom",
             fontsize=10, fontweight="bold", fontfamily="monospace",
             color=stat_clr,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#0D2818" if solutions else "#2D1B1B",
                       edgecolor=stat_clr, lw=1.5))

    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight", facecolor=CLR_BG)
    print(f"\n  [INFO] Visualization saved  ->  {OUTPUT_FILE}")
    plt.show()


# =====================================================================
#  USER INPUT
# =====================================================================

def get_words():
    print()
    print("+" + "=" * 62 + "+")
    print("|   CSP  --  Cryptarithmetic Solver                            |")
    print("|   Solves :  WORD1 + WORD2 = WORD3                           |")
    print("+" + "=" * 62 + "+")
    print()
    print("  Press Enter to use default  [ SEND + MORE = MONEY ]")
    choice = input("  or type  custom  to enter your own words : ").strip().lower()

    if choice == "custom":
        print()
        word1 = input("  Enter WORD1 : ").strip().upper()
        word2 = input("  Enter WORD2 : ").strip().upper()
        word3 = input("  Enter WORD3 (=result) : ").strip().upper()
        # Validate only alpha
        for w in [word1, word2, word3]:
            if not w.isalpha():
                print("  [!] Only alphabetic characters allowed. Using default.")
                return "SEND", "MORE", "MONEY"
        return word1, word2, word3
    return "SEND", "MORE", "MONEY"


# =====================================================================
#  MAIN
# =====================================================================

def main():
    word1, word2, word3 = get_words()
    all_letters = set(word1 + word2 + word3)

    print(f"\n  Problem   : {word1} + {word2} = {word3}")
    print(f"  Variables : {', '.join(sorted(all_letters))}")
    print(f"  Solving via Backtracking CSP ...\n")

    solutions, tree_nodes, node_count = solve_cryptarithmetic(word1, word2, word3)

    print_results(word1, word2, word3, solutions, node_count)
    visualize(word1, word2, word3, solutions, tree_nodes, node_count)


if __name__ == "__main__":
    main()