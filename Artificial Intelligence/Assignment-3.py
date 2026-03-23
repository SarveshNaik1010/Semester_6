"""
=============================================================
  Hill Climbing Technique to Solve the 8-Puzzle Problem
  Heuristic : Manhattan Distance
=============================================================
"""

import os
import sys
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Save image next to THIS script (works on Windows / Linux / Mac) ──
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "hill_climbing_8puzzle_visualization.png")

MAX_ITERATIONS = 100

# ── Colour palette ──
CLR_BG        = "#1A1A2E"
CLR_TILE      = "#16213E"
CLR_TILE_EDGE = "#0F3460"
CLR_NUM       = "#E94560"
CLR_BLANK     = "#0F3460"
CLR_GOAL_TILE = "#1B4332"
CLR_GOAL_EDGE = "#52B788"
CLR_GOAL_NUM  = "#B7E4C7"
CLR_HEADER    = "#E94560"
CLR_ARROW     = "#E94560"

# =====================================================================
#  USER INPUT
# =====================================================================

def _read_row(prompt, used):
    """Read and validate one row of 3 unique numbers (0-8)."""
    while True:
        try:
            raw  = input(prompt).strip()
            nums = list(map(int, raw.split()))
            if len(nums) != 3:
                print(f"  [!] Need exactly 3 numbers -- you gave {len(nums)}. Try again.")
                continue
            if any(n < 0 or n > 8 for n in nums):
                print("  [!] Each number must be between 0 and 8. Try again.")
                continue
            overlap = set(nums) & used
            if overlap:
                print(f"  [!] Number(s) {sorted(overlap)} already used. Try again.")
                continue
            return nums
        except ValueError:
            print("  [!] Please enter integers only.")


def get_puzzle_input(label="INITIAL"):
    """Prompt the user to enter a full 3x3 puzzle state row by row."""
    print()
    print("+" + "=" * 62 + "+")
    print(f"|   Enter {label:<8} State  (use 0 for the blank tile)         |")
    print("|   Type 3 numbers per row, separated by spaces.               |")
    print("|   Example ->  Row 1 : 1 2 3                                  |")
    print("+" + "=" * 62 + "+")

    state = []
    used  = set()
    for i in range(1, 4):
        row = _read_row(f"  Row {i} : ", used)
        state.append(row)
        used.update(row)

    print(f"  OK  {label} state accepted!\n")
    return state


def get_goal_choice():
    """Let user pick default goal or enter a custom one."""
    print("-" * 64)
    print("  GOAL STATE -- press Enter to use default [ 1 2 3 / 4 5 6 / 7 8 _ ]")
    choice = input("  or type  custom  to enter your own : ").strip().lower()
    print()
    if choice == "custom":
        return get_puzzle_input("GOAL")
    return [[1, 2, 3], [4, 5, 6], [7, 8, 0]]


# =====================================================================
#  PUZZLE UTILITIES
# =====================================================================

def find_blank(state):
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                return r, c


def manhattan_distance(state, goal):
    goal_pos = {goal[r][c]: (r, c) for r in range(3) for c in range(3)}
    dist = 0
    for r in range(3):
        for c in range(3):
            t = state[r][c]
            if t != 0:
                gr, gc = goal_pos[t]
                dist += abs(r - gr) + abs(c - gc)
    return dist


def get_neighbors(state):
    moves = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
    br, bc = find_blank(state)
    result = []
    for dr, dc, label in moves:
        nr, nc = br + dr, bc + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            ns = copy.deepcopy(state)
            ns[br][bc], ns[nr][nc] = ns[nr][nc], ns[br][bc]
            result.append((ns, label))
    return result


def as_tuple(state):
    return tuple(state[r][c] for r in range(3) for c in range(3))


# =====================================================================
#  HILL CLIMBING
# =====================================================================

def hill_climbing(initial, goal):
    current    = copy.deepcopy(initial)
    path       = [copy.deepcopy(current)]
    heuristics = [manhattan_distance(current, goal)]
    moves      = ["START"]
    visited    = {as_tuple(current)}

    for _ in range(MAX_ITERATIONS):
        if heuristics[-1] == 0:
            return path, heuristics, moves, "SOLVED"

        best_state, best_h, best_move = None, heuristics[-1], None

        for ns, mv in get_neighbors(current):
            if as_tuple(ns) in visited:
                continue
            h = manhattan_distance(ns, goal)
            if h < best_h:
                best_h, best_state, best_move = h, ns, mv

        if best_state is None:
            status = "SOLVED" if heuristics[-1] == 0 else "LOCAL_MINIMUM"
            return path, heuristics, moves, status

        visited.add(as_tuple(best_state))
        current = best_state
        path.append(copy.deepcopy(current))
        heuristics.append(best_h)
        moves.append(best_move)

    return path, heuristics, moves, "MAX_ITER"


# =====================================================================
#  TERMINAL HEURISTICS TABLE
# =====================================================================

def print_heuristics_table(path, heuristics, moves):
    col_w = [6, 14, 27, 12, 18]

    def cell(v, w):
        return " " + str(v).center(w - 2) + " "

    def row(*cells):
        parts = [cell(v, w) for v, w in zip(cells, col_w)]
        return "|" + "|".join(parts) + "|"

    def hline(lc="+", rc="+", mc="+", fc="-"):
        return lc + mc.join(fc * w for w in col_w) + rc

    total = sum(col_w) + len(col_w) + 1

    print()
    print(" " + "=" * (total - 2))
    print(" " + "  HILL CLIMBING -- HEURISTICS TABLE (Manhattan Distance)  ".center(total - 2))
    print(" " + "=" * (total - 2))
    print()
    print(" " + hline("+", "+", "+", "="))
    print(" " + row("Step", "Move", "State (Flat)", "H(n)", "Remark"))
    print(" " + hline("+", "+", "+", "="))

    prev_h = None
    for i, (state, h, mv) in enumerate(zip(path, heuristics, moves)):
        flat = " ".join(
            str(state[r][c]) if state[r][c] != 0 else "_"
            for r in range(3) for c in range(3)
        )
        if prev_h is None:
            remark = "Initial State"
        elif h == 0:
            remark = "** GOAL REACHED!"
        elif h < prev_h:
            remark = f"Better ({prev_h} -> {h})"
        else:
            remark = "No Improvement"

        print(" " + row(i, mv, flat, h, remark))
        if i < len(path) - 1:
            print(" " + hline())
        prev_h = h

    print(" " + hline("+", "+", "+", "="))
    print()
    print(f"  Total Steps : {len(path) - 1}")
    print(f"  Final H(n)  : {heuristics[-1]}")
    stat = "GOAL STATE REACHED" if heuristics[-1] == 0 else "LOCAL MINIMUM -- Stuck"
    print(f"  Status      : {stat}")
    print()


# =====================================================================
#  DRAW ONE PUZZLE GRID
# =====================================================================

def draw_puzzle(ax, state, title, step_num, h_val, goal,
                is_goal=False, is_initial=False):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(CLR_BG)

    border = CLR_GOAL_EDGE if is_goal else (CLR_HEADER if is_initial else CLR_TILE_EDGE)
    lw     = 3 if (is_goal or is_initial) else 1.5
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border)
        spine.set_linewidth(lw)

    goal_pos = {goal[r][c]: (r, c) for r in range(3) for c in range(3)}

    for r in range(3):
        for c in range(3):
            tile = state[r][c]
            x, y = c, 2 - r

            if tile == 0:
                face, edge, num = CLR_BLANK, CLR_TILE_EDGE, ""
            else:
                correct = (goal_pos[tile] == (r, c))
                if is_goal:
                    face, edge = CLR_GOAL_TILE, CLR_GOAL_EDGE
                elif correct:
                    face, edge = "#1B3A4B", "#52B788"
                else:
                    face, edge = CLR_TILE, CLR_TILE_EDGE
                num = str(tile)

            ax.add_patch(FancyBboxPatch(
                (x + 0.07, y + 0.07), 0.86, 0.86,
                boxstyle="round,pad=0.04", linewidth=1.5,
                edgecolor=edge, facecolor=face, zorder=2
            ))

            if num:
                clr = (CLR_GOAL_NUM if is_goal
                       else ("#52B788" if goal_pos.get(tile) == (r, c)
                             else CLR_NUM))
                ax.text(x + 0.5, y + 0.5, num,
                        ha="center", va="center",
                        fontsize=20, fontweight="bold",
                        color=clr, zorder=3, fontfamily="monospace")

    badge = CLR_GOAL_EDGE if is_goal else CLR_HEADER
    ax.text(1.5,  3.22, title,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=badge, fontfamily="monospace")
    ax.text(1.5, -0.30,
            "H = 0  GOAL" if h_val == 0 else f"H(n) = {h_val}",
            ha="center", va="center",
            fontsize=8.5, color="#A0A0C0", fontfamily="monospace")
    ax.text(1.5, -0.60, f"Step {step_num}",
            ha="center", va="center",
            fontsize=7.5, color="#606080", fontfamily="monospace")


# =====================================================================
#  DRAW ALL PUZZLES
# =====================================================================

def draw_all_puzzles(path, heuristics, moves, status, goal):
    n    = len(path)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 2.8 + 1.5, rows * 3.6 + 2.2),
                     facecolor=CLR_BG)
    fig.suptitle(
        "Hill Climbing -- 8-Puzzle Problem\n"
        "Heuristic: Manhattan Distance   |   0 = Blank Tile",
        color=CLR_HEADER, fontsize=14, fontweight="bold",
        fontfamily="monospace", y=0.98
    )

    gs = gridspec.GridSpec(rows, cols, figure=fig,
                           hspace=0.75, wspace=0.45,
                           left=0.04, right=0.96, top=0.90, bottom=0.06)

    for idx in range(n):
        r, c = divmod(idx, cols)
        ax   = fig.add_subplot(gs[r, c])
        ax.set_facecolor(CLR_BG)

        draw_puzzle(
            ax, path[idx],
            title      = f"[{'INITIAL' if idx == 0 else moves[idx]}]",
            step_num   = idx,
            h_val      = heuristics[idx],
            goal       = goal,
            is_goal    = (heuristics[idx] == 0),
            is_initial = (idx == 0),
        )

        # horizontal arrow between adjacent nodes on same row
        if idx < n - 1 and (idx % cols) < cols - 1:
            pos   = ax.get_position()
            mid_y = (pos.y0 + pos.y1) / 2 + 0.03
            fig.add_artist(FancyArrowPatch(
                (pos.x1 + 0.005, mid_y), (pos.x1 + 0.035, mid_y),
                arrowstyle="->", color=CLR_ARROW,
                linewidth=1.5, mutation_scale=12,
                transform=fig.transFigure, zorder=10
            ))

    # Status banner at bottom
    colors = {
        "SOLVED":        ("#B7E4C7", "#1B4332"),
        "LOCAL_MINIMUM": ("#FFB3B3", "#4B1818"),
        "MAX_ITER":      ("#FFE5B3", "#4B3A18"),
    }
    tc, bc = colors.get(status, ("#FFFFFF", "#333333"))
    labels = {
        "SOLVED":        f"SOLVED in {n-1} step(s)  --  Goal state reached!",
        "LOCAL_MINIMUM": f"LOCAL MINIMUM after {n-1} step(s) -- No better neighbour found.",
        "MAX_ITER":      f"MAX ITERATIONS ({MAX_ITERATIONS}) reached without solution.",
    }
    fig.text(0.5, 0.013, labels.get(status, status),
             ha="center", va="bottom", fontsize=10, fontweight="bold",
             fontfamily="monospace", color=tc,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=bc, edgecolor=tc, lw=1.5))

    # ── Save to the SAME folder as this script ──
    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight", facecolor=CLR_BG)
    print(f"\n  [INFO] Visualization saved  ->  {OUTPUT_FILE}")
    plt.show()   # opens the window on Windows / Mac / Linux


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print()
    print("+" + "=" * 62 + "+")
    print("|      Hill Climbing -- 8-Puzzle Problem Solver                |")
    print("|      Heuristic : Manhattan Distance                          |")
    print("+" + "=" * 62 + "+")

    initial = get_puzzle_input("INITIAL")
    goal    = get_goal_choice()

    print("  Your Initial State :")
    for r in initial:
        print(f"    {r[0]}  {r[1]}  {r[2]}")
    print("\n  Goal State :")
    for r in goal:
        print(f"    {r[0]}  {r[1]}  {r[2]}")
    print(f"\n  Starting H(n) = {manhattan_distance(initial, goal)}\n")

    path, heuristics, moves, status = hill_climbing(initial, goal)

    print_heuristics_table(path, heuristics, moves)
    draw_all_puzzles(path, heuristics, moves, status, goal)


if __name__ == "__main__":
    main()