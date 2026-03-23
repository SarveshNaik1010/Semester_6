from collections import deque
import uuid
import copy

# =====================================================
# BUILD DFS TREE
# =====================================================
def build_dfs_tree():
    nodes = {}
    root_id = "root"

    nodes[root_id] = {
        "state": (0, 0), "depth": 0, "parent": None,
        "label": "", "status": "start", "children": [],
        "visit_num": 0
    }

    goal_id = [None]
    visit_counter = [0]

    def dfs(node_id, visited, depth=0):
        if goal_id[0] is not None:
            return

        state = nodes[node_id]["state"]
        visited = visited | {state}

        for nxt, lbl in successors(state):
            if goal_id[0] is not None:
                return

            cid = str(uuid.uuid4())[:8]

            if nxt in visited:
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
                    return

                dfs(cid, visited, depth+1)

    dfs(root_id, set(), 0)
    return nodes, root_id, goal_id[0]


# =====================================================
# BUILD BFS TREE
# =====================================================
def build_bfs_tree():
    nodes = {}
    root_id = "root"

    nodes[root_id] = {
        "state": (0, 0), "depth": 0, "parent": None,
        "label": "", "status": "start", "children": [],
        "visit_num": 0
    }

    queue = deque([root_id])
    visited = set([(0, 0)])
    visit_counter = 0
    goal_id = None

    while queue:
        current_id = queue.popleft()
        current_node = nodes[current_id]

        for nxt, lbl in successors(current_node["state"]):
            cid = str(uuid.uuid4())[:8]

            if nxt in visited:
                nodes[cid] = {
                    "state": nxt,
                    "depth": current_node["depth"] + 1,
                    "parent": current_id,
                    "label": lbl,
                    "status": "pruned",
                    "children": [],
                    "visit_num": None
                }
                nodes[current_id]["children"].append(cid)

            else:
                visit_counter += 1
                visited.add(nxt)

                status = "goal" if is_goal(nxt) else "new"

                nodes[cid] = {
                    "state": nxt,
                    "depth": current_node["depth"] + 1,
                    "parent": current_id,
                    "label": lbl,
                    "status": status,
                    "children": [],
                    "visit_num": visit_counter
                }

                nodes[current_id]["children"].append(cid)
                queue.append(cid)

                if status == "goal":
                    goal_id = cid
                    queue.clear()
                    break

    return nodes, root_id, goal_id


# =====================================================
# MARK SOLUTION PATH
# =====================================================
def mark_solution(nodes, goal_id):
    sol_ids = set()
    nid = goal_id

    while nid is not None:
        sol_ids.add(nid)
        nid = nodes[nid]["parent"]

    for nid in sol_ids:
        if nodes[nid]["status"] not in ("start", "goal"):
            nodes[nid]["status"] = "solution"

    return sol_ids


# =====================================================
# DRAW FUNCTION (REUSED)
# =====================================================
def draw_tree(nodes, root_id, goal_id, title, filename, show_backtrack=False):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ---- Layout ----
    def assign_x(nid, counter):
        ch = nodes[nid]["children"]
        if not ch:
            nodes[nid]["x"] = counter[0]
            counter[0] += 1
            return
        for c in ch:
            assign_x(c, counter)
        xs = [nodes[c]["x"] for c in ch]
        nodes[nid]["x"] = (min(xs) + max(xs)) / 2

    assign_x(root_id, [0])

    for nid, nd in nodes.items():
        nd["y"] = -nd["depth"] * 1.8

    # ---- Draw ----
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0f0f1a")

    for nid, nd in nodes.items():
        if nd["parent"] is None:
            continue
        par = nodes[nd["parent"]]

        ax.plot([par["x"], nd["x"]],
                [par["y"], nd["y"]], "w-", lw=1)

    for nid, nd in nodes.items():
        x, y = nd["x"], nd["y"]

        color = {
            "start": "orange",
            "goal": "green",
            "solution": "yellow",
            "new": "blue",
            "pruned": "red"
        }.get(nd["status"], "blue")

        ax.scatter(x, y, c=color, s=600)
        ax.text(x, y, str(nd["state"]),
                ha="center", va="center", color="white", fontsize=8)

    ax.set_title(title, color="white")
    ax.axis("off")

    plt.savefig(filename)
    print(f"Saved → {filename}")
    plt.close()


# =====================================================
# 🚀 RUN BOTH
# =====================================================
dfs_nodes, dfs_root, dfs_goal = build_dfs_tree()
dfs_sol = mark_solution(dfs_nodes, dfs_goal)

draw_tree(dfs_nodes, dfs_root, dfs_goal,
          "DFS Tree", "dfs_tree.png", show_backtrack=True)


bfs_nodes, bfs_root, bfs_goal = build_bfs_tree()
bfs_sol = mark_solution(bfs_nodes, bfs_goal)

draw_tree(bfs_nodes, bfs_root, bfs_goal,
          "BFS Tree", "bfs_tree.png", show_backtrack=False)