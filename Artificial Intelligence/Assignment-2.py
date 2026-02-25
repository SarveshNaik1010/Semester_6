from collections import deque
import matplotlib.pyplot as plt

# ---------------- DFS ----------------
def dfs(x, y, capA, capB, target, visited, path):
    # Goal condition
    if x == target or y == target:
        path.append((x, y))
        return True

    # Avoid infinite loop
    if (x, y) in visited:
        return False

    visited.add((x, y))
    path.append((x, y))

    # 1. Fill Jug A
    if dfs(capA, y, capA, capB, target, visited, path):
        return True

    # 2. Fill Jug B
    if dfs(x, capB, capA, capB, target, visited, path):
        return True

    # 3. Empty Jug A
    if dfs(0, y, capA, capB, target, visited, path):
        return True

    # 4. Empty Jug B
    if dfs(x, 0, capA, capB, target, visited, path):
        return True

    # 5. Pour A -> B
    pour = min(x, capB - y)
    if dfs(x - pour, y + pour, capA, capB, target, visited, path):
        return True

    # 6. Pour B -> A
    pour = min(y, capA - x)
    if dfs(x + pour, y - pour, capA, capB, target, visited, path):
        return True

    path.pop()
    return False


# ---------------- BFS ----------------
def bfs(capA, capB, target):
    queue = deque()
    visited = set()

    queue.append((0, 0, []))
    visited.add((0, 0))

    while queue:
        x, y, path = queue.popleft()
        path = path + [(x, y)]

        if x == target or y == target:
            return path

        states = [
            (capA, y),     # Fill A
            (x, capB),     # Fill B
            (0, y),        # Empty A
            (x, 0),        # Empty B
        ]

        # Pour A -> B
        pour = min(x, capB - y)
        states.append((x - pour, y + pour))

        # Pour B -> A
        pour = min(y, capA - x)
        states.append((x + pour, y - pour))

        for state in states:
            if state not in visited:
                visited.add(state)
                queue.append((state[0], state[1], path))

    return None


# ---------------- Plotting ----------------
def plot_path(path, title):
    x_vals = [state[0] for state in path]
    y_vals = [state[1] for state in path]

    plt.figure(figsize=(6, 6))

    # Draw arrows between consecutive states
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        plt.arrow(
            x1, y1,
            x2 - x1, y2 - y1,
            length_includes_head=True,
            head_width=0.15,
            head_length=0.15,
            fc='black',
            ec='black'
        )

    # Plot points
    plt.plot(x_vals, y_vals, marker='o')

    # Label each state with step number
    for i, (x, y) in enumerate(path):
        plt.text(x + 0.05, y + 0.05, f"{i}:({x},{y})", fontsize=9)

    plt.xlabel("Jug A")
    plt.ylabel("Jug B")
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, max(x_vals) + 1)
    plt.ylim(0, max(y_vals) + 1)
    plt.show()



# ---------------- Driver Code ----------------
capA = int(input("Enter capacity of Jug A: "))
capB = int(input("Enter capacity of Jug B: "))
target = int(input("Enter target quantity: "))

# DFS Execution
visited = set()
dfs_path = []

if dfs(0, 0, capA, capB, target, visited, dfs_path):
    print("\nDFS Solution Path:")
    for state in dfs_path:
        print(state)
    plot_path(dfs_path, "DFS Water Jug State Space Path")
else:
    print("No DFS solution exists")

# BFS Execution
bfs_path = bfs(capA, capB, target)

if bfs_path:
    print("\nBFS Solution Path:")
    for state in bfs_path:
        print(state)
    plot_path(bfs_path, "BFS Water Jug Shortest Path")
else:
    print("No BFS solution exists")
