from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

# ---------------- Node Class ----------------
class Node:
    def __init__(self, state):
        self.state = state
        self.left = None
        self.right = None


# ---------------- Generate Moves ----------------
def generate_moves(x, y, capA, capB):
    moves = []

    moves.append((capA, y))   # Fill A
    moves.append((x, capB))   # Fill B
    moves.append((0, y))      # Empty A
    moves.append((x, 0))      # Empty B

    pour = min(x, capB - y)
    moves.append((x - pour, y + pour))

    pour = min(y, capA - x)
    moves.append((x + pour, y - pour))

    return moves


# ---------------- Build Binary Tree ----------------
def build_binary_tree(capA, capB, max_nodes=15):
    root = Node((0, 0))
    queue = deque([root])
    visited = set([(0, 0)])

    while queue and len(visited) < max_nodes:
        current = queue.popleft()
        x, y = current.state

        children = []

        for move in generate_moves(x, y, capA, capB):
            if move not in visited:
                visited.add(move)
                child_node = Node(move)
                children.append(child_node)
                queue.append(child_node)

        # Strict Binary: only 2 children
        if len(children) > 0:
            current.left = children[0]
        if len(children) > 1:
            current.right = children[1]

    return root


# ---------------- Convert Tree to Graph ----------------
def add_edges(G, node):
    if node is None:
        return

    if node.left:
        G.add_edge(str(node.state), str(node.left.state))
        add_edges(G, node.left)

    if node.right:
        G.add_edge(str(node.state), str(node.right.state))
        add_edges(G, node.right)


# ---------------- Visualize Tree ----------------
def visualize_tree(root):
    G = nx.DiGraph()
    add_edges(G, root)

    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos,
            with_labels=True,
            node_size=2500,
            font_size=10)

    plt.title("Binary Tree Representation of Water Jug")
    plt.show()


# ---------------- Driver ----------------
capA = int(input("Enter capacity of Jug A: "))
capB = int(input("Enter capacity of Jug B: "))

root = build_binary_tree(capA, capB)
visualize_tree(root)