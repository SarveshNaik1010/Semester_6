from collections import deque

goal = "012345678"

# Moves: UP, DOWN, LEFT, RIGHT
moves = ["UP", "DOWN", "LEFT", "RIGHT"]
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


def solve_puzzle(start):
    queue = deque([start])
    parent = {start: None}
    move_taken = {}

    while queue:
        curr = queue.popleft()

        if curr == goal:
            break

        zero_pos = curr.index('0')
        x, y = divmod(zero_pos, 3)

        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]

            if 0 <= nx < 3 and 0 <= ny < 3:
                new_pos = nx * 3 + ny

                next_state = list(curr)

                # swap
                next_state[zero_pos], next_state[new_pos] = (
                    next_state[new_pos],
                    next_state[zero_pos],
                )

                next_state = ''.join(next_state)

                if next_state not in parent:
                    parent[next_state] = curr
                    move_taken[next_state] = moves[i]
                    queue.append(next_state)

    # Backtrack path
    path = []
    curr = goal

    while curr != start:
        path.append(move_taken[curr])
        curr = parent[curr]

    path.reverse()
    return path


# -------- Input -------
k = int(input())   # should be 3 for 8-puzzle
start = ""

for _ in range(k * k):
    start += input().strip()


# -------- Solve -------
result = solve_puzzle(start)


# -------- Output -------
print(len(result))
for move in result:
    print(move)