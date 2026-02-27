import random
import copy

# ----------------------------
# GOAL STATE
# ----------------------------
GOAL = [[1,2,3],
        [8,0,4],
        [7,6,5]]

# ----------------------------
# PRINT FUNCTION
# ----------------------------
def print_state(state):
    for row in state:
        print(row)
    print()

# ----------------------------
# COUNT INVERSIONS
# ----------------------------
def count_inversions(arr):
    inv_count = 0
    arr = [x for x in arr if x != 0]  # ignore blank
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count

# ----------------------------
# GENERATE RANDOM SOLVABLE PUZZLE
# ----------------------------
def generate_puzzle():
    while True:
        puzzle = list(range(9))
        random.shuffle(puzzle)
        if count_inversions(puzzle) % 2 == 0:
            return [puzzle[i:i+3] for i in range(0, 9, 3)]

# ----------------------------
# MANHATTAN DISTANCE HEURISTIC
# ----------------------------
def heuristic(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                for x in range(3):
                    for y in range(3):
                        if GOAL[x][y] == state[i][j]:
                            distance += abs(x-i) + abs(y-j)
    return distance

# ----------------------------
# FIND BLANK POSITION
# ----------------------------
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# ----------------------------
# GENERATE NEIGHBORS
# ----------------------------
def get_neighbors(state):
    neighbors = []
    x, y = find_blank(state)

    moves = [(1,0), (-1,0), (0,1), (0,-1)]  # down, up, right, left

    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = copy.deepcopy(state)
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)

    return neighbors

# ----------------------------
# HILL CLIMBING ALGORITHM
# ----------------------------
def hill_climbing(initial):
    current = initial
    steps = 0

    while True:
        print("Step:", steps)
        print_state(current)

        current_h = heuristic(current)

        if current == GOAL:
            print("Goal reached!")
            return current

        neighbors = get_neighbors(current)

        best = current
        best_h = current_h

        for neighbor in neighbors:
            h = heuristic(neighbor)
            if h < best_h:
                best = neighbor
                best_h = h

        if best_h >= current_h:
            print("Reached Local Optimum (Hill Climbing Stopped)")
            return current

        current = best
        steps += 1


# ----------------------------
# MAIN PROGRAM
# ----------------------------
if __name__ == "__main__":

    initial_state = generate_puzzle()

    print("Initial Random Solvable State:")
    initial = [[2,8,3],
           [1,6,4],
           [7,0,5]]
    
    print_state(initial_state)

    result = hill_climbing(initial_state)

    print("Final State:")
    print_state(result)