# ////// TIC TAC TOE
# Draw the board
board = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']

def print_board():
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])
        if i < 6:
            print("--+---+--")

# Win check condition
def check_win(player):
    wins = [
        [0,1,2], [3,4,5], [6,7,8],  # rows
        [0,3,6], [1,4,7], [2,5,8],  # columns
        [0,4,8], [2,4,6]            # diagonals
    ]
    for w in wins:
        if board[w[0]] == player and board[w[1]] == player and board[w[2]] == player:
            return True
    return False

def is_draw():
    return ' ' not in board

current_player = 'X'

while True:
    print_board()
    pos = int(input(f"Player {current_player}, enter position (1-9): ")) - 1

    if board[pos] != ' ':
        print("Invalid move!")
        continue

    board[pos] = current_player

    if check_win(current_player):
        print_board()
        print(f"Player {current_player} wins!")
        break

    if is_draw():
        print_board()
        print("Draw!")
        break

    current_player = 'O' if current_player == 'X' else 'X'

# ////// N QUEEN
# N Queens Problem using Backtracking
def solve_n_queens(n):
    board = [[0]*n for _ in range(n)]

    def is_safe(row, col):
        # Check row
        for i in range(col):
            if board[row][i] == 1:
                return False

        # Check upper left diagonal
        i, j = row, col
        while i >= 0 and j >= 0:
            if board[i][j] == 1:
                return False
            i -= 1
            j -= 1

         # Check lower left diagonal
        i, j = row, col
        while i < n and j >= 0:
            if board[i][j] == 1:
                return False
            i += 1
            j -= 1

        return True

    def solve(col):
        if col >= n:
            return True

        for row in range(n):
            if is_safe(row, col):
                board[row][col] = 1
                if solve(col + 1):
                    return True
                board[row][col] = 0

        return False

    if solve(0):
        for row in board:
            print(row)
    else:
        print("No solution")

solve_n_queens(4)


# ////// MAGIC SQUARE
# Magic Square for Odd n using Siamese Method
def magic_square(n):
    if n % 2 == 0:
        print("Magic square works only for odd n")
        return

    magic = [[0]*n for _ in range(n)]

    i = 0
    j = n // 2

    for num in range(1, n*n + 1):
        magic[i][j] = num
        new_i = (i - 1) % n
        new_j = (j + 1) % n

        if magic[new_i][new_j]:
            i = (i + 1) % n
        else:
            i = new_i
            j = new_j

    for row in magic:
        print(row)

# Example
magic_square(3)
