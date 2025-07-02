"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = sum(row.count(X) for row in board)
    o = sum(row.count(O) for row in board)
    return X if x == o or x == 0 else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.add((i,j))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Invalid move")

    i, j = action
    nuevo = [row.copy() for row in board]
    nuevo[i][j] = player(board)
    return nuevo


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    lines = [
        # Filas
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # Columnas
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # Diagonales
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]],
    ]
    for line in lines:
        if all(cell == X for cell in line):
            return X
        elif all(cell == O for cell in line):
            return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) is not None or all(cell != EMPTY for row in board for cell in row)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    actual_player = player(board)

    if actual_player == X:  # Maximizador
        val = -float("inf")
        mej = None
        for action in actions(board):
            nuevo = min_value(result(board, action))
            if nuevo > val:
                val = nuevo
                mej = action
                if val == 1:  # Atajo: si encontramos una victoria, retornar inmediatamente
                    return mej
        return mej
    else:  # Minimizador (O)
        val = float("inf")  # ← Corrección clave aquí
        mej = None
        for action in actions(board):
            nuevo = max_value(result(board, action))  # ← Evaluamos con max_value (no min_value)
            if nuevo < val:  # ← Buscamos el valor más pequeño
                val = nuevo
                mej = action
                if val == -1:  # Atajo: si bloqueamos una victoria de X, retornar
                    return mej
        return mej
    
def max_value(board):
    if terminal(board):
        return utility(board)
    value = -float("inf")
    for action in actions(board):
        value = max(value, min_value(result(board, action)))
        if value == 1:
            return value
    return value

def min_value(board):
    if terminal(board):
        return utility(board)
    value = float("inf")
    for action in actions(board):
        value = min(value, max_value(result(board, action)))
        if value == -1:
            return value
    return value
