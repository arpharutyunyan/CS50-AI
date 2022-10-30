"""
Tic Tac Toe Player

Result
    https://www.youtube.com/watch?v=Ln4yQe6bSTQ
    
"""

import math
from copy import deepcopy

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
    steps_count = 0
    for row in board:
        for column in row:
            if column == X or column == O:
                steps_count += 1
    if steps_count % 2 == 0:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    all_possible_actions = set()
    for row in range(len(board[0])):
        for column in range(len(board)):
            if board[row][column] == EMPTY:
                all_possible_actions.add((row, column))
    return all_possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action is None:
        raise Exception('Not a valid action')

    copy_board = deepcopy(board)
    copy_board[action[0]][action[1]] = player(copy_board)
    return copy_board


def winner(board):
    """
    Returns the winner of the game, if there is on   e.
    """
    # check horizontally
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2]:
            return board[row][0]
    #     check vertically
    for column in range(3):
        if board[0][column] == board[1][column] == board[2][column]:
            return board[0][column]
    # check diagonally
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # there is one winner
    if winner(board)!= EMPTY:
        return True
    # the game not over
    for row in board:
        for column in row:
            if column == EMPTY:
                return False
    #  game over (tie)
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def max_value(board):
    if terminal(board):
        return utility(board)
    v = -100
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
        if v == 1:
            return v
    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    v = 100
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
        if v == -1:
            return v
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        high_value = -1
        move = None
        for action in actions(board):
            v = min_value(result(board, action))
            if v == 1:
                return action
            if v > high_value:
                move = action
                high_value = v
        return move

    if player(board) == O:
        move = None
        low_value = 1
        for action in actions(board):
            v = max_value(result(board, action))
            if v == -1:
                return action
            if v < low_value:
                move = action
                low_value = v
        return move
