import numpy as np
from typing import Optional, Tuple
from agents.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state


"""
this is a raw minimax agent without alpha-beta pruning
"""


def heuristic_basic(
        board: np.ndarray, player: BoardPiece
) -> np.int:
    """
    using similar code as what is used to check for a connect_n
    we can create a heuristic that assigns a higher value to a board
    where a player has more adjacent pieces and the highest value
    for when the player has a winning board
    """
    """
    this is the starting part of a better heuristic. 
    points that could be made better are that depth is not taken into account,
    which then makes the minimax agent not know that it could be losing if it 
    does not play a certain move, and just sees more adjecent pieces as better
    """
    from agents.connectn.common import CONNECT_N

    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    rating = 0  # starting rating (no connecting pieces)
    degree = 5  # weighting for the number connected (exponential)

    for i in range(rows):  # count vertical pieces
        for j in range(cols_edge):
            for k in range(CONNECT_N-1):  # count up to a winning board
                if np.all(board[i, j:j+CONNECT_N-k] == player):  # connected_n-k
                    rating += CONNECT_N ** degree

    for i in range(rows_edge):  # count horizontal pieces
        for j in range(cols):
            for k in range(CONNECT_N-1):  # count up to a winning board
                if np.all(board[i:i+CONNECT_N-k, j] == player):  # connected_n-k
                    rating += CONNECT_N ** degree

    for i in range(rows_edge):  # count diagonal pieces
        for j in range(cols_edge):
            for k in range(CONNECT_N-1):  # count up to a winning board
                block = board[i:i+CONNECT_N-k, j:j+CONNECT_N-k]  # connected_n-k
                if np.all(np.diag(block) == player):
                    rating += CONNECT_N ** degree
                if np.all(np.diag(block[::-1, :]) == player):
                    rating += CONNECT_N ** degree
    return rating


def heuristic_better(
        board: np.ndarray, player: BoardPiece,
) -> np.int:
    """
    taking our basic heuristic, let's make it a bit better
    """
    from agents.connectn.common import CONNECT_N

    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    rating = 0  # starting rating (no connecting pieces)
    degree = 5  # weighting for the number connected (exponential)

    for i in range(rows):  # count vertical pieces
        for j in range(cols_edge):
            for k in range(CONNECT_N-1):  # count up to a winning board
                if np.all(board[i, j:j+CONNECT_N-k] == player):  # connected_n-k
                    rating += CONNECT_N ** degree

    for i in range(rows_edge):  # count horizontal pieces
        for j in range(cols):
            for k in range(CONNECT_N-1):  # count up to a winning board
                if np.all(board[i:i+CONNECT_N-k, j] == player):  # connected_n-k
                    rating += CONNECT_N ** degree

    for i in range(rows_edge):  # count diagonal pieces
        for j in range(cols_edge):
            for k in range(CONNECT_N-1):  # count up to a winning board
                block = board[i:i+CONNECT_N-k, j:j+CONNECT_N-k]  # connected_n-k
                if np.all(np.diag(block) == player):
                    rating += CONNECT_N ** degree
                if np.all(np.diag(block[::-1, :]) == player):
                    rating += CONNECT_N ** degree
    # return rating
    return NotImplemented


def minimax(
    board: np.ndarray, player: BoardPiece, depth: np.int, max_player: bool, heuristic
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    implement the minimax move generator (without pruning for now)
    the idea for this generate move function is to minimize the possible loss for a worst-case scenario
    in practice, to do this a tree search algorithm is created
    """
    from agents.connectn.common import apply_player_action
    from agents.connectn.common import pretty_print_board

    # other_player variable is used to alternate player pieces as depth increases
    if player == 1:
        other_player = 2
    else:
        other_player = 1

    # should define which are columns are free here to reduce computation time
    node_num = np.shape(board)[1]  # number of moves initially allowed

    # pretty_print_board(board)

    if depth == 0:  # if a terminal node, then use the heuristic
        return heuristic(board, player)

    if max_player:  # maximizing player level
        node_value = np.full(node_num, int(-1e10))  # empty array for node values, negative inf
        for node in range(node_num):  # for each child node, maximize
            board_copy = apply_player_action(board, node, player, copy=True)
            comparison = minimax(board_copy, player, depth-1, False, heuristic)
            if np.min(comparison) > node_value[node]:
                node_value[node] = np.min(comparison)
        return node_value

    else:  # minimizing player level
        node_value = np.full(node_num, int(1e10))  # empty array for node values, positive inf
        for node in range(node_num):  # for each child node, minimize
            #board_copy = apply_player_action(board, node, player, copy=True)
            board_copy = apply_player_action(board, node, other_player, copy=True)
            #comparison = minimax(board_copy, player, depth-1, True, heuristic)
            comparison = minimax(board_copy, other_player, depth-1, True, heuristic)
            if np.max(comparison) < node_value[node]:
                node_value[node] = np.max(comparison)
        return node_value



def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    generate the move for the agent from the minimax function using our heuristic
    """
    depth = 4
    heuristic = heuristic_basic  # choose which heuristic to use
    #heuristic = heuristic_better
    action_set = minimax(board, player, depth, True, heuristic=heuristic)
    print(action_set)
    if len(set(action_set)) == 1:  # if all are the same
        action = 3  # put piece in the center
    else:
        action = np.argmax(action_set)  # maximize the best move
    return action, saved_state
