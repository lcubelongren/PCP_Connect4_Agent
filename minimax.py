import numpy as np
from typing import Optional, Tuple
from agents.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state


"""
this is a raw minimax agent without alpha-beta pruning
"""


def heuristic(
        board: np.ndarray, player: BoardPiece
) -> np.int:
    """
    using similar code as what is used to check for a connect_n
    we can create a heuristic that assigns a higher value to a board
    where a player has more adjacent pieces and the highest value
    or when the player has a winning board
    """
    from agents.connectn.common import CONNECT_N

    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    rating = 0  # starting rating
    rating_base = int(1e6)  # base rating for each check

    for i in range(rows):  # count vertical pieces
        for j in range(cols_edge):
            for k in range(CONNECT_N-1):
                if np.all(board[i, j:j+CONNECT_N-k] == player):  # connected_n-k
                    rating += int(rating_base - k*rating_base/CONNECT_N)

    for i in range(rows_edge):  # count horizontal pieces
        for j in range(cols):
            for k in range(CONNECT_N-1):
                if np.all(board[i:i+CONNECT_N-k, j] == player):  # connected_n-k
                    rating += int(rating_base - k*rating_base/CONNECT_N)

    for i in range(rows_edge):  # count diagonal pieces
        for j in range(cols_edge):
            for k in range(CONNECT_N-1):
                block = board[i:i+CONNECT_N-k, j:j+CONNECT_N-k]  # connected_n-k
                if np.all(np.diag(block) == player):
                    rating += int(rating_base - k*rating_base/CONNECT_N)
                if np.all(np.diag(block[::-1, :]) == player):
                    rating += int(rating_base - k*rating_base/CONNECT_N)
    return rating


def minimax(
    board: np.ndarray, player: BoardPiece, depth: np.int, max_player: bool
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    implement the minimax move generator (without pruning for now)
    the idea for this generate move function is to minimize the possible loss for a worst-case scenario
    in practice, to do this a tree search algorithm is created
    """
    from agents.connectn.common import apply_player_action
    from agents.connectn.common import pretty_print_board

    # should define which are columns are free here to reduce computation time
    node_num = np.shape(board)[1]  # number of moves initially allowed
    depth_result = np.zeros(node_num)  # empty array for result

    # pretty_print_board(board)

    if depth == 0:  # if a terminal node, then use the heuristic
        return heuristic(board, player)

    if max_player:  # maximizing player level
        value = int(-1e6)  # negative infinity
        for node in range(node_num):  # for each child node, maximize
            board_copy = apply_player_action(board, node, player, copy=True)
            comparison = minimax(board_copy, player, depth-1, False)
            depth_result[node] = np.max([value, comparison])

    else:  # minimizing player level
        value = int(1e6)  # positive infinity
        for node in range(node_num):  # for each child node, minimize
            board_copy = apply_player_action(board, node, player, copy=True)
            comparison = minimax(board_copy, player, depth-1, True)
            depth_result[node] = np.min([value, comparison])
    return depth_result


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    generate the move for the agent from the minimax function using our heuristic
    """
    depth = 1
    action_set = minimax(board, player, depth, True)
    action = np.argmax(action_set)  # maximize the best move
    return action, saved_state
