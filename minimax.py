import numpy as np
from typing import Optional, Tuple
from agents.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state


def heuristic(
        board: np.ndarray, player: BoardPiece
) -> np.int:
    # using similar code as what is used to check for a connect_n
    # we can create a heuristic that assigns a higher value to a board
    # where a player has more adjacent pieces and the highest value
    # for when the player has a winning board
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
    board: np.ndarray, player: BoardPiece, depth: np.int, max_player: bool, node: np.int
) -> np.int:
    # implement the minimax move generator (without pruning for now)
    # the idea for this generate move function is to minimize the possible loss for a worst-case scenario
    # in practice, to do this a tree search algorithm is created
    from agents.connectn.common import initialize_game_state
    from agents.connectn.common import apply_player_action
    from agents.connectn.common import pretty_print_board

    num_parents = np.shape(board)[1]  # number of moves initially
    result = np.zeros((depth, num_parents))  # empty array for result

    # get the correct board
    apply_player_action(board, node, player)
    pretty_print_board(board)

    if depth == 0:  # if a terminal node, then use the heuristic
        return heuristic(board, player)
    if max_player:  # maximizing player level
        value = int(-1e6)  # negative infinity
        for i in range(depth):
            for node in range(num_parents):  # for each child node, maximize
                result[i, node] = np.max([value, minimax(board, player, depth-1, False, node)])
    else:  # minimizing player level
        value = int(1e6)  # positive infinity
        for i in range(depth):
            for node in range(num_parents):  # for each child node, minimize
                result[i, node] = np.min([value, minimax(board, player, depth-1, True, node)])
    print(result)
    return result


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    from agents.connectn.common import initialize_game_state

    depth = 0
    num_parents = np.shape(board)[1]  # possible move choices
    action_set = np.zeros(num_parents)  # empty move rating array
    for node in range(num_parents):  # root node
        action_set[node] = np.argmax(minimax(board, player, depth, True, node))
    # print(action_set)
    action = np.argmax(action_set)  # maximize the best move
    return action, saved_state
