import numpy as np
from typing import Optional, Tuple
from agents.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state
from agents.agent_minimax.minimax import heuristic_basic, heuristic_better


"""
this is a copy of the basic minimax agent, with alpha-beta pruning added
alpha: best already explored option for the maximizer
beta: best already explored option for the minimizer 
"""


def minimax_ab(
    board: np.ndarray, alpha: np.int, beta: np.int, player: BoardPiece, depth: np.int, max_player: bool, heuristic
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    implement the minimax move generator with alpha-beta pruning
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
            comparison = np.min(minimax_ab(board_copy, alpha, beta, player, depth-1, False, heuristic))
            if comparison > node_value[node]:
                node_value[node] = comparison
            # alpha-beta pruning:
            if comparison >= beta:
                return node_value
            if comparison > alpha:
                alpha = comparison
        return node_value

    else:  # minimizing player level
        node_value = np.full(node_num, int(1e10))  # empty array for node values, positive inf
        for node in range(node_num):  # for each child node, minimize
            board_copy = apply_player_action(board, node, other_player, copy=True)
            comparison = np.max(minimax_ab(board_copy, alpha, beta, other_player, depth-1, True, heuristic))
            if comparison < node_value[node]:
                node_value[node] = comparison
            # alpha-beta pruning
            if comparison <= alpha:
                return node_value
            if comparison < beta:
                beta = comparison
        return node_value


def generate_move_minimax_ab(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    generate the move for the agent from the minimax with pruning function using our heuristic
    """
    depth = 4
    # choose which heuristic to use
    heuristic = heuristic_basic
    # heuristic = heuristic_better
    alpha, beta = int(-1e10), int(1e10)  # starting alpha-beta values
    action_set = minimax_ab(board, alpha, beta, player, depth, True, heuristic=heuristic)
    print(action_set)
    # if np.all(board) == 0:  # if the board is empty
    if len(set(action_set)) == 1:  # if all options are the same
        action = 3  # put piece in the center
    else:
        action = np.argmax(action_set)  # maximize the best move
    return action, saved_state
