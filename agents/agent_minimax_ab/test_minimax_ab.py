import numpy as np
from agents.connectn.common import initialize_game_state
from agents.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state
from agents.agent_minimax.minimax import heuristic_basic, heuristic_better


def test_minimax_ab():
    """should return an array the size of the possible move number"""
    from agents.agent_minimax_ab.minimax_ab import minimax_ab

    board = initialize_game_state()
    heuristic = heuristic_basic
    alpha, beta = int(-1e10), int(1e10)  # starting alpha-beta values

    ret = minimax_ab(board=board, alpha=alpha, beta=beta, player=BoardPiece(1),
                     depth=4, max_player=True, heuristic=heuristic)

    assert ret.shape == np.shape(board[0])


def test_generate_move_minimax_ab():
    """should return an int of the type player action"""
    from agents.agent_minimax_ab.minimax_ab import generate_move_minimax_ab

    board = initialize_game_state()
    ret = generate_move_minimax_ab(board=board, player=BoardPiece(1), saved_state=None)[0]

    assert type(ret) == PlayerAction
