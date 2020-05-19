import numpy as np
from agents.connectn.common import BoardPiece


def test_initialize_game_state():
    from agents.connectn.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    np.all(ret == 0)


def test_pretty_print_board():
    from agents.connectn.common import pretty_print_board
    from agents.connectn.common import initialize_game_state

    ret = pretty_print_board(board=initialize_game_state())

    assert isinstance(ret, str)


def test_apply_player_action():
    from agents.connectn.common import apply_player_action
    from agents.connectn.common import initialize_game_state

    BoardPiece = np.int8
    PlayerAction = np.int8

    ret = apply_player_action(board=initialize_game_state(), action=PlayerAction, player=BoardPiece)

    assert isinstance(ret, str)


def test_string_to_board():
    from agents.connectn.common import string_to_board
    from agents.connectn.common import pretty_print_board

    ret = string_to_board(pp_board=pretty_print_board())

    assert isinstance(ret, np.ndarray)


def test_connected_four():
    from agents.connectn.common import connected_four
    from agents.connectn.common import pretty_print_board
    from agents.connectn.common import string_to_board
    from agents.connectn.common import initialize_game_state

    ret = connected_four(board=string_to_board(pp_board=pretty_print_board(initialize_game_state())), player=BoardPiece)

    assert isinstance(ret, bool)


def test_check_end_state():
    from agents.connectn.common import check_end_state
    from agents.connectn.common import pretty_print_board
    from agents.connectn.common import string_to_board
    from agents.connectn.common import initialize_game_state

    ret = check_end_state(board=string_to_board(pp_board=pretty_print_board(initialize_game_state())), player=BoardPiece)

    assert isinstance(ret, np.int8)
