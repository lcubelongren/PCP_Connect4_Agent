import numpy as np


def test_initialize_game_state():
    """should return an nd.array of the shape (6,7)"""
    from agents.connectn.common import initialize_game_state

    ret = initialize_game_state()  # add function to test

    assert isinstance(ret, np.ndarray)  # check type
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)  # check shape
    assert np.all(ret == 0)  # check board is empty


def test_pretty_print_board():
    """should print the board in string form"""
    from agents.connectn.common import pretty_print_board, initialize_game_state

    ret = pretty_print_board(board=initialize_game_state())  # add function to test

    assert isinstance(ret, str)  # check type


def test_apply_player_action():
    """should return a copy of the board with another piece added"""
    from agents.connectn.common import apply_player_action, initialize_game_state
    from agents.connectn.common import BoardPiece, PlayerAction

    for i in range(1, 3):  # for each possible piece assignment
        for j in range(7):  # for each player action
            board = initialize_game_state()  # initial board
            ret = apply_player_action(board=board, action=PlayerAction(j),
                                      player=BoardPiece(i), copy=True)

            assert np.sum(np.sum(ret)) > np.sum(np.sum(board))  # check a piece was played


def test_string_to_board():
    return NotImplemented


def test_connected_four():
    """
    should return a bool signifying if the game has been won (binary)
    this test will be a bit more difficult than the previous,
    let's try to make a board for each angle of connections
    """
    from agents.connectn.common import apply_player_action, initialize_game_state, connected_four, pretty_print_board
    from agents.connectn.common import BoardPiece, PlayerAction, CONNECT_N

    for i in range(1, 3):  # for each possible piece assignment
        board1 = initialize_game_state()  # initial board
        # putting down connect_n pieces
        for j in range(CONNECT_N):  # check for a horizontal connection
            ret = apply_player_action(board=board1, action=PlayerAction(j),
                                      player=BoardPiece(i), copy=False)
        # pretty_print_board(board1)
        assert connected_four(board=ret, player=BoardPiece(i))  # check returns 'True'

        board2 = initialize_game_state()  # initial board
        for _ in range(CONNECT_N):  # check for a vertical connection
            ret = apply_player_action(board=board2, action=PlayerAction(0),
                                      player=BoardPiece(i), copy=False)
        # pretty_print_board(board2)
        assert connected_four(board=ret, player=BoardPiece(i))  # check returns 'True'

        board3 = initialize_game_state()  # initial board
        count = 0
        for j in range(10):  # check for a diagonal connection (only works for connect_n=4)
            if j in [0, 2, 5, 9]:  # column peak
                ret = apply_player_action(board=board3, action=PlayerAction(count), player=BoardPiece(i), copy=False)
                count += 1  # increase the action column
            else:
                ret = apply_player_action(board=board3, action=PlayerAction(count), player=np.int8(3), copy=False)
        # pretty_print_board(board3)
        assert connected_four(board=ret, player=BoardPiece(i))  # check returns 'True'


def test_check_end_state():
    """
    should return class variable for the state of the game
    hardest part here is to check the draw condition
    """
    from agents.connectn.common import apply_player_action, initialize_game_state, connected_four
    from agents.connectn.common import check_end_state, GameState, pretty_print_board
    from agents.connectn.common import BoardPiece, PlayerAction, CONNECT_N

    # check the draw condition
    board1 = initialize_game_state()  # initial board
    width = np.shape(board1)[0]
    height = np.shape(board1)[1]
    player_choice = np.ones(width*height)
    # the following creates a draw state board
    player_choice[:width*height//2:2] += 1
    player_choice[width*height//2::2] += 1
    count = 0
    for h in range(np.shape(board1)[0]):
        for w in range(np.shape(board1)[1]):
            apply_player_action(board=board1, action=PlayerAction(w),
                                player=BoardPiece(player_choice[count]), copy=False)
            count += 1
    # pretty_print_board(board1)
    check_state1 = connected_four(board=board1, player=BoardPiece(1))
    check_state2 = connected_four(board=board1, player=BoardPiece(2))
    assert check_state1 == False  # check not a winning state, player1
    assert check_state2 == False  # check not a winning state, player2
    assert check_end_state(board=board1, player=BoardPiece(1)) == GameState.IS_DRAW  # check the function, player1
    assert check_end_state(board=board1, player=BoardPiece(2)) == GameState.IS_DRAW  # check the function, player2

    # check the winning condition
    board2 = initialize_game_state()  # initial board
    # create a winning board
    for j in range(CONNECT_N):  # check for a horizontal connection
        ret = apply_player_action(board=board2, action=PlayerAction(j), player=BoardPiece(1), copy=False)
    # pretty_print_board(board2)
    assert check_end_state(board=board2, player=BoardPiece(1)) == GameState.IS_WIN

    # check the still playing condition
    board3 = initialize_game_state()  # initial board, thus still playing
    # pretty_print_board(board3)
    assert check_end_state(board=board3, player=BoardPiece(1)) == GameState.STILL_PLAYING
