from game_CONNECTN.connectn.common import *
from agent_MCTS.MCTS import *


def test_allowed_moves():
    """should return an array the size of the possible move number"""

    board = initialize_game_state()
    ret = allowed_moves(board)

    assert np.shape(ret) == np.shape(board[0])
    assert type(ret[0]) == np.int


def test_grab_player():
    """should return an a value assigned to the player"""

    board1 = initialize_game_state()
    board1[np.random.randint(np.shape(board1)[0]),
           np.random.randint(np.shape(board1)[1])] = PLAYER1
    ret1 = grab_player(board1)
    assert ret1 == PLAYER2

    board2 = initialize_game_state()
    board2[np.random.randint(np.shape(board2)[0]),
           np.random.randint(np.shape(board2)[1])] = PLAYER2
    ret2 = grab_player(board2)
    assert ret2 == PLAYER1


def test_totuple():
    """should take an array and return a tuple"""

    some_array = np.random.random_sample((3, 3))
    ret = totuple(some_array)

    assert type(ret) == tuple


def test_mcts_best_move():
    """should return a value that is the best move"""

    board_shape = np.shape(initialize_game_state())
    board = np.random.randint(0, 3, board_shape)

    calc_time, move_max, c = 0.1, 100, np.sqrt(2)
    mcts = MCTSnet(board, state=None, calc_time=calc_time, move_max=move_max, c=c, debug=None)
    ret = mcts.best_move()

    assert isinstance(ret, np.integer)


