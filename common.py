import numpy as np
from typing import Optional, Callable, Tuple
from enum import Enum

BoardPiece = np.int8
NO_PLAYER = BoardPiece(0)  # Empty position
PLAYER1 = BoardPiece(1)  # Position where Player 1 has a piece
PLAYER2 = BoardPiece(2)  # Position where Player 2 has a piece

PlayerAction = np.int8  # The column next to be played in

CONNECT_N = 4


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    return np.zeros((6, 7), dtype=BoardPiece(0))


Board = initialize_game_state()
# board[0, 0] = 2  # Lower left corner of board
# board[5,6] = 1  # Upper right corner of board


def pretty_print_board(board: np.ndarray) -> str:
    # return print(board[::-1])

    # Convert the board of integers into string values for each player
    players = {"NO_PLAYER": " ", "PLAYER1": "x", "PLAYER2": "o"}
    string_board = np.empty((6, 7), dtype=str)
    for i in range(6):
        for j in range(7):
            if board[i, j] == PLAYER1:
                string_board[i, j] = players["PLAYER1"]
            elif board[i, j] == PLAYER2:
                string_board[i, j] = players["PLAYER2"]
            elif board[i, j] == 0:
                string_board[i, j] = players["NO_PLAYER"]

    # Print the strings out in a nice way
    print('_____________________________')
    print('|', string_board[5, 0], '|', string_board[5, 1], '|', string_board[5, 2], '|',
          string_board[5, 3], '|', string_board[5, 4], '|', string_board[5, 5], '|', string_board[5, 6], '|')
    print('|', string_board[4, 0], '|', string_board[4, 1], '|', string_board[4, 2], '|',
          string_board[4, 3], '|', string_board[4, 4], '|', string_board[4, 5], '|', string_board[4, 6], '|')
    print('|', string_board[3, 0], '|', string_board[3, 1], '|', string_board[3, 2], '|',
          string_board[3, 3], '|', string_board[3, 4], '|', string_board[3, 5], '|', string_board[3, 6], '|')
    print('|', string_board[2, 0], '|', string_board[2, 1], '|', string_board[2, 2], '|',
          string_board[2, 3], '|', string_board[2, 4], '|', string_board[2, 5], '|', string_board[2, 6], '|')
    print('|', string_board[1, 0], '|', string_board[1, 1], '|', string_board[1, 2], '|',
          string_board[1, 3], '|', string_board[1, 4], '|', string_board[1, 5], '|', string_board[1, 6], '|')
    print('|', string_board[0, 0], '|', string_board[0, 1], '|', string_board[0, 2], '|',
          string_board[0, 3], '|', string_board[0, 4], '|', string_board[0, 5], '|', string_board[0, 6], '|')
    print('=============================')
    print('|', '1', '|', '2', '|', '3', '|', '4', '|', '5', '|', '6', '|', '7', '|')
    return ''


# pretty_print_board(Board)


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    num_row = np.shape(board)[0]
    for row in range(num_row):  # looping over the number of rows
        if board[row, action] == 0:  # if empty, add a piece
            board[row, action] = player
            break
        else:  # otherwise continue until empty slot is found
            pass
    return board


def string_to_board(pp_board: str) -> np.ndarray:
    raise NotImplemented()


def connected_four_first_implementation(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    # Go through every point on the board
    for i in board[0, :3]:
        for j in board[:3, i]:
            print(i, j)
            # Check the three nearby points in every direction
            count = 0
            for k in range(4):
                if board[i+k, j+k] == player:  # Diagonally
                    count += 1
                elif board[i+k, j] == player:  # Sideways
                    count += 1
                elif board[i, j+k] == player:  # Vertically
                    count += 1
                else:
                    pass
            if count == 4:
                print('Connected Four!')
                return True
            else:
                return False


def connected_four(
    board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1

    for i in range(rows):  # check for a vertical connect_n
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player):
                return True

    for i in range(rows_edge):  # check for a horizontal connect_n
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player):
                return True

    for i in range(rows_edge):  # check for a diagonal connect_n
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True

    return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    end_state = connected_four(board, player)
    if board is -1:  # not implemented draw yet
        return GameState.IS_DRAW
    elif end_state is True:
        return GameState.IS_WIN
    elif end_state is False:
        return GameState.STILL_PLAYING


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]
