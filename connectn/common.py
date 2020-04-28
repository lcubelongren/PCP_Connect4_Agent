import numpy as np
from typing import Optional
from enum import Enum

BoardPiece = np.int8
NO_PLAYER = BoardPiece(0)  # Empty position
PLAYER1 = BoardPiece(1)  # Position where Player 1 has a piece
PLAYER2 = BoardPiece(2)  # Position where Player 2 has a piece

PlayerAction = np.int8  # The column next to be played in


class GameState(Enum):
    WIN = 1
    DRAW = -1
    PLAYING = 0


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
    return 'Pretty Print Board'


# pretty_print_board(Board)


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    for i in range(6):
        if board[i, action] == 0:
            board[i, action] = player
            break
        else:
            pass
    return board


Board = apply_player_action(board=Board, action=np.int8(0), player=PLAYER1)
Board = apply_player_action(board=Board, action=np.int8(2), player=PLAYER1)
Board = apply_player_action(board=Board, action=np.int8(1), player=PLAYER1)
Board = apply_player_action(board=Board, action=np.int8(5), player=PLAYER2)
Board = apply_player_action(board=Board, action=np.int8(2), player=PLAYER1)
Board = apply_player_action(board=Board, action=np.int8(1), player=PLAYER2)
Board = apply_player_action(board=Board, action=np.int8(3), player=PLAYER1)
pretty_print_board(Board)


def string_to_board(pp_board: str) -> np.ndarray:
    raise NotImplemented()


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    # Go through every point on the board
    for i in board[0, :3]:
        for j in board[:3, i]:
            print(i, j)
            # Check the three nearby points in every direction
            count = 0
            for k in range(4):
                if board[i+k, j+k] == PLAYER1:  # Diagonally
                    count += 1
                elif board[i+k, j] == PLAYER1:  # Sideways
                    count += 1
                elif board[i, j+k] == PLAYER1:  # Vertically
                    count += 1
                else:
                    pass
            if count == 4:
                return True
            else:
                return False


print(connected_four(board=Board, player=PLAYER1))


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    raise NotImplemented()
