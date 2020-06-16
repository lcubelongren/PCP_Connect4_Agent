import numpy as np
from typing import Optional, Tuple
from agents.connectn.common import PlayerAction, BoardPiece, SavedState


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    row_num, col_num = np.shape(board)[0], np.shape(board)[1]
    action_set = []  # empty array for possible valid moves
    for i in range(col_num):  # checking each column
        if board[row_num-1, i] == 0:  # if valid, add the possible move
            action_set = np.append(i, action_set)
    action = int(np.random.choice(action_set))  # choose random possible move
    return action, saved_state
