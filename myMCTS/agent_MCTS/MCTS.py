import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import Optional, Tuple
from game_CONNECTN.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state, initialize_game_state
from game_CONNECTN.connectn.common import apply_player_action, NO_PLAYER, PLAYER1, PLAYER2, GameState
from math import log, sqrt


def allowed_moves(board):
    if np.ndim(board) == 2:
        pass
    elif np.ndim(board) == 3:
        board = board[-1]
    elif np.ndim(board) == 0 or np.ndim(board) == 1:
        return np.arange(0, 7, 1)
    else:
        print('broken allowed_moves input dimensions')
    allowed = list([])
    for i in range(np.shape(board)[1]):
        if board[-1, i] == NO_PLAYER:
            allowed.append(i)
    return allowed


def grab_player(board):
    num_1 = len(np.where(board == PLAYER1)[0])
    num_2 = len(np.where(board == PLAYER2)[0])
    if num_2 >= num_1:
        return PLAYER1
    if num_1 > num_2:
        return PLAYER2
    else:
        print('broken grab_player inequality')


def totuple(array):
    return tuple(map(tuple, array))


class MCTSnet(object):

    def __init__(self, board, state=None, debug=False, **kwargs):
        """
        kwargs ->
        calc_time: calculation length [s]
        move_max: maximum number of moves
        C: configuration parameter
        """
        super().__init__()
        # initialize the game board, player, and states
        self.board = board  # the current board state
        self.player = grab_player(self.board)  # the player to play
        self.states = [board]
        if state is not None:
            for s in state:
                self.states.append(s)
        # set for how long to run a calculation
        self.calc_time = kwargs.get('calc_time')
        # set a maximum number of moves and track depth
        self.move_max = kwargs.get('move_max')
        self.depth_max = 0
        # statistics to keep track of the game states
        self.wins = {}
        self.plays = {}
        # define the exploration parameter
        self.c = kwargs.get('c')
        self.debug = debug

    def best_move(self):
        # this function is what is ran in the
        # generate move function outside of the class
        # determine the best move and return it
        state = self.board
        player = grab_player(state)
        allowed = allowed_moves(state)
        # print(allowed)
        # ensure a choice is to be made
        if len(allowed) == 1:
            return allowed[0]
        elif len(allowed) == 0:
            return None

        # run the simulation
        sim_num = 0
        start = time.time()
        while time.time() - start < self.calc_time:
            self.run_sim()
            sim_num += 1
        time_elap = time.time() - start
        print('simulation number: {0}, time elapsed: {1}'.format(sim_num, time_elap))

        # choose the best move
        poss_moves = [(n, totuple(apply_player_action(np.array(state[0]), n, player, copy=True))) for n in allowed]
        win_ratio = ((self.wins.get((player, s), 0) / self.plays.get((player, s), 1), p) for p, s in poss_moves)
        to_maximize = np.transpose(np.array(list(win_ratio)))
        move = to_maximize[1, np.argmax(to_maximize[0])]

        # debugging statistics ---
        if self.debug:
            for x in sorted(
                ((100 * self.wins.get((player, s), 0) / self.plays.get((player, s), 1),
                 self.wins.get((player, s), 0), self.plays.get((player, s), 0), p+1)
                    for p, s in poss_moves),
                reverse=True
            ):
                print("{3}: {0:.2f}% ({1} / {2})".format(*x))
            print('Depth maximum: {}'.format(self.depth_max))
            # print('Possible moves: {}'.format(allowed))

        return move

    def run_sim(self):
        # add the current ratio numbers
        plays = self.plays
        wins = self.wins
        # ready to check if current state is new
        visited_states = set()
        # play a 'random' game from current position
        # and update the statistic tables with the result
        states_copy = self.states
        state = states_copy[-1]
        # grab player
        player = grab_player(state)

        # run a simulation
        expansion = True
        has_won = False
        for t in range(self.move_max):
            # define the legal moves for current position
            allowed = allowed_moves(np.array(states_copy))
            poss_moves = [(n, totuple(apply_player_action(np.array(state), n, player, copy=True))) for n in allowed]
            # check if we know the statistics for the possible moves
            # here we use the UCB1 algorithm
            if len(poss_moves) < 1:
                break
            elif all(plays.get((player, s)) for p, s in poss_moves):
                # print(np.shape(poss_moves))
                # If we have stats on all of the legal moves here, use them.
                log_sum = log(sum(plays[(player, s)] for p, s in poss_moves))
                _, move, state = max(
                    ((wins[(player, s)] / plays[(player, s)]) +
                     self.c * sqrt(log_sum / plays[(player, s)]), p, s)
                    for p, s in poss_moves
                )
            # otherwise just make a random decision
            else:
                move, state = random.choice(poss_moves)

            # update the states array
            # print(xp.shape(states_copy), np.shape(state))
            # states_copy = np.append(states_copy, np.array(state), axis=1)
            states_copy.append(np.array(state))
            # print(states_copy)
            # -------
            if (player, state) not in self.plays and expansion:
                expansion = False
                plays[(player, state)] = 0
                wins[(player, state)] = 0
                if t > self.depth_max:
                    self.depth_max = t

            # only add the new player state combos
            visited_states.add((player, state))
            # update the player
            player = grab_player(state)
            # check if the state is winning
            # print(np.shape(np.array(states_copy)))
            for s in np.array(states_copy):
                if check_end_state(s, player) == GameState.IS_WIN:
                    has_won = True
                    break
                else:
                    has_won = False
                    pass

        # back-propagation
        # update plays and wins from our set
        for player, state in visited_states:
            if (player, state) not in self.plays:
                continue
            plays[(player, state)] += 1
            if player == has_won:
                wins[(player, state)] += 1


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    generate the move for the agent from the MCTS class
    """
    calc_time = 1
    move_max = int(6*7)
    c = 2

    mcts = MCTSnet(board, state=None, calc_time=calc_time, move_max=move_max, c=c, debug=True)
    action = mcts.best_move()
    print(np.shape(mcts.states))

    # hist = SavedState(first_state=[])

    # if np.ndim(hist.record(new_state=mcts.states)) == 2:
    #    saved_state = SavedState(first_state=mcts.states)
    # else:
    #    saved_state = SavedState.record(new_state=mcts.states)

    # print(saved_state)

    return action, saved_state

