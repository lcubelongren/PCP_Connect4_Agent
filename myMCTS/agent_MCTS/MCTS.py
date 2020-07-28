import numpy as np
import random
import time
from math import log, sqrt
from typing import Optional, Tuple
from game_CONNECTN.connectn.common import PlayerAction, BoardPiece, SavedState, NO_PLAYER, PLAYER1, PLAYER2, \
    apply_player_action, check_end_state, GameState


# functions to be used within the MCTS Class:

def allowed_moves(board: np.ndarray):
    """
    input ->
    board: the array for the given board state
    output ->
    allowed: the allowed moves from the input board state
    """
    allowed = list([])
    for i in range(np.shape(board)[1]):
        if board[-1, i] == NO_PLAYER:
            allowed.append(i)
    return allowed


def grab_player(board: np.ndarray or Tuple):
    """
    input ->
    board: the array for the given board state
    output ->
    allowed: the player who's turn it is to make a move
    """
    num_1 = len(np.where(board == PLAYER1)[0])
    num_2 = len(np.where(board == PLAYER2)[0])
    if num_2 >= num_1:
        return PLAYER1
    if num_1 > num_2:
        return PLAYER2
    else:
        print('broken grab_player inequality')


def totuple(array: np.ndarray):
    """
    input ->
    array: any numpy array
    output ->
    type(array) == Tuple
    """
    return tuple(map(tuple, array))


# the MCTS network Class:

class MCTSnet(object):

    """the entire Monte Carlo Tree Search network"""

    def __init__(self, board, state=None, debug=None, **kwargs):
        """
        kwargs ->
        calc_time: calculation length [s]
        move_max: maximum number of moves
        C: exploration/exploitation parameter
        """
        super().__init__()
        # initialize the game board, player, and states
        self.board = totuple(board)  # the current board state
        self.player = grab_player(self.board)  # the player to play
        self.states = []
        self.new_states = set()
        # # the following add SaveState ability
        # if state is not None:
        #     for s in state:
        #         self.states.append(s)
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
        # debugging and recording random stuff
        self.debug = debug
        self.random_count = 0
        self.not_random_count = 0
        self.new_state_count = 0

    def best_move(self):
        """
        this function is what is ran in the
        generate move function outside of the class
        determine the best move and return it
        """
        self.states.append(self.board)  # start with the current board
        state = self.states[-1]
        player = grab_player(state)
        allowed = allowed_moves(np.array(state))
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

        # determine possible moves and the win ratio
        poss_moves = [(n, totuple(apply_player_action(np.array(self.board), n, player, copy=True))) for n in allowed]
        # win_ratio = ((self.wins.get((player, s), 0) / self.plays.get((player, s), 1), p) for p, s in poss_moves)
        # to_maximize = np.transpose(np.array(list(play_num)))
        # move = np.argmax(to_maximize[0])
        # choose the best move by maximizing the number of visits
        play_num = list(self.plays.get((player, s), 0) for p, s in poss_moves)
        move = np.argmax(play_num)
        # could implement something here that randomizes equal play number results

        # debugging statistics ---
        if self.debug == 'short' or self.debug == 'long':
            print('simulation number: {0}, time elapsed: {1}'.format(sim_num, time_elap))
            for data in sorted(
                (((self.wins.get((player, s), 0) * 100) / self.plays.get((player, s), 1),  # 0. win percentage
                 self.wins.get((player, s), 0),  # 1. win number
                  self.plays.get((player, s), 0),  # 2. play number
                  p + 1)  # 3. play
                    for p, s in poss_moves), reverse=True):
                if move == data[3]-1:  # point to the chosen value
                    print("{3}: ({1} / {2}) = {0:.2f}% <--".format(*data))
                else:
                    print("{3}: ({1} / {2}) = {0:.2f}%".format(*data))
            if self.debug == 'short':
                pass
            elif self.debug == 'long':
                print('Depth maximum     : {}'.format(self.depth_max))
                print('Possible moves  : {}'.format(allowed))
                print('Random choice #   : {}'.format(self.random_count))
                print('Educated choice # : {}'.format(self.not_random_count))
                print('Visited state #   : {}'.format(self.new_state_count))
                print('State array shape : {}'.format(np.shape(self.states)))
        elif not self.debug:
            pass
        else:
            print('Error: unknown debug arg. string')
            print('-> possible choices include: \'short\', \'long\'')

        return move

    def run_sim(self):
        """
        this function runs until the maximum moves have
        been reached or the board state is a win or draw.
        in the process, depending upon if a player/state combo is new,
        the agent will go between exploration and exploitation.
        UCB1 is used to determine the ratio between them.
        """
        # play a 'random' game from current position
        # that increasingly gains more information over simulations
        # and update the statistic tables with each result
        states_copy = self.states  # create copy to be used for this simulation
        state = self.states[-1]  # the node state that expansion has stopped at

        # clear the set to record the states
        # that have been visited this simulation
        self.new_state_count += len(self.new_states)
        self.new_states.clear()

        # grab the player
        player = grab_player(state)

        # run a simulation
        count = 0
        expansion = True
        has_won = False
        has_drawn = False
        # --- SIMULATION PHASE is contained here
        while count < self.move_max and not (has_won or has_drawn):
            count += 1
            # --- SELECTION PHASE
            # define the legal moves for current position
            allowed = allowed_moves(np.array(state))

            # create all the possible next moves -> (play, board)
            poss_moves = [(n, totuple(apply_player_action(np.array(state), n, player, copy=True))) for n in allowed]
            # print(poss_moves)
            # print(check_end_state(np.array(state), player))
            # check if we know the statistics for the possible moves
            # here we use the UCB1 algorithm
            # if check_end_state(np.array(state), player) == GameState.IS_DRAW:
            #     has_drawn = True
            # if len(poss_moves) == 0:
            #     has_drawn = True

            # if we have statistics for each play, make an educated decision
            if all(self.plays.get((player, s)) for p, s in poss_moves):
                self.not_random_count += 1
                for p, s in poss_moves:
                    win_ratio = self.wins[(player, s)] / self.plays[(player, s)]
                    log_sum = log(sum(self.plays[(player, s)]))
                    exploration_term = self.c * sqrt(log_sum / self.plays[(player, s)])
                    out, move, state = max(win_ratio + exploration_term, p, np.array(s))

            # otherwise just make a random decision
            else:
                self.random_count += 1
                move, state = random.choice(poss_moves)

            # update the copied states array
            states_copy.append(state)

            # if a new node and expanding, stop expanding
            # and add a starting dictionary entry for the move
            if (player, state) not in self.plays and expansion:
                # --- EXPANSION PHASE (or end of)
                # initialize the new node dictionary values
                expansion = False
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0
                if count > self.depth_max:
                    self.depth_max = count

            # add only the new player state combos
            self.new_states.add((player, state))

            # update the player
            player = grab_player(np.array(state))

            # check if the state is winning or drawn
            # if so, it breaks out of the while loop
            if check_end_state(np.array(state), player) == GameState.IS_WIN:
                has_won = True
            elif check_end_state(np.array(state), player) == GameState.IS_DRAW:
                has_drawn = True

        # update plays and wins from our set
        # --- BACKPROPAGATION PHASE
        for player, state in self.new_states:
            # print((player, state))
            if (player, state) not in self.plays:
                # skip the given play if state has not been tried
                continue
            self.plays[(player, state)] += 1
            if has_won:
                self.wins[(player, state)] += 1


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    """generate the move for the agent using the MCTS network"""

    calc_time = 0.3
    move_max = 100
    c = np.sqrt(2)

    mcts = MCTSnet(board, state=None, calc_time=calc_time, move_max=move_max, c=c, debug=None)
    action = mcts.best_move()

    return action, saved_state

