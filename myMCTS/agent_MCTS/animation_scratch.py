from agent_MCTS.MCTS import grab_player, allowed_moves, MCTSnet
from game_CONNECTN.connectn.common import PlayerAction, BoardPiece, SavedState, check_end_state, initialize_game_state
from game_CONNECTN.connectn.common import apply_player_action, NO_PLAYER, PLAYER1, PLAYER2, GameState
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

board = initialize_game_state()

calc_time = 0.05
move_max = int(6*7)
c = np.sqrt(2)

mcts = MCTSnet(board, state=None, calc_time=calc_time, move_max=move_max, c=c, debug=False)
action = mcts.best_move()

def state_hist(states):
    board = initialize_game_state()

    p1count = 0
    p2count = 0
    for key, state in mcts.plays:
        if key == 1:
            p1count += 1
        elif key == 2:
            p2count += 1
    print(p1count, p2count)

    w1count = 0
    w2count = 0
    for key, state in mcts.wins:
        if key == 1:
            w1count += 1
        elif key == 2:
            w2count += 1
    print(w1count, w2count)


def simple_analysis(states):
    fig, ax1 = plt.subplots(1,1)
    ims = []
    sim_template = 'simulation # = {}'
    state_template = 'state # = {}'

    sim_count = -1
    state_count = -1
    for i in range(np.shape(states)[0]):
        if np.sum(mcts.states[i] == NO_PLAYER) >= np.size(mcts.states[0]) - 1:
            sim_count += 1
        ax1 = plt.imshow(mcts.states[i], animated=True, origin='lower')
        txt1 = plt.text(1, 5.75, sim_template.format(sim_count))
        txt2 = plt.text(3.5, 5.75, state_template.format(state_count))
        ims.append([ax1, txt1, txt2])

    ani = animation.ArtistAnimation(fig, ims, interval=150, repeat_delay=250, blit=False, repeat=True)
    # ani.save("simple.mp4")
    plt.show()

simple_analysis(mcts.states)

# state_hist(mcts.states)


