import utils
import numpy as np
from game import Game
from config import *

"""
Run this file directly from terminal if you
want to play human-vs-greedy game
"""

if __name__ == '__main__':
    count = { PLAYER_ONE : 0, PLAYER_TWO : 0 }
    num_games = 50
    end_states = []
    for i in range(num_games):
        utils.stress_message('Game {}'.format(i + 1))
        game = Game(p1_type='greedy', p2_type='greedy', verbose=False)
        winner = game.start()
        if winner is not None:
            count[winner] += 1

        end_states.append(game.board.board[..., 0])

    unique_states = np.unique(np.array(end_states), axis=0)
    print('\n{} end game states, {} of them is unique\n'.format(num_games, len(unique_states)))

    print('Player {} wins {} matches'.format(PLAYER_ONE, count[PLAYER_ONE]))
    print('Player {} wins {} matches'.format(PLAYER_TWO, count[PLAYER_TWO]))
