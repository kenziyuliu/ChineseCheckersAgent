import sys

import utils
from game import Game
from config import *
from model import *
"""
Run this file with argument specifying the model from terminal if you
want to play ai-vs-greedy game
e.g. python3 ai-vs-greedy.py saved-models/version0033.h5
Player one is ai, player two is greedy
"""

if __name__ == '__main__':
    count = { PLAYER_ONE : 0, PLAYER_TWO : 0 }
    if len(sys.argv) == 2:
        model = ResidualCNN()
        filename = sys.argv[1]
        print("\nLoading model from path {}".format(filename))
        model.load_weights(filename)
        print("Model is loaded sucessfully\n")
        for i in range(1):
            utils.stress_message('Game {}'.format(i + 1))
            game = Game(p1_type='ai', p2_type='greedy', verbose=True, model1=model)
            winner = game.start()
            if winner is not None:
                count[winner] += 1  

        print('AiPlayer {} wins {} matches'.format(PLAYER_ONE, count[PLAYER_ONE]))
        print('GreedyPlayer {} wins {} matches'.format(PLAYER_TWO, count[PLAYER_TWO]))
