from game import Game
from constants import *
import sys
from model import *
"""
Run this file with argument specifying the model from terminal if you
want to play ai-vs-greedy game
Player one is ai, player two is greedy
"""

if __name__ == '__main__':
    count = { PLAYER_ONE : 0, PLAYER_TWO : 0 }
    if len(sys.argv) == 2:
        model = ResidualCNN()
        filename = sys.argv[1]
        model.load(filename)
        for i in range(1):
            print(i, end=' ')
            game = Game(p1_type='ai', p2_type='greedy', verbose=True, model1 = model)
            winner = game.start()
            count[winner] += 1

        print('AiPlayer {} wins {} matches'.format(PLAYER_ONE, count[PLAYER_ONE]))
        print('GreedyPlayer {} wins {} matches'.format(PLAYER_TWO, count[PLAYER_TWO]))
