from game import Game
from config import *

"""
Run this file directly from terminal if you
want to play human-vs-greedy game
"""

if __name__ == '__main__':
    count = { PLAYER_ONE : 0, PLAYER_TWO : 0 }
    for i in range(1):
        print(i, end=' ')
        game = Game(p1_type='greedy', p2_type='greedy', verbose=False)
        winner = game.start()
        count[winner] += 1

    print('Player {} wins {} matches'.format(PLAYER_ONE, count[PLAYER_ONE]))
    print('Player {} wins {} matches'.format(PLAYER_TWO, count[PLAYER_TWO]))
