from game import Game

"""
Run this file directly from terminal if you
want to play human-vs-greedy game
"""

if __name__ == '__main__':
    game = Game(p1_type='human', p2_type='greedy')
    game.start()
