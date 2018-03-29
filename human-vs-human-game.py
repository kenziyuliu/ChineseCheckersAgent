from game import Game

"""
Run this file directly from terminal if you
want to play human-vs-human game
"""
# human vs greedy AI
if __name__ == '__main__':
    human_game = Game(p1_type='human', p2_type='human')
    human_game.start()
