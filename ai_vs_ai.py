import re
import sys

from game import Game
from config import *
from model import *
from utils import find_version_given_filename
"""
Run this file with argument specifying the models from terminal if you
want to play ai-vs-ai game
e.g. python3 ai-vs-ai.py saved-models/version0000.h5 saved-models/version0033.h5
"""

def load_agent(model_path, model_num):
    model = ResidualCNN()
    version = utils.find_version_given_filename(model_path)
    model.version = version
    print('\nLoading model {} from path {}'.format(model_num, model_path))
    model.load_weights(model_path)
    print('Model {} is loaded sucessfully\n'.format(model_num))

    return model


def agent_match(model1_path, model2_path, num_games, verbose=False):
    players = { PLAYER_ONE: model1_path, PLAYER_TWO: model2_path }
    win_count = { PLAYER_ONE: 0, PLAYER_TWO: 0 }

    model1 = load_agent(model1_path, 1)
    model2 = load_agent(model2_path, 2)

    for i in range(1):
        utils.stress_message('Game {}'.format(i + 1))
        game = Game(p1_type='ai', p2_type='ai', verbose=verbose, model1=model1, model2=model2)
        winner = game.start()
        if winner is not None:
            win_count[winner] += 1

    print('Agent "{}" wins {} matches'.format(model1_path, win_count[PLAYER_ONE]))
    print('Agent "{}" wins {} matches'.format(model2_path, win_count[PLAYER_TWO]))

    return players[max(win_count)]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 ai-vis-ai.py <model1_path> <model2_path>')
        exit()

    agent_match(sys.argv[1], sys.argv[2], 1, True)

