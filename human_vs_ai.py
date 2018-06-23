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

def load_agent(model_path, verbose=False):
    model = ResidualCNN()
    model.version = utils.find_version_given_filename(model_path)
    if verbose:
        print('\nLoading model from path {}'.format(model_path))
    model.load_weights(model_path)
    if verbose:
        print('Model is loaded sucessfully\n')

    return model


def human_agent_match(model_path, verbose=False, tree_tau=DET_TREE_TAU):
    model = load_agent(model_path)
    game = Game(p1_type='ai', p2_type='human', verbose=verbose, model1=model)
    winner = game.start()
    return winner


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nUsage: python3 human_vs_ai.py <Model Path> [<tree tau>]\n')
        exit()

    model_path = sys.argv[1]
    tt = DET_TREE_TAU

    if len(sys.argv) == 3:
        tt = float(sys.argv[2])
        utils.stress_message('Using tree_tau {} initially'.format(tt))


    human_agent_match(model_path, verbose=True, tree_tau=tt)
