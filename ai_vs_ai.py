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

def load_agent(model_path, model_num, verbose=False):
    model = ResidualCNN()
    model.version = utils.find_version_given_filename(model_path)
    if verbose:
        print('\nLoading model {} from path {}'.format(model_num, model_path))

    model.load_weights(model_path)

    if verbose:
        print('Model {} is loaded sucessfully\n'.format(model_num))

    return model


def agent_match(model1_path, model2_path, num_games, verbose=False, tree_tau=DET_TREE_TAU, enforce_move_limit=False):
    win_count = { PLAYER_ONE: 0, PLAYER_TWO: 0 }

    model1 = load_agent(model1_path, 1, verbose)
    model2 = load_agent(model2_path, 2, verbose)

    for i in range(num_games):
        if verbose:
            utils.stress_message('Game {}'.format(i + 1))
        game = Game(p1_type='ai', p2_type='ai', verbose=verbose, model1=model1, model2=model2, tree_tau=tree_tau)
        winner = game.start(enforce_move_limit=enforce_move_limit)
        if winner is not None:
            win_count[winner] += 1

    if verbose:
        print('Agent "{}" wins {} matches'.format(model1_path, win_count[PLAYER_ONE]))
        print('Agent "{}" wins {} matches'.format(model2_path, win_count[PLAYER_TWO]))

    # Return the winner by at least 55% win rate
    if win_count[PLAYER_ONE] > int(0.55 * num_games):
        return model1_path
    elif win_count[PLAYER_TWO] > int(0.55 * num_games):
        return model2_path
    else:
        return None


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 ai-vis-ai.py <model1_path> <model2_path> [<tree tau>]')
        exit()

    if len(sys.argv) == 3:
        agent_match(sys.argv[1], sys.argv[2], 1, True)
    else:
        tree_tau = int(sys.argv[3])
        utils.stress_message('Using tree_tau {} initially'.format(tree_tau))
        agent_match(sys.argv[1], sys.argv[2], 1, True, tree_tau)
