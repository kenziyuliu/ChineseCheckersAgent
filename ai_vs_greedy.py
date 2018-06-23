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


def agent_greedy_match(model_path, num_games, verbose=False, tree_tau=DET_TREE_TAU):
    player1 = 'ai'
    player2 = 'greedy'
    win_count = { player1 : 0, player2 : 0 }
    model = load_agent(model_path)

    for i in range(num_games):
        if verbose:
            utils.stress_message('Game {}'.format(i + 1))

        if player1 == 'ai':
            game = Game(p1_type=player1, p2_type=player2, verbose=verbose, model1=model)
        else:
            game = Game(p1_type=player1, p2_type=player2, verbose=verbose, model2=model)

        winner = game.start()
        if winner is not None:
            if winner == PLAYER_ONE:
                win_count[player1] += 1
            else:
                win_count[player2] += 1
        # Swap
        player1, player2 = player2, player1

    if verbose:
        utils.stress_message('Agent wins {} games and Greedy wins {} games with total games {}'
            .format(win_count['ai'], win_count['greedy'], num_games))

    if win_count['ai'] > win_count['greedy']:
        return model_path
    elif win_count['greedy'] > win_count['ai']:
        return 'greedy'
    else:
        return None



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nUsage: python3 ai_vs_greedy.py <Model Path> [<tree tau>]\n')
        exit()

    model_path = sys.argv[1]
    tt = DET_TREE_TAU

    if len(sys.argv) == 3:
        tt = float(sys.argv[2])
        utils.stress_message('Using tree_tau {} initially'.format(tt))

    agent_greedy_match(model_path, num_games=1, verbose=True, tree_tau=tt)
