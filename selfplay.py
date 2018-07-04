import copy
import numpy as np
import random

import utils
from config import *
from board import Board
from MCTS import MCTS, Node


def selfplay(model1, model2=None, randomised=False):
    '''
    Generate an agent self-play given two models
    TODO: if `randomised`, randomise starting board state
    '''
    if model2 is None:
        model2 = model1

    player_progresses = [0, 0]
    player_turn = 0
    num_useless_moves = 0
    play_history = []
    tree_tau = TREE_TAU

    board = Board(randomised=randomised)
    root = Node(board, PLAYER_ONE)          # initial game state
    use_model1 = True

    while True:
        model = model1 if use_model1 else model2

        if len(root.state.hist_moves) < INITIAL_RANDOM_MOVES:
            root = make_random_move(root)
        else:
            # Use Current model to make a move
            root = make_move(root, model, tree_tau, play_history)

        assert root.isLeaf()

        hist_moves = root.state.hist_moves
        cur_player_hist_moves = [hist_moves[i] for i in range(len(hist_moves) - 1, -1, -2)]
        history_dests = set([move[1] for move in cur_player_hist_moves])

        # If limited destinations exist in the past moves, then there is some kind of repetition
        if len(cur_player_hist_moves) * 2 >= TOTAL_HIST_MOVES and len(history_dests) <= UNIQUE_DEST_LIMIT:
            print('Repetition detected: stopping and discarding game')
            return None, None

        # Evaluate player progress for stopping
        progress_evaluated = root.state.player_progress(player_turn + 1)
        if progress_evaluated > player_progresses[player_turn]:
            num_useless_moves = int(num_useless_moves * (NUM_CHECKERS - 1) / NUM_CHECKERS)
            player_progresses[player_turn] = progress_evaluated
        else:
            num_useless_moves += 1

        # Change player
        player_turn = 1 - player_turn
        use_model1 = not use_model1

        # Change TREE_TAU to very small if game has certain progress so actions are deterministic
        if len(play_history) + INITIAL_RANDOM_MOVES > TOTAL_MOVES_TILL_TAU0:
            if tree_tau == TREE_TAU:
                print('selfplay: Changing tree_tau to {} as total number of moves is now {}'.format(DET_TREE_TAU, len(play_history)))
            tree_tau = DET_TREE_TAU

        if root.state.check_win():
            print('END GAME REACHED')
            break

        # Stop (and discard) the game if it's nonsense
        if num_useless_moves >= PROGRESS_MOVE_LIMIT:
            print('Game stopped by reaching progress move limit; Game Discarded')
            return None, None

    if randomised:
        # Discard the first `BOARD_HIST_MOVES` as the history is not enough
        return play_history[BOARD_HIST_MOVES:], utils.get_p1_winloss_reward(root.state)
    else:
        return play_history, utils.get_p1_winloss_reward(root.state)


def make_random_move(root):
    '''
    Independent on MCTS.
    Instead sample a random move from current board's valid moves.
    '''
    random.seed()

    cur_state = root.state
    player = root.currPlayer

    valid_actions = cur_state.get_valid_moves(player) # dict, key: checker pos, value: possible dest from pos

    random_start = random.choice(list(valid_actions.keys()))
    while len(valid_actions[random_start]) == 0:
        random_start = random.choice(list(valid_actions.keys()))
    random_end = random.choice(valid_actions[random_start])

    next_state = copy.deepcopy(cur_state)
    next_state.place(player, random_start, random_end)
    new_player = PLAYER_ONE + PLAYER_TWO - player

    return Node(next_state, new_player)


def make_move(root, model, tree_tau, play_history):
    '''
    Given a current board state, perform tree search
    and make a move
    (Code inside original while loop of selfplay())
    '''
    assert root.isLeaf()
    tree = MCTS(root, model, tree_tau=tree_tau)

    # Make the first expansion to possible next states
    tree.expandAndBackUp(tree.root, breadcrumbs=[])     # breadcrumbs=[] as root has empth path back to root
    assert len(tree.root.edges) > 0 # as root has been expanded

    # Add Dirichlet noise to prior probs at the root to ensure all moves may be tried
    dirichlet_noise = np.random.dirichlet(np.ones(len(tree.root.edges)) * DIRICHLET_ALPHA)
    for i in range(len(tree.root.edges)):
        tree.root.edges[i].stats['P'] *= (1. - DIR_NOISE_FACTOR)
        tree.root.edges[i].stats['P'] += DIR_NOISE_FACTOR * dirichlet_noise[i]

    # Decide next move from the root with 1 level of prior probability
    pi, sampled_edge = tree.search()
    play_history.append((tree.root.state, pi))

    outNode = sampled_edge.outNode
    outNode.edges.clear()

    return copy.deepcopy(outNode) # root for next iteration


# def get_reward(board):
#     """
#     return the reward for player one
#     """
#     winner = board.check_win()
#     if winner == PLAYER_ONE:
#         return REWARD["win"]
#     if winner == PLAYER_TWO:
#         return REWARD["lose"]
#
#     player_one_distance = board.player_forward_distance(PLAYER_ONE)
#     player_two_distance = board.player_forward_distance(PLAYER_TWO)
#
#     if abs(player_one_distance - player_two_distance) <= DIST_THRES_FOR_REWARD:
#         return REWARD["draw"]
#
#     return 1 if (player_one_distance - player_two_distance >= DIST_THRES_FOR_REWARD) else -1


if __name__ == '__main__':
    '''
    Some tests here
    '''
    import sys
    import time
    from model import ResidualCNN

    if len(sys.argv) != 2:
        print('Model needed for testing: python3 selfplay.py <model path>')
        exit()

    model_path = sys.argv[1]
    model = ResidualCNN()
    model.load_weights(model_path)

    history, reward = selfplay(model)
    for i in range(8):
        board, pi = history[i]
        board.visualise()
        time.sleep(3)


