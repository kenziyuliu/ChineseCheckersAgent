import numpy as np

import utils
from MCTS import MCTS, Node
from config import *
from board import Board

def selfplay(model1, model2):
    if model2 is None:
        print ('model2 is None')
        model2 = model1

    player_progresses = [0, 0]
    player_turn = 0
    num_useless_moves = 0
    play_history = []
    tree_tau = TREE_TAU

    board = Board() # 1 selfplay, 1 game
    root = Node(board, PLAYER_ONE) # initial root
    use_model1 = True
    while True:
        model = model1 if use_model1 else model2
        root = make_move(root, model, tree_tau, play_history)
        assert root.isLeaf()
        cur_player_hist_moves = [move for i, move in enumerate(root.state.hist_moves) if i % 2 == 0]
        history_dests = set([move[1] for move in cur_player_hist_moves])

        # If limited destinations exist in the past moves, then there is some kind of repetition
        if (len(cur_player_hist_moves) * 2) >= TOTAL_HIST_MOVES and len(history_dests) <= UNIQUE_DEST_LIMIT:
            break

        # Evaluate player progress for stopping
        progress_evaluated = root.state.player_progress(player_turn + 1)
        if progress_evaluated > player_progresses[player_turn]:
            utils.stress_message('Reduced number of useless moves as some progress was made')
            num_useless_moves = int(PROGRESS_MOVE_LIMIT * (NUM_CHECKERS - 1) / NUM_CHECKERS)
            player_progresses[player_turn] = progress_evaluated
        else:
            num_useless_moves += 1

        # Change player
        player_turn = 1 - player_turn
        use_model1 = not use_model1

        # Change TREE_TAU to very small if game has certain progress so actions are deterministic
        if len(play_history) > TOTAL_MOVES_TILL_TAU0:
            tree_tau = 0.01

        # Stop if the game is nonsense or someone wins
        if num_useless_moves >= PROGRESS_MOVE_LIMIT:
            utils.stress_message('Game stopped by reaching progress move limit')
            break

        if root.state.check_win():
            utils.stress_message('END GAME REACHED')
            break

    return play_history, get_reward(root.state)


# code inside original while loop of selfplay()
def make_move(root, model, tree_tau, play_history):
    assert root.isLeaf()
    tree = MCTS(root, model)

    # add Dirichlet noise to prior probs at the root to ensure all moves may be tried
    tree.expandAndBackUp(tree.root, breadcrumbs=[])     # board.place is called in the expandAndBackUp()
                                                        # breadcrumbs=[] as root has empth path back to root
    assert len(tree.root.edges) > 0 # as root has been expanded
    dirichlet_noise = np.random.dirichlet(np.ones(len(tree.root.edges)) * DIRICHLET_ALPHA)
    for i in range(len(tree.root.edges)):
        tree.root.edges[i].stats['P'] *= (1. - DIR_NOISE_FACTOR)
        tree.root.edges[i].stats['P'] += DIR_NOISE_FACTOR * dirichlet_noise[i]
    # Decide next move from the root with 1 level of prior probability
    pi, sampled_edge = tree.search()
    play_history.append((tree.root.state, pi))
    outNode = sampled_edge.outNode
    outNode.edges.clear()

    return outNode # root for next iteration


def get_reward(board):
    """
    return the reward for player one
    """
    winner = board.check_win()
    if winner == PLAYER_ONE:
        return REWARD["win"]
    if winner == PLAYER_TWO:
        return REWARD["lose"]

    player_one_distance = board.player_forward_distance(PLAYER_ONE)
    player_two_distance = board.player_forward_distance(PLAYER_TWO)

    if abs(player_one_distance - player_two_distance) <= DIST_THRES_FOR_REWARD:
        return REWARD["draw"]

    return 1 if (player_one_distance - player_two_distance >= DIST_THRES_FOR_REWARD) else -1
