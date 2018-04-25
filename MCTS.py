import numpy as np
import board
from model import *
import copy
import random
from constants import *
import math

class Node:
    def __init__(self, state, currPlayer):
        self.state = state
        self.currPlayer = currPlayer
        self.edges = []
        self.pi = np.zeros(NUM_CHECKERS * BOARD_WIDTH * BOARD_HEIGHT, dtype='float64')

    def isLeaf(self):
        return len(self.edges) == 0


class Edge:
    def __init__(self, inNode, outNode, prior, fromPos, toPos):
        self.inNode = inNode
        self.outNode = outNode
        self.currPlayer = inNode.currPlayer
        self.fromPos = fromPos
        self.toPos = toPos

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior
        }


class MCTS:
    def __init__(self, root, model, cpuct=C_PUCT, num_itr=MCTS_SIMULATIONS):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.num_itr = num_itr
        self.model = model


    def moveToLeaf(self):
        breadcrumbs = []
        currentNode = self.root

        while not currentNode.isLeaf():
            maxQU = float("-inf");
            chosen_edges = []
            N_sum = 0

            for edge in currentNode.edges:
                N_sum += edge.stats['N']

            for edge in currentNode.edges:
                U = self.cpuct * edge.stats['P'] * math.sqrt(N_sum) / (1 + edge.stats['N'])
                QU = edge.stats['Q'] + U
                if QU > maxQU:
                    maxQU = QU
                    chosen_edges = [edge]
                elif math.fabs(QU - maxQU) < EPSILON:
                    chosen_edges.append(edge)

            # Choose a random node to continue simulation
            sampled_edge = random.choice(chosen_edges)
            breadcrumbs.append(sampled_edge)
            currentNode = sampled_edge.outNode

        return currentNode, breadcrumbs


    def expandAndBackUp(self, leafNode, breadcrumbs):

        winner = leafNode.state.check_win()
        if winner:
            for edge in breadcrumbs:
                direction = 1 if edge.currPlayer == leafNode.currPlayer else -1
                edge.stats['N'] += 1
                edge.stats['W'] += REWARD["win"] * direction
                edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
            return

        p_evaluated, v_evaluated = self.model.predict(Model.to_model_input(leafNode.state, leafNode.currPlayer))
        p_evaluated = p_evaluated.squeeze()

        valid_actions = leafNode.state.get_valid_moves(leafNode.currPlayer)

        for checker_pos, action_set in valid_actions.items():
            checker_id = leafNode.state.checkers_id[leafNode.currPlayer][checker_pos]
            for destination_pos in action_set:
                # Get index in neural net output vector
                prior_index = Model.encode_checker_index(checker_id, destination_pos)
                next_player = PLAYER_ONE + PLAYER_TWO - leafNode.currPlayer
                # Set up new state of game
                next_state = copy.deepcopy(leafNode.state)
                next_state.place(leafNode.currPlayer, checker_pos, destination_pos)
                # Build new edge and node for the new state
                newNode = Node(next_state, next_player)
                newEdge = Edge(leafNode, newNode, p_evaluated[prior_index], checker_pos, destination_pos)
                leafNode.edges.append(newEdge)

        for edge in breadcrumbs:
            direction = 1 if edge.currPlayer == leafNode.currPlayer else -1
            edge.stats['N'] += 1
            edge.stats['W'] += v_evaluated * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']


    def selfPlay(self):
        player_progresses = [0, 0]
        player_turn = 0
        num_useless_moves = 0
        actual_play_history = []

        while True:
            # Decide next move
            pi, sampled_edge = self.search()
            actual_play_history.append((self.root.state, pi))
            # Move to next board state
            self.root = sampled_edge.outNode

            progress_evaluated = self.root.state.player_progress(player_turn + 1)
            if progress_evaluated == player_progresses[player_turn]:
                num_useless_moves += 1
            else:
                num_useless_moves = 0
                player_progresses[player_turn] = progress_evaluated

            # Change player
            player_turn = 1 - player_turn

            if num_useless_moves >= PROGRESS_MOVE_LIMIT or self.root.state.check_win():
                break

        return actual_play_history, self.root.state.check_win()


    def search(self):
        for i in range(self.num_itr):
            leafNode, breadcrumbs = self.moveToLeaf()
            self.expandAndBackUp(leafNode, breadcrumbs)

        chosen_edges = []
        maxN = float("-inf")
        sumPi = 0

        for edge in self.root.edges:
            probability = edge.stats['N'] ** (1/TREE_TAU)
            checker_id = self.root.state.checkers_id[self.root.currPlayer][edge.fromPos]
            neural_net_index = Model.encode_checker_index(checker_id, edge.toPos)
            self.root.pi[neural_net_index] = probability
            sumPi += probability

            if probability > maxN:
                maxN = probability
                chosen_edges = [edge]
            elif math.fabs(probability - maxN) < EPSILON:
                chosen_edges.append(edge)

        self.root.pi /= sumPi
        sampled_edge = random.choice(chosen_edges)
        return self.root.pi, sampled_edge



if __name__ == '__main__':
    count = 0
    board = board.Board()
    node = Node(board, 1)
    model = ResidualCNN()
    tree = MCTS(node, model)
    for state, pi in tree.selfPlay()[0]:
        state.visualise()





