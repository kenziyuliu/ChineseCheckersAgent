import gc
import math
import copy
import random
import numpy as np

import utils
import board
from model import *
from config import *


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
    def __init__(self, root, model, cpuct=C_PUCT, num_itr=MCTS_SIMULATIONS, tree_tau=TREE_TAU):
        self.root = root
        self.tree = {}
        self.cpuct = cpuct
        self.num_itr = num_itr
        self.model = model
        self.tree_tau = tree_tau


    def moveToLeaf(self):
        breadcrumbs = []
        currentNode = self.root

        while not currentNode.isLeaf():
            maxQU = float('-inf')
            chosen_edges = []
            N_sum = 0

            for edge in currentNode.edges:
                N_sum += edge.stats['N']

            for edge in currentNode.edges:
                U = self.cpuct * edge.stats['P'] * np.sqrt(N_sum) / (1. + edge.stats['N'])
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
        assert leafNode.isLeaf()
        winner = leafNode.state.check_win()
        if winner:
            for edge in breadcrumbs:
                # If a win state occurred, then then leafNode must be the turn of the lost player
                # Therefore when backing up, the leafNode player gets negative reward
                direction = -1 if edge.currPlayer == leafNode.currPlayer else 1
                edge.stats['N'] += 1
                edge.stats['W'] += REWARD['win'] * direction
                edge.stats['Q'] = edge.stats['W'] / float(edge.stats['N'])  # Use float() for python2 compatibility
            return

        # Use model to make prediction at a leaf node
        p_evaluated, v_evaluated = self.model.predict(utils.to_model_input(leafNode.state, leafNode.currPlayer))

        valid_actions = leafNode.state.get_valid_moves(leafNode.currPlayer)

        for checker_pos, action_set in valid_actions.items():
            checker_id = leafNode.state.checkers_id[leafNode.currPlayer][checker_pos]
            for destination_pos in action_set:
                # Get index in neural net output vector
                prior_index = utils.encode_checker_index(checker_id, destination_pos)
                next_player = PLAYER_ONE + PLAYER_TWO - leafNode.currPlayer
                # Set up new state of game
                next_state = copy.deepcopy(leafNode.state)
                next_state.place(leafNode.currPlayer, checker_pos, destination_pos)
                # Build new edge and node for the new state
                newNode = Node(next_state, next_player)
                newEdge = Edge(leafNode, newNode, p_evaluated[prior_index], checker_pos, destination_pos)
                leafNode.edges.append(newEdge)

        # Back up the value
        for edge in breadcrumbs:
            # The value is from the perspective of leafNode player
            # so the direction is positive for the leafNode player
            direction = 1 if edge.currPlayer == leafNode.currPlayer else -1
            edge.stats['N'] += 1
            edge.stats['W'] += v_evaluated * direction
            edge.stats['Q'] = edge.stats['W'] / float(edge.stats['N']) # Use float() for python2 compatibility


    def search(self):
        # Build Monte Carlo tree from root using lots of simulations
        for i in range(self.num_itr):
            leafNode, breadcrumbs = self.moveToLeaf()
            self.expandAndBackUp(leafNode, breadcrumbs)

        # Calculat PI and sample an edge
        chosen_edges = []
        maxN = float('-inf')

        for edge in self.root.edges:
            probability = pow(edge.stats['N'], (1. / self.tree_tau))
            checker_id = self.root.state.checkers_id[self.root.currPlayer][edge.fromPos]
            neural_net_index = utils.encode_checker_index(checker_id, edge.toPos)
            self.root.pi[neural_net_index] = probability

        self.root.pi /= np.sum(self.root.pi)

        # Sample an action with given probablities
        sampled_index = np.random.choice(np.arange(len(self.root.pi)), p=self.root.pi)
        sampled_checker_id, sampled_to = utils.decode_checker_index(sampled_index)
        sampled_from = self.root.state.checkers_pos[self.root.currPlayer][sampled_checker_id]

        # Get the edge corresponding to the sampled action
        sampled_edge = None
        for edge in self.root.edges:
            if edge.fromPos == sampled_from and edge.toPos == sampled_to:
                sampled_edge = edge
                break

        assert sampled_edge != None

        return self.root.pi, sampled_edge



if __name__ == '__main__':
    count = 0
    board = board.Board()
    node = Node(board, 1)
    model = ResidualCNN()
    tree = MCTS(node, model)

