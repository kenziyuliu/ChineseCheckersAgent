import numpy as np
import board
import model
import copy
import random
from constants import *

class Node:
	def __init__(self, state, currPlayer):
		self.state = state
		self.currPlayer = currPlayer
		self.edges = []
		self.pai = 0

	def isLeaf(self):
		if len(self.edges) > 0:
			return False
		else:
			return True

class Edge:
	def __init__(self, inNode, outNode, prior, fromPos, toPos):
		self.inNode = inNode
		self.outNode = outNode
		self.currPlayer = inNode.currPlayer
		self.fromPos = fromPos
		self.toPos = toPos

		self.stats =  {
					'N': 0,
					'W': 0,
					'Q': 0,
					'P': prior,
				}

class MCTS():

	def __init__(self, root, cpuct, num_itr):
		self.root = root
		self.tree = {}
		self.cpuct = cpuct
		self.num_itr = num_itr

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
				U = self.cpuct * edge.stats['P'] * N_sum / (1 + edge.stats['N'])
				if edge.stats['Q'] + U > maxQU:
					maxQU = edge.stats['Q'] + U

			for edge in currentNode.edges:
				U = self.cpuct * edge.stats['P'] * N_sum / (1 + edge.stats['N'])
				if edge.stats['Q'] + U == maxQU:
					chosen_edges.append(edge)

			sampled_edge = random.choice(chosen_edges)

			breadcrumbs.append(sampled_edge)

			currentNode = sampled_edge.outNode

		return currentNode, breadcrumbs


	def expandAndBackUp(self, leafNode, breadcrumbs):

		hasWin = leafNode.state.check_win()

		if hasWin != 0:
			for edge in breadcrumbs:
				edge.stats['N'] += 1
				edge.stats['W'] += REWARD["win"]
				edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
			return

		# p_evaluated, v_evaluated = model.predict(leafNode.state)
		p_evaluated = np.random.rand(6*49)
		v_evaluated = random.uniform(-1, 1)
		valid_actions = leafNode.state.get_valid_moves(leafNode.currPlayer)

		for checker_piece_pos, action_set in valid_actions.items():
			checker_piece_id = leafNode.state.checkers_id[leafNode.currPlayer][checker_piece_pos]
			for destination_pos in action_set:
				prior_index = model.Model.encode_checker_index(checker_piece_id, destination_pos)
				next_player = PLAYER_ONE + PLAYER_TWO - leafNode.currPlayer
				next_state = copy.deepcopy(leafNode.state)
				newFromPos = leafNode.state.checkers_pos[leafNode.currPlayer][checker_piece_id]
				next_state.place(leafNode.currPlayer, newFromPos, destination_pos)
				newNode = Node(next_state, next_player)
				newEdge = Edge(leafNode, newNode, p_evaluated[prior_index], newFromPos, destination_pos)
				leafNode.edges.append(newEdge)


		for edge in breadcrumbs:
			edge.stats['N'] += 1
			edge.stats['W'] += v_evaluated
			edge.stats['Q'] = edge.stats['W'] / edge.stats['N']


	def selfPlay(self):
		curr_progress1 = 0
		curr_progress2 = 0
		count = 0
		player_turn = 0
		actual_play_history = []

		while True:

			for i in range(self.num_itr):
				leafNode, breadcrumbs = self.moveToLeaf()
				self.expandAndBackUp(leafNode, breadcrumbs)

			chosen_edges = []
			maxN = float("-inf")
			sumPai = 0

			for edge in self.root.edges:
				maxN = max(maxN, edge.stats['N'])
				sumPai += edge.stats['N'] ** (1/TREE_TAU)

			for edge in self.root.edges:
				if edge.stats['N'] == maxN:
					chosen_edges.append(edge)

			sampled_edge = random.choice(chosen_edges)

			self.root.edges = []
			self.root.edges.append(sampled_edge)
			self.root.pai = (sampled_edge.stats['N'] ** (1/TREE_TAU)) / sumPai

			actual_play_history.append(self.root)

			self.root = sampled_edge.outNode
			self.root.edges = []

			if player_turn % 2 == 0:
				progress_evaluated = self.root.state.player_progress(PLAYER_ONE)
				if progress_evaluated == curr_progress1:
					count += 1
				else:
					count = 0
					curr_progress1 = progress_evaluated

			else:
				progress_evaluated = self.root.state.player_progress(PLAYER_TWO)
				if progress_evaluated == curr_progress2:
					count += 1
				else:
					count = 0
					curr_progress2 = progress_evaluated

			if count >= PROGRESS_MOVE_LIMIT:
				break

			if self.root.state.check_win() != 0:
				actual_play_history.append(self.root)
				break

		return actual_play_history, self.root.state.check_win()


	def search(self):
		for i in range(self.num_itr):
			leafNode, breadcrumbs = self.moveToLeaf()
			self.expandAndBackUp(leafNode, breadcrumbs)

		chosen_edges = []
		maxN = float("-inf")
		sumPai = 0

		for edge in self.root.edges:
			maxN = max(maxN, edge.stats['N'])
			sumPai += edge.stats['N'] ** (1/TREE_TAU)
			print(edge.fromPos, edge.toPos)

		for edge in self.root.edges:
			if edge.stats['N'] == maxN:
				chosen_edges.append(edge)

		sampled_edge = random.choice(chosen_edges)
		return sampled_edge.fromPos, sampled_edge.toPos

if __name__ == '__main__':
	count = 0
	board = board.Board()
	while True:
		node = Node(board, 1)
		tree = MCTS(node, 1, 5)
		if tree.selfPlay()[1] != 0:
			break
		count += 1
		print("Calulating...")

			








