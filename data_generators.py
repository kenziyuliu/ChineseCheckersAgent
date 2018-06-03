import copy
import random
import numpy as np

import utils
from player import GreedyPlayer
from board import *
from datetime import datetime, timedelta
from config import *
import board_utils
from model import Model


class GreedyDataGenerator:
    def __init__(self, randomised=False):
        self.cur_player = GreedyPlayer(player_num=1)
        self.next_player = GreedyPlayer(player_num=2)
        self.randomised = randomised
        self.board = Board(randomised=randomised)


    def swap_players(self):
        self.cur_player, self.next_player = self.next_player, self.cur_player


    def generate_play(self):
        play_history = []
        final_winner = None
        count = 0
        start_time = datetime.now()

        while True:
            best_moves = self.cur_player.decide_move(self.board, verbose=False, training=True)
            pi = np.zeros(NUM_CHECKERS * BOARD_WIDTH * BOARD_HEIGHT, dtype='float64')

            for move in best_moves:
                start = board_utils.human_coord_to_np_index(move[0])
                end = board_utils.human_coord_to_np_index(move[1])
                checker_id = self.board.checkers_id[self.cur_player.player_num][start]
                neural_net_index = utils.encode_checker_index(checker_id, end)
                pi[neural_net_index] = 1.0 / len(best_moves)

            # 2 is the threshold to keep meaningful move history
            if not self.randomised or count > THRESHOLD_FOR_RANDOMIZATION:
                play_history.append((copy.deepcopy(self.board), pi))

            pick_start, pick_end = random.choice(best_moves)
            move_from = board_utils.human_coord_to_np_index(pick_start)
            move_to = board_utils.human_coord_to_np_index(pick_end)

            winner = self.board.place(self.cur_player.player_num, move_from, move_to)  # Make the move on board and check winner
            if winner:
                final_winner = winner
                break

            # Check if game is stuck
            if datetime.now() - start_time > timedelta(seconds=STUCK_TIME_LIMIT):
                return play_history[:AVERAGE_TOTAL_MOVE], REWARD['draw']

            self.swap_players()
            count += 1

        reward = utils.get_p1_winloss_reward(self.board, final_winner)

        # Reset generator for next game
        self.board = Board(randomised=self.randomised)

        return play_history, reward



if __name__ == "__main__":
    for i in range(200):
        generator = GreedyDataGenerator(randomised=True)
        print(len(generator.generate_play()[0]))
