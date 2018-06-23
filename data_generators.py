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
        start_time = datetime.now()

        while True:
            best_moves = self.cur_player.decide_move(self.board, verbose=False, training=True)
            pi = np.zeros(NUM_CHECKERS * BOARD_WIDTH * BOARD_HEIGHT)

            for move in best_moves:
                start = board_utils.human_coord_to_np_index(move[0])
                end = board_utils.human_coord_to_np_index(move[1])
                checker_id = self.board.checkers_id[self.cur_player.player_num][start]
                neural_net_index = utils.encode_checker_index(checker_id, end)
                pi[neural_net_index] = 1.0 / len(best_moves)

            play_history.append((copy.deepcopy(self.board), pi))

            pick_start, pick_end = random.choice(best_moves)
            move_from = board_utils.human_coord_to_np_index(pick_start)
            move_to = board_utils.human_coord_to_np_index(pick_end)

            # Make the move on board and check winner
            winner = self.board.place(self.cur_player.player_num, move_from, move_to)
            if winner:
                final_winner = winner
                break

            # Check if game is stuck
            if datetime.now() - start_time > timedelta(seconds=STUCK_TIME_LIMIT):
                return play_history[:AVERAGE_TOTAL_MOVE], REWARD['draw']

            self.swap_players()

        reward = utils.get_p1_winloss_reward(self.board, final_winner)

        # Reset generator for next game
        self.board = Board(randomised=self.randomised)

        # Keep meaningful move history
        if self.randomised:
            return play_history[BOARD_HIST_MOVES:], reward
        else:
            return play_history, reward



if __name__ == "__main__":
    sum = 0
    for i in range(200):
        generator = GreedyDataGenerator(randomised=True)
        history, reward = generator.generate_play()
        sum += len(history)

        # history[0][0].visualise()
        # print(history[0][1])
        # history[1][0].visualise()
        # print(history[1][1])
        # history[2][0].visualise()
        # print(history[2][1])
        #
        # history[-3][0].visualise()
        # print(history[-3][1])
        # history[-2][0].visualise()
        # print(history[-2][1])
        # history[-1][0].visualise()
        # print(history[-1][1])
        #
        # import time
        # time.sleep(500)

        # print(len(generator.generate_play()[0]))
    print('Average game length:', sum / 200)
