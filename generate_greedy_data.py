from player import GreedyPlayer
from board import *
from model import *
from datetime import datetime
from config import *
import copy
import random
import board_utils
import numpy as np

class GreedyDataGenerator:
    def __init__(self, randomize=False):
        self.cur_player = GreedyPlayer(player_num=1)
        self.next_player = GreedyPlayer(player_num=2)
        self.board = Board()
        self.randomize = randomize
        if self.randomize:
            self.board.board[:, :, 0] = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype='uint8')
            position_list = []
            for row in range(BOARD_WIDTH):
                for col in range(BOARD_HEIGHT):
                    position_list.append((row, col))

            chosen_indexes = np.random.choice(len(position_list), size=NUM_CHECKERS*2,replace=False)
            chosen_position = []

            for chosen_index in chosen_indexes:
                chosen_position.append(position_list[chosen_index])

            self.board.checkers_pos = [None, {}, {}]
            self.board.checkers_id = [None, {}, {}]
            index = 0
            for player_num in [PLAYER_ONE, PLAYER_TWO]:
                for checker_id in range(NUM_CHECKERS):
                    self.board.board[chosen_position[index][0], chosen_position[index][1], 0] = player_num
                    self.board.checkers_pos[player_num][checker_id] = chosen_position[index]
                    self.board.checkers_id[player_num][chosen_position[index]] = checker_id
                    index += 1

    def swap_players(self):
        self.cur_player, self.next_player = self.next_player, self.cur_player

    def self_play(self):
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
                neural_net_index = Model.encode_checker_index(checker_id, end)
                pi[neural_net_index] = 1.0 / len(best_moves)

            # 2 is the threshold to keep meaningful move history
            if not self.randomize or count > THRESHOLD_FOR_RANDOMIZATION:
                play_history.append((copy.deepcopy(self.board), pi))

            pick_start, pick_end = random.choice(best_moves)
            move_from = board_utils.human_coord_to_np_index(pick_start)
            move_to = board_utils.human_coord_to_np_index(pick_end)

            winner = self.board.place(self.cur_player.player_num, move_from, move_to)  # Make the move on board and check winner
            if winner:
                final_winner = winner
                break

            self.swap_players()

            count += 1

            if (datetime.now() - start_time).seconds >= STUCK_TIME_LIMIT:
                return play_history[0:AVERAGE_TOTAL_MOVE], REWARD['draw']

        return play_history, final_winner

if __name__ == "__main__":
    for i in range(200):
        generator = GreedyDataGenerator(randomize=True)
        print(len(generator.self_play()[0]))
