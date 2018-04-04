import numpy as np
import board_utils
from constants import *
import copy
import os

"""
Both players must support decide_move(board) method
"""

class HumanPlayer:
    def __init__(self, player_num):
        self._player_num = player_num
        # TODO: include more members if needed


    @property
    def player_num(self):
        return self._player_num


    @player_num.setter
    def player_num(self, new_player_num):
        self._player_num = new_player_num


    def decide_move(self, board):
        """
        Given current board, return a move to play.
        :type board: Class Board
        :rtype A list of 2 tuples, specifying the move's FROM and TO.
        """

        # First print game info
        os.system('clear')
        board.visualise(cur_player=self._player_num)

        valid_moves = board.get_valid_moves(self._player_num)
        human_valid_moves = dict()

        for key in valid_moves:
            human_valid_moves[board_utils.np_index_to_human_coord(key)] = [board_utils.np_index_to_human_coord(to) for to in valid_moves[key]]

        for checker in human_valid_moves:
            print("Checker {} can move to: {}".format(checker, sorted(human_valid_moves[checker])))

        print()
        (from_i, from_j), (to_i, to_j) = (-1, -1), (-1, -1)
        while 1:
            # x = the row number on visualised board, y = the position of the checker in that row from left
            print('You should specify position by row number and the count from left.')
            print('Please input your move with format: start_row start_col end_row end_col')

            try:
                human_from_row, human_from_col, human_to_row, human_to_col = map(int, input().split())
            except ValueError:
                print("\nInvalid Move Format! Try again!")
                continue

            (from_i, from_j), (to_i, to_j) = board_utils.human_coord_to_np_index((human_from_row, human_from_col)), \
                                             board_utils.human_coord_to_np_index((human_to_row, human_to_col))

            if (from_i, from_j) in valid_moves and (to_i, to_j) in valid_moves[(from_i, from_j)]:
                break

            print("\nInvalid Move! Try again!")

        return (from_i, from_j), (to_i, to_j)



# TODO: Not Implemented
class AiPlayer:
    def __init__(self, player_num):
        self._player_num = player_num
        # TODO: include more members if needed

    @property
    def player_num(self):
        return self._player_num

    @player_num.setter
    def player_num(self, new_player_num):
        self._player_num = new_player_num

    def decide_move(self, board):
        """
        Given current board, return a move to play.
        :type board: Class Board
        :rtype A list of 2 tuples, specifying the move's FROM and TO.
        """
        # TODO
        pass
