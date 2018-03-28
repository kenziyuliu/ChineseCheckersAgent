import numpy as np
import board_utils
from constants import *
import copy

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
        board.visualise(cur_player=self._player_num)

        valid_moves = board.valid_moves(self._player_num)
        human_valid_moves = dict()

        for key in valid_moves:
            human_valid_moves[board_utils.np_index_to_human_coord(key)] = [board_utils.np_index_to_human_coord(to) for to in valid_moves[key]]

        for checker in human_valid_moves:
            print("{} can move to: {}".format(checker, human_valid_moves[checker]))
            
        print()

        (from_i, from_j), (to_i, to_j) = (-1, -1), (-1, -1)
        while 1:
            # x = the row number on visualised board, y = the position of the checker in that row from left
            from_coord = map(int, input('Which checker to move? Specify row number and count from left, separated by a space: ').split())
            to_coord = map(int, input('where to move this checker? Specify row number and count from left, separated by a space: ').split())

            # NOTE: the above `map` objects "from_coord", "to_coord" can only be used once

            (from_i, from_j), (to_i, to_j) = board_utils.human_coord_to_np_index(tuple(from_coord)), \
                                             board_utils.human_coord_to_np_index(tuple(to_coord))

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

