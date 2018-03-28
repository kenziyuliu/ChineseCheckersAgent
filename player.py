import numpy as np
import board_utils
from constants import *

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
        print('='*75)
        print('Current player: {}'.format(self._player_num))
        board.visualise()

        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # FIXME: Use "board.valid_moves()" to check for human move validity
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

        # x = the row number on visualised board, y = the position of the checker in that row from left
        from_coord = map(int, input('Which checker to move? Specify row number and count from left, separated by a space: ').split())
        to_coord = map(int, input('where to move this checker? Specify row number and count from left, separated by a space: ').split())

        return board_utils.human_coord_to_np_index(tuple(from_coord)), \
               board_utils.human_coord_to_np_index(tuple(to_coord))



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

