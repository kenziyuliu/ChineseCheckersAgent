import numpy as np
from player import HumanPlayer, GreedyPlayer, AiPlayer
from board import Board
from constants import *

class Game:
    def __init__(self, p1_type=None, p2_type=None, verbose=True):

        if not p1_type or not p2_type:
            p1_type, p2_type = self.get_player_types()

        p1_type = p1_type[0].lower()
        p2_type = p2_type[0].lower()

        if p1_type == 'h':
            self.player_one = HumanPlayer(player_num=1)
        elif p1_type == 'g':
            self.player_one = GreedyPlayer(player_num=1)
        else:
            self.player_one = AiPlayer(player_num=1)

        if p2_type == 'h':
            self.player_two = HumanPlayer(player_num=2)
        elif p2_type == 'g':
            self.player_two = GreedyPlayer(player_num=2)
        else:
            self.player_two = AiPlayer(player_num=2)

        self.cur_player = self.player_one
        self.next_player = self.player_two
        self.verbose = verbose
        self.board = Board()


    def get_player_types(self):
        p1_type = p2_type = ''
        while 1:
            p1_type = input('Enter player type of player 1 ([H]uman/[G]reedyRobot/[A]I): ')
            if p1_type[0].lower() in TYPES_OF_PLAYERS:
                break
            print('Invalid input. Try again.')

        while 1:
            p2_type = input('Enter player type of player 2 ([H]uman/[G]reedyRobot/[A]I): ')
            if p2_type[0].lower() in TYPES_OF_PLAYERS:
                break
            print('Invalid input. Try again.')

        return p1_type, p2_type


    def swap_players(self):
        self.cur_player, self.next_player = self.next_player, self.cur_player


    def start(self):
        # needed for later calling Model.to_model_input()
        while True:
            move_from, move_to = self.cur_player.decide_move(self.board, verbose=self.verbose)    # Get move from player
            winner = self.board.place(self.cur_player.player_num, move_from, move_to)  # Make the move on board and check winner
            if winner:
                break
            self.swap_players()

        if self.verbose:
            self.board.visualise()
        print('Player {} wins!'.format(winner))

        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO: once we have a winner, start network training procedure here    TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO

        return winner

if __name__ == '__main__':
    print('Do not run this file directly!')
