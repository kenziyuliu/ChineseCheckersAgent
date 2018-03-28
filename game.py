import numpy as np
from player import HumanPlayer, AiPlayer
from board import Board
from constants import *

class Game:
    def __init__(self, p1_type=None, p2_type=None):

        if not p1_type or not p2_type:
            p1_type, p2_type = self.get_player_types()

        self.player_one = HumanPlayer(player_num=1) if p1_type[0].lower() == 'h' else AiPlayer(player_num=1)
        self.player_two = HumanPlayer(player_num=2) if p2_type[0].lower() == 'h' else AiPlayer(player_num=2)
        self.cur_player = self.player_one
        self.next_player = self.player_two

        self.board = Board()

    def get_player_types(self):
        p1_type = p2_type = ''
        while 1:
            p1_type = input('Enter player type of player 1 ([H]uman/[R]obot): ')
            if p1_type[0].lower() in ['h', 'r']:
                break
            print('Invalid input. Try again.')

        while 1:
            p2_type = input('Enter player type of player 2 ([H]uman/[R]obot): ')
            if p2_type[0].lower() in ['h', 'r']:
                break
            print('Invalid input. Try again.')

        return p1_type, p2_type


    def swap_players(self):
        self.cur_player, self.next_player = self.next_player, self.cur_player


    def start(self):
        while True:
            move_from, move_to = self.cur_player.decide_move(self.board)    # Get move from player
            winner = self.board.place(move_from, move_to)                   # Make the move on board and check winner
            if winner:
                break
            self.swap_players()

        self.board.visualise()
        print('Player {} wins!'.format(winner))

        # TODO: once we have a winner, start network training procedure here


if __name__ == '__main__':
    game = Game()
    game.start()
