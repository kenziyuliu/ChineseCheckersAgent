import numpy as np
from constants import *
import board_utils

class Board:
    def __init__(self):
        self._board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, NUM_HIST_MOVES), dtype='uint8')  # Initialize empty board
        self._board[:, :, 0] = np.array([[0, 0, 0, 0, 2, 2, 2],
                                         [0, 0, 0, 0, 0, 2, 2],
                                         [0, 0, 0, 0, 0, 0, 2],
                                         [0, 0, 0, 0, 0, 0, 0],
                                         [1, 0, 0, 0, 0, 0, 0],
                                         [1, 1, 0, 0, 0, 0, 0],
                                         [1, 1, 1, 0, 0, 0, 0]])

        self.checker_pos = [[],
                            [(BOARD_HEIGHT-1, 0), (BOARD_HEIGHT-2, 0), (BOARD_HEIGHT-1, 1),
                             (BOARD_HEIGHT-3, 0), (BOARD_HEIGHT-2, 1), (BOARD_HEIGHT-1, 2)],
                            [(0, BOARD_WIDTH-1), (1, BOARD_WIDTH-1), (0, BOARD_WIDTH-2),
                             (2, BOARD_WIDTH-1), (1, BOARD_WIDTH-2), (0, BOARD_WIDTH-3)]]

    @property
    def board(self):
        """
        Get the numpy array representing this board.
        Array is shaped 7x7x3, where the first 7x7 plane
        is the current board, while the latter are the
        two previous steps.
        PLAYER_ONE and PLAYER_TWO's checkers are initialised
        at bottom left and top right corners respectively.
        """
        return self._board


    def check_win(self):
        """
        Returns the winner given the current board state; 0 if game still going
        To win:
            player 1: all checkers to upper right
            player 2: all checkers to lower left
        """
        cur_board = self._board[:, :, 0]
        one_win = two_win = True
        for k in range(BOARD_WIDTH - ROWS_OF_CHECKERS, BOARD_WIDTH):
            if one_win:
                up_diag = cur_board.diagonal(k)
                if not np.array_equal(up_diag, [PLAYER_ONE]*len(up_diag)):
                    one_win = False
            if two_win:
                down_diag = cur_board.diagonal(-k)
                if not np.array_equal(down_diag, [PLAYER_TWO]*len(down_diag)):
                    two_win = False

            if not one_win and not two_win:
                return 0
        return PLAYER_ONE if one_win else PLAYER_TWO


    def visualise(self, gap_btw_checkers=3):
        """ Prints the current board for human visualisation """
        print('=' * 75)
        print('Current Status:\n')

        cur_board = self._board[:, :, 0]    # Get current board from the topmost layer
        visual_width = BOARD_WIDTH * (gap_btw_checkers + 1) - gap_btw_checkers
        visual_height = BOARD_HEIGHT * 2 - 1    # Dimensions for visualisation
        leading_spaces = visual_width // 2

        for i in range(1, visual_height + 1):
            # Number of slots in the board row
            num_slots = i if i <= BOARD_WIDTH else visual_height - i + 1
            print('\tRow {:2}\t\t'.format(i), end='')
            # Print leading spaces
            print(' ' * ((leading_spaces - (num_slots - 1) * ((gap_btw_checkers + 1) // 2))), end='')
            print((' ' * gap_btw_checkers).join(map(str, cur_board.diagonal(BOARD_WIDTH - i))), end='\n\n')  # Board contents

        print('=' * 75)

    def valid_checker_moves(self, cur_player, checker_id):
        """ Returns all valid moves for one checker piece"""
        curr_row = self.checker_pos[cur_player][checker_id][0]
        curr_col = self.checker_pos[cur_player][checker_id][1]
        result = []
        # map to check already explored moves 
        checkMap = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype='uint8')
        result.append((curr_row, curr_col))
        checkMap[curr_row][curr_col] = 1

        # check direct up move
        if (curr_row != 0 and self.board[curr_row-1, curr_col, 0] == 0):
            result.append((curr_row-1, curr_col))
            checkMap[curr_row-1][curr_col] = 1
        # check direct down move
        if (curr_row != BOARD_HEIGHT-1 and self.board[curr_row+1, curr_col, 0] == 0):
            result.append((curr_row+1, curr_col))
            checkMap[curr_row+1][curr_col] = 1
        # check direct left move
        if (curr_col !=0 and self.board[curr_row, curr_col-1, 0] == 0):
            result.append((curr_row, curr_col-1))
            checkMap[curr_row][curr_col-1] = 1
        # check direct right move
        if (curr_col != BOARD_WIDTH-1 and self.board[curr_row, curr_col+1, 0] == 0):
            result.append((curr_row, curr_col+1))
            checkMap[curr_row][curr_col+1] = 1
        # check direct upLeft move
        if (curr_row != 0 and curr_col !=0 and self.board[curr_row-1, curr_col-1, 0] == 0):
            result.append((curr_row-1, curr_col-1))
            checkMap[curr_row-1][curr_col-1] = 1
        # check direct downRight move
        if (curr_row != BOARD_HEIGHT-1 and curr_col != BOARD_WIDTH-1 and self.board[curr_row+1, curr_col+1, 0] == 0):
            result.append((curr_row+1, curr_col+1))
            checkMap[curr_row+1][curr_col+1] = 1
        # check continous jump moves
        self.jump_recursion_helper(result, checkMap, (curr_row, curr_col))
        return result

    def jump_recursion_helper(self, valid_moves_set, checkMap, position):
        """ Add all recursive jumping moves into the valid_moves_set"""
        curr_row = position[0]
        curr_col = position[1]
        # check up jump and recursion
        if (curr_row-2 >= 0 and checkMap[curr_row-2][curr_col] == 0 and self.board[curr_row-1, curr_col, 0] != 0 and self.board[curr_row-2, curr_col, 0] == 0):
            valid_moves_set.append((curr_row-2, curr_col))
            checkMap[curr_row-2][curr_col] = 1
            self.jump_recursion_helper(valid_moves_set, checkMap, (curr_row-2, curr_col))
        # check down jump and recursion
        if (curr_row+2 <= BOARD_HEIGHT-1 and checkMap[curr_row+2][curr_col] == 0 and self.board[curr_row+1, curr_col, 0] != 0 and self.board[curr_row+2, curr_col, 0] == 0):
            valid_moves_set.append((curr_row+2, curr_col))
            checkMap[curr_row+2][curr_col] = 1
            self.jump_recursion_helper(valid_moves_set, checkMap, (curr_row+2, curr_col))
        # check left jump and recursion
        if (curr_col-2 >= 0 and checkMap[curr_row][curr_col-2] == 0 and self.board[curr_row, curr_col-1, 0] != 0 and self.board[curr_row, curr_col-2, 0] == 0):
            valid_moves_set.append((curr_row, curr_col-2))
            checkMap[curr_row][curr_col-2] = 1
            self.jump_recursion_helper(valid_moves_set, checkMap, (curr_row, curr_col-2))
        # check right jump and recursion
        if (curr_col+2 <= BOARD_WIDTH-1 and checkMap[curr_row][curr_col+2] == 0 and self.board[curr_row, curr_col+1, 0] != 0 and self.board[curr_row, curr_col+2, 0] == 0):
            valid_moves_set.append((curr_row, curr_col+2))
            checkMap[curr_row][curr_col+2] = 1
            self.jump_recursion_helper(valid_moves_set, checkMap, (curr_row, curr_col+2))
        # check upLeft jump and recursion
        if (curr_row-2 >= 0 and curr_col-2 >= 0 and checkMap[curr_row-2][curr_col-2] == 0 and self.board[curr_row-1, curr_col-1, 0] != 0 and self.board[curr_row-2, curr_col-2, 0] == 0):
            valid_moves_set.append((curr_row-2, curr_col-2))
            checkMap[curr_row-2][curr_col-2] = 1
            self.jump_recursion_helper(valid_moves_set, checkMap, (curr_row-2, curr_col-2))
        # check downRight jump and recursion
        if (curr_row+2 <= BOARD_HEIGHT-1 and curr_col+2 <= BOARD_WIDTH-1 and checkMap[curr_row+2][curr_col+2] == 0 and self.board[curr_row+1, curr_col+1, 0] != 0 and self.board[curr_row+2, curr_col+2, 0] == 0):
            valid_moves_set.append((curr_row+2, curr_col+2))
            checkMap[curr_row+2][curr_col+2] = 1
            self.jump_recursion_helper(valid_moves_set, checkMap, (curr_row+2, curr_col+2))
    
    def valid_moves(self, cur_player):
        """ Returns the list of valid moves given the current player """
        valid_moves_set = {}
        for i in range(0, 6):
            valid_moves_set[self.checker_pos[cur_player][i]] = self.valid_checker_moves(cur_player, i)
        return valid_moves_set


    def place(self, origin_pos, dest_pos):
        """ Makes a move with array indices """
        # TODO: move a chess piece from its original position to a destination in machine indexing system
        # TODO FIXME: Record history to board!
        cur_board = self._board[:, :, 0]
        cur_board[origin_pos], cur_board[dest_pos] = cur_board[dest_pos], cur_board[origin_pos]

        return self.check_win()


if __name__ == '__main__':
    board = Board()

    # print(board.board[board.checker_pos[PLAYER_ONE][0][0],
    # board.checker_pos[PLAYER_ONE][0][1], 0])

    # print(board.board[6, 0, 0])
    # print(board.board)
    # board.print_board()
    print(board.valid_moves(1))
    # print(board.check_win())
    # print(board.check_win())
    # for i in range(50000):

