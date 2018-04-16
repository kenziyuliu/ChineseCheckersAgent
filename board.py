import numpy as np
from constants import *
import board_utils
import operator

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
        # == Directions Map ==
        #
        #   NW north
        #  west     east
        #      south SE
        self.directions = [
            (-1, 0),    # north
            (0, 1),     # east
            (1, 1),     # southeast
            (1, 0),     # south
            (0, -1),    # west
            (-1, -1)    # northwest
        ]

        self.checkers_pos = [{},
                            {0: (BOARD_HEIGHT-1, 0), 1: (BOARD_HEIGHT-2, 0), 2: (BOARD_HEIGHT-1, 1),
                                 3: (BOARD_HEIGHT-3, 0), 4: (BOARD_HEIGHT-2, 1), 5: (BOARD_HEIGHT-1, 2)},
                            {0: (0, BOARD_WIDTH-1), 1: (1, BOARD_WIDTH-1), 2: (0, BOARD_WIDTH-2),
                                 3: (2, BOARD_WIDTH-1), 4: (1, BOARD_WIDTH-2), 5: (0, BOARD_WIDTH-3)}]

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


    def visualise(self, cur_player=None, gap_btw_checkers=3):
        """
        Prints the current board for human visualisation
        """
        print('=' * 75)
        print('Current Status:' + ' ' * 40 + 'Current Player: {}\n'.format(cur_player))

        cur_board = self._board[:, :, 0]        # Get current board from the topmost layer
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


    def valid_checker_moves(self, cur_player, checker_pos):
        """
        Returns all valid moves for one checker piece
        """
        result = []
        # map to check already explored moves
        check_map = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype='uint8')
        # expand to each directions without jump
        result.append(checker_pos)
        check_map[checker_pos] = 1
        for walk_dir in self.directions:
            (row, col) = tuple(map(operator.add, checker_pos, walk_dir))
            if not board_utils.is_valid_pos(row, col):
                continue
            if self._board[row, col, 0] == 0:
                result.append((row, col))
                check_map[row, col] = 1

        # check continous jump moves
        self.board[checker_pos[0], checker_pos[1]] = 0;
        self.valid_checker_jump_moves(result, check_map, checker_pos)
        self.board[checker_pos[0], checker_pos[1]] = cur_player;
        result.remove(checker_pos)
        return result


    def valid_checker_jump_moves(self, valid_moves, check_map, checker_pos):
        """
        Add all recursive jumping moves into the list of valid moves
        """
        curr_row, curr_col = checker_pos
        # expand with jump
        for walk_dir in self.directions:
            step = 1
            row_inc, col_inc = walk_dir
            row, col = curr_row + row_inc, curr_col + col_inc
            valid_pos = True

            # Go along the direction to find the first checker and record steps
            while True:
                if not board_utils.is_valid_pos(row, col):
                    valid_pos = False
                    break
                if self.board[row, col, 0] != 0:
                    break
                step += 1
                row += row_inc
                col += col_inc

            if not valid_pos:
                continue

            # Continue in the direction to find the mirror move
            for i in range(step):
                row += row_inc
                col += col_inc
                if not board_utils.is_valid_pos(row, col) or self.board[row, col, 0] != 0:
                    valid_pos = False
                    break

            if not valid_pos:
                continue

            # get the row and col ready to jump
            # check whether the destination is visited
            if check_map[row, col] == 1:
                continue

            # store moves
            valid_moves.append((row, col))
            check_map[row][col] = 1
            self.valid_checker_jump_moves(valid_moves, check_map, (row, col))


    def get_valid_moves(self, cur_player):
        """
        Returns the collection of valid moves given the current player
        """
        valid_moves_set = {}
        for checker_pos in self.checkers_pos[cur_player].values():
            valid_moves_set[checker_pos] = self.valid_checker_moves(cur_player, checker_pos)
        return valid_moves_set


    def place(self, cur_player, origin_pos, dest_pos):
        """
        Makes a move with array indices
        """
        cur_board = np.copy(self._board[:, :, 0])
        cur_board[origin_pos], cur_board[dest_pos] = cur_board[dest_pos], cur_board[origin_pos]

        # Move the checker
        for checker_num, checker_pos in self.checkers_pos[cur_player].items():
            if (checker_pos == origin_pos):
                self.checkers_pos[cur_player][checker_num] = dest_pos
        

        # Update history
        self._board = np.concatenate((np.expand_dims(cur_board, axis=2), self._board[:, :, :NUM_HIST_MOVES - 1]), axis=2)

        return self.check_win()


if __name__ == '__main__':
    """
    Put board.py testcases here
    """
    board = Board()

    # print(board.board[board.checker_pos[PLAYER_ONE][0][0],
    # board.checker_pos[PLAYER_ONE][0][1], 0])

    # print(board.board[6, 0, 0])
    # print(board.board)
    # board.print_board()
    print(board.get_valid_moves(1))
    # print(board.check_win())
    # print(board.check_win())
    # for i in range(50000):
