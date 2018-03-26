import numpy as np

# Fixed 2 player
PLAYER_ONE = 1
PLAYER_TWO = 2

ROWS_OF_CHECKER = 3
BOARD_WIDTH = ROWS_OF_CHECKER * 2 + 1
BOARD_HEIGHT = ROWS_OF_CHECKER * 2 + 1
NUM_HIST_MOVES = 3      # Number of history moves to keep

class Board:

    def __init__(self):
        self._board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, NUM_HIST_MOVES))  # Initialize empty board

    @property
    def board(self):
        """ Get the numpy array representing this board """
        return self._board

    def check_win(self):
        """ Returns the winner given the current board state; -1 if game still going """
        cur_board = self._board[:, :, 0]
        # TODO: Check if there is a winner given the current board
        pass

    def print_board(self):
        cur_board = self._board[:, :, 0]
        # TODO: visualise the current board in a nice way

    def valid_moves(self, cur_player):
        # TODO: return a list of valid moves given the current players
        pass



if __name__ == '__main__':
    board = Board()
    print(board.board)
    board.print_board()
    board.check_win()