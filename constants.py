''' Player '''
# Fixed 2 player
PLAYER_ONE = 1
PLAYER_TWO = 2

''' Board/Game '''
ROWS_OF_CHECKERS = 3
NUM_CHECKERS = (1 + ROWS_OF_CHECKERS) * ROWS_OF_CHECKERS // 2
NUM_DIRECTIONS = 6
BOARD_WIDTH = BOARD_HEIGHT = ROWS_OF_CHECKERS * 2 + 1
NUM_HIST_MOVES = 3      # Number of history moves to keep
TYPES_OF_PLAYERS = ['h', 'g', 'a']

''' MCTS and RL '''
PROGRESS_MOVE_LIMIT = 50
TREE_TAU = 1
REWARD = {"lose" : -1, "draw" : 0, "win" : 1}
C_PUCT = 1
MCTS_SIMULATIONS = 20
EPSILON = 1e-4

''' Train '''
NUM_THREADS = 1
NUM_SELF_PLAY = 1
