''' Player '''
# Fixed 2 player
PLAYER_ONE = 1
PLAYER_TWO = 2

''' Board/Game '''
ROWS_OF_CHECKERS = 3
NUM_CHECKERS = (1 + ROWS_OF_CHECKERS) * ROWS_OF_CHECKERS // 2
NUM_DIRECTIONS = 6
BOARD_WIDTH = BOARD_HEIGHT = ROWS_OF_CHECKERS * 2 + 1
BOARD_HIST_MOVES = 3                          # Number of history moves to keep
TYPES_OF_PLAYERS = ['h', 'g', 'a']
PLAYER_ONE_DISTANCE_OFFSET = 70
PLAYER_TWO_DISTANCE_OFFSET = -14
TOTAL_HIST_MOVES = 16                       # Total number of history moves to keep for checking repetitions
UNIQUE_DEST_LIMIT = 3

''' Dirichlet Noise '''
DIRICHLET_ALPHA = 0.03                      # Alpha for ~ Dir(), assuming symmetric Dirichlet distribution
DIR_NOISE_FACTOR = 0.25                     # Weight of Dirichlet noise on root prior probablities

''' Model '''
# Model input dimensions
INPUT_DIM = (BOARD_WIDTH, BOARD_HEIGHT, BOARD_HIST_MOVES * 2 + 1)
NUM_FILTERS = 64                            # Default number of filters for conv layers
NUM_RESIDUAL_BLOCKS = 12                    # Number of residual blocks in the model

''' MCTS and RL '''
PROGRESS_MOVE_LIMIT = 100
REWARD = {'lose' : -1, 'draw' : 0, 'win' : 1}
REWARD_FACTOR = 10                          # Scale the reward if necessary
TREE_TAU = 1
DET_TREE_TAU = 0.01
C_PUCT = 3.5
MCTS_SIMULATIONS = 150
EPSILON = 1e-5
TOTAL_MOVES_TILL_TAU0 = 16
DIST_THRES_FOR_REWARD = 2                   # Threshold for reward for player forward distance difference
EVAL_GAMES = 24

''' Loss Weights depending on training '''
LOSS_WEIGHTS = { 'policy_head': 1., 'value_head': 1. }

''' Train '''
SAVE_MODELS_DIR = 'saved-models/'
SAVE_WEIGHTS_DIR = 'saved-weights/'
MODEL_PREFIX = 'version'
SAVE_TRAIN_DATA_DIR = 'generated-training-data/'
SAVE_TRAIN_DATA_PREF = 'data-for-iter-'
PAST_ITER_COUNT = 1                         # Number of past iterations to use
DEF_DATA_RETENTION_RATE = 0.5               # Default percentage of training data to keep when sampling
BATCH_SIZE = 32
REG_CONST = 6e-3                            # Weight decay constant (l1/l2 regularizer)
LEARNING_RATE = 0.0001                       # Traning learning rate
EPOCHS = 5                                 # Training Epochs
NUM_SELF_PLAY = 120                          # Total number of self plays to generate
NUM_WORKERS = 12                            # For generating self plays in parallel
SELF_PLAY_DIFF_MODEL = False

''' Greedy-Supervised Training '''
# G_NUM_GAMES = 60000
G_AVG_GAME_LEN = 21
G_DATA_RETENTION_RATE = 1. / G_AVG_GAME_LEN
G_EPOCHS = 100
G_GAMES_PER_EPOCH = 15000
G_VAL_SPLIT = 0.1
G_NORMAL_GAME_RATIO = 0.01
G_MODEL_PREFIX = 'greedy-model'
G_BATCH_SIZE = 32
# G_NUM_VAL_DATA = 3000

''' Greedy Data Generator '''
THRESHOLD_FOR_RANDOMIZATION = 2
AVERAGE_TOTAL_MOVE = 43
STUCK_TIME_LIMIT = 0.1

"""
NOTE:
When training in greedy:
    - use lower regulurisation (1e-4)
    - 1 weight on value head

When training in RL:
    - higher REG (6e-3)
    - 1. weight on both value and policy head
    - ~5 Epoch
    - 12 Workers
    - ~144 MCTS Simulations
    - TREE_TAU = 1

When testing:
    - Use DET_TREE_TAU, which is already set in Game.py  (Or TREE_TAU = 0.01)
    - MCST Similations = depending on need

"""
