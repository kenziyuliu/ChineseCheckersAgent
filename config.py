from keras import regularizers

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

''' MCTS and RL '''
PROGRESS_MOVE_LIMIT = 72
TREE_TAU = 1
REWARD = {"lose" : -10, "draw" : 0, "win" : 10}
C_PUCT = 1
MCTS_SIMULATIONS = 128
EPSILON = 1e-5
DIST_THRES_FOR_REWARD = 2                   # Threshold for reward for player forward distance difference
TOTAL_MOVES_TILL_TAU0 = 24

''' Dirichlet Noise '''
DIRICHLET_ALPHA = 0.03                      # Alpha for ~ Dir(), assuming symmetric Dirichlet distribution
DIR_NOISE_FACTOR = 0.25                     # Weight of Dirichlet noise on root prior probablities

''' Model '''
# Model input dimensions
INPUT_DIM = (BOARD_WIDTH, BOARD_HEIGHT, BOARD_HIST_MOVES * 2 + 1)
NUM_FILTERS = 64                            # Default number of filters for conv layers
NUM_RESIDUAL_BLOCKS = 16                    # Number of residual blocks in the model

''' Train '''
SAVE_MODELS_DIR = 'saved-models/'
MODEL_PREFIX = 'version'
SAVE_TRAIN_DATA_DIR = 'generated-training-data/'
SAVE_TRAIN_DATA_PREF = 'data-for-iter-'
PAST_ITER_COUNT = 3                         # Number of past iterations to use
NUM_WORKERS = 12                            # For generating self plays in parallel
NUM_SELF_PLAY = 60                          # Total number of self plays to generate
DEF_DATA_RETENTION_RATE = 0.5              # Default percentage of training data to keep when sampling
BATCH_SIZE = 32
REG_CONST = 6e-5                            # Weight decay constant (l1/l2 regularizer)
LEARNING_RATE = 0.001                       # Traning learning rate
REGULARIZER = regularizers.l2(REG_CONST)    # Default kernal regularizer
EPOCHS = 20                                 # Training Epochs

''' Greedy Data Generator '''
THRESHOLD_FOR_RANDOMIZATION = 2
AVERAGE_TOTAL_MOVE = 43
STUCK_TIME_LIMIT = 5





