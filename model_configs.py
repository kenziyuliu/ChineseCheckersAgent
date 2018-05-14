from constants import *
from keras import regularizers

''' Model '''
# Model input dimensions
INPUT_DIM = (BOARD_WIDTH, BOARD_HEIGHT, NUM_HIST_MOVES * 2 + 1)
# default number of filters for conv layers
NUM_FILTERS = 64
# number of residual blocks in the model
NUM_RESIDUAL_BLOCKS = 20

''' Training '''
# Training batch size
BATCH_SIZE = 32
# Weight decay constant (l1/l2 regularizer)
REG_CONST = 5e-5
# Traning learning rate
LEARNING_RATE = 0.001
# Default kernal regularizer
REGULARIZER = regularizers.l2(REG_CONST)
# Training Epochs
EPOCHS = 30
