import os
import re
import h5py
import datetime
import numpy as np
from sys import getsizeof
from collections import Mapping, Container

from config import *


def find_version_given_filename(filename):
    pattern = '({}|{})([0-9]{{4}})(-weights|)\.h5'.format(MODEL_PREFIX, G_MODEL_PREFIX)
    matches = re.search(pattern, filename)
    if matches is None:
        print('No 4-digit version number found in filename "{}"!'.format(filename))
        return -1

    return int(matches.group(2))


def get_model_path_from_version(version):
    return '{}/{}{:0>4}.h5'.format(SAVE_MODELS_DIR, MODEL_PREFIX, version)


def cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stress_message(message, extra_newline=False):
    print('{2}{0}\n{1}\n{0}{2}'.format('='*len(message), message, '\n' if extra_newline else ''))


def get_p1_winloss_reward(board, winner=None):
    """
    Return the reward for player one in the game, given the final board state
    """
    winner = winner or board.check_win()
    if winner == PLAYER_ONE:
        return REWARD['win']
    elif winner == PLAYER_TWO:
        return REWARD['lose']
    else:
        return REWARD['draw']



def save_train_data(board_x, pi_y, v_y, version):
    ''' Write current iteration training data to disk '''
    if not os.path.exists(SAVE_TRAIN_DATA_DIR):
        os.makedirs(SAVE_TRAIN_DATA_DIR)

    with h5py.File('{}/{}{}.h5'.format(SAVE_TRAIN_DATA_DIR, SAVE_TRAIN_DATA_PREF, version), 'w') as H:
        H.create_dataset('board_x', data=board_x)
        H.create_dataset('pi_y', data=pi_y)
        H.create_dataset('v_y', data=v_y)



def convert_to_train_data(self_play_games):
    ''' Return python lists containing training data '''
    board_x, pi_y, v_y = [], [], []
    for game in self_play_games:
        history, reward = game
        curr_player = PLAYER_ONE
        for board, pi in history:
            board_x.append(to_model_input(board, curr_player))
            pi_y.append(pi)
            v_y.append(reward)
            reward = -reward
            curr_player = PLAYER_ONE + PLAYER_TWO - curr_player

    return board_x, pi_y, v_y



def augment_train_data(board_x, pi_y, v_y):
    ''' Augment training data by horizontal flipping of the board '''
    new_board_x, new_pi_y, new_v_y = [], [], []
    for i in range(len(board_x)):
        new_board = np.copy(board_x[i])
        new_pi = np.copy(pi_y[i])
        new_v = v_y[i]

        # Flip the board along the other diagonal, in the last dimension
        for j in range(new_board.shape[-1]):
            new_board[:, :, j] = np.fliplr(np.rot90(new_board[:, :, j]))

        new_board_x.append(new_board)
        new_pi_y.append(new_pi)
        new_v_y.append(new_v)

    board_x += new_board_x
    pi_y += new_pi_y
    v_y += new_v_y

    return board_x, pi_y, v_y   # Return the same references



def to_model_input(board, cur_player):
    """
    Input:
        board: 7 x 7 x 3 board._board. Each channel contains positions of both players' checkers.
        cur_player: player number of the current player
    Output:
        7 x 7 x 7. First 3 channel is player 1, next 3 channel is player 2, last channel is all 0 if player 1 is to play.
    """
    # initialise the model input
    model_input = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, BOARD_HIST_MOVES * 2 + 1)) # may change dtype afterwards
    # get np array board
    new_board = board.board
    # get history moves
    hist_moves = board.hist_moves
    # get opponent player
    op_player = PLAYER_ONE + PLAYER_TWO - cur_player

    # firstly, construct the current state layers
    op_layer = np.copy(new_board[:, :, 0])
    cur_layer = np.copy(new_board[:, :, 0])
    # construct layer for current player
    np.putmask(cur_layer, cur_layer != cur_player, 0)
    for checker_id, checker_pos in board.checkers_pos[cur_player].items():
        cur_layer[checker_pos[0], checker_pos[1]] = checker_id + 1
    # construct layer for opponent player
    np.putmask(op_layer, op_layer != op_player, 0)
    for checker_id, checker_pos in board.checkers_pos[op_player].items():
        op_layer[checker_pos[0], checker_pos[1]] = checker_id + 1

    model_input[:, :, 0] = np.copy(cur_layer)
    model_input[:, :, 1] = np.copy(op_layer)

    # construct the latter layers
    moved_player = op_player
    hist_index = len(hist_moves) - 1
    for channel in range(1, BOARD_HIST_MOVES):
        if not np.any(new_board[:, :, channel]): # timestep < 0
            break
        move = hist_moves[hist_index]
        orig_pos = move[0]
        dest_pos = move[1]

        if moved_player == cur_player:
            value = cur_layer[dest_pos]
            cur_layer[dest_pos] = cur_layer[orig_pos]
            cur_layer[orig_pos] = value
        else:
            value = op_layer[dest_pos]
            op_layer[dest_pos] = op_layer[orig_pos]
            op_layer[orig_pos] = value

        hist_index -= 1
        moved_player = PLAYER_ONE + PLAYER_TWO - moved_player
        model_input[:, :, channel * 2] = np.copy(cur_layer)
        model_input[:, :, channel * 2 + 1] = np.copy(op_layer)

    if cur_player == PLAYER_TWO: # player 2 to play
        model_input[:, :, BOARD_HIST_MOVES * 2] = np.ones((BOARD_WIDTH, BOARD_HEIGHT))

    return model_input



def encode_checker_index(checker_id, coord):
    """
    Convert a checker and its destination
    to the model's output encoding.
    """
    region = checker_id * BOARD_WIDTH * BOARD_HEIGHT # get the element-block in the model's output
    offset = coord[0] * BOARD_WIDTH + coord[1]          # offset in this region
    return region + offset



def decode_checker_index(model_output_index):
    """
    Convert the index in the model's output vector
    to the checker number and its destination on board
    """
    checker_id = model_output_index // (BOARD_WIDTH * BOARD_HEIGHT)
    offset = model_output_index % (BOARD_WIDTH * BOARD_HEIGHT)
    dest = offset // BOARD_WIDTH, offset % BOARD_WIDTH
    return checker_id, dest



def softmax(input):
    ''' Compute the softmax (prediction) given input '''
    input = np.copy(input).astype('float64')
    input -= np.max(input, axis=-1, keepdims=True)  # For numerical stability
    exps = np.exp(input)
    return exps / np.sum(exps, axis=-1, keepdims=True)



def deepsizeof(obj, visited):
    d = deepsizeof
    if id(obj) in visited:
        return 0

    r = getsizeof(obj)
    visited.add(id(obj))

    if isinstance(obj, Mapping):
        r += sum(d(k, visited) + d(v, visited) for k, v in obj.items())

    if isinstance(obj, Container):
        r += sum(d(x, visited) for x in obj)

    return r


