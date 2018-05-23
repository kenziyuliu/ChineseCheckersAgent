import re
import os
import sys
import h5py
import datetime
import threading
import multiprocessing as mp

import utils
from config import *
from model import *
from MCTS import *

"""
This file coordinates training procedure, including:

1. invoke self play
2. store result from self play & start training NN immediately based on that single example
3. evaluate how well the training is, via loss, (# games drawn due to 50 moves no progress?, ) etc.
4. save model checkpoints for each 'x' self play
5. allow loading saved model checkpoints given argument
"""

# Count the number of training iterations done; used for naming models
ITERATION_COUNT = 0

def generate_self_play(worker_id, model_path, num_self_play):
    # Load the current model in the worker only for prediction and set GPU limit
    import tensorflow as tf
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    from keras.backend.tensorflow_backend import set_session
    set_session(session=session)

    # Decide what model to use
    model = ResidualCNN()
    if model_path is not None:
        print('Worker {}: loading model "{}"'.format(worker_id, model_path))
        model.load(model_path)
        print('Worker {}: model load successful'.format(worker_id))
    else:
        print('Worker {}: using un-trained model'.format(worker_id))

    # Worker start generating self plays according to their workload
    worker_result = []
    for i in range(num_self_play):
        board = Board()
        node = Node(board, PLAYER_ONE)
        tree = MCTS(node, model)
        play_history, outcome = tree.selfPlay()
        worker_result.append((play_history, outcome))
        print('Worker {}: generated {} self-plays'.format(worker_id, len(worker_result)))

    return worker_result



def generate_self_play_in_parallel(model_path, num_self_play, num_workers):
    # Process pool for parallelism
    process_pool = mp.Pool(processes=num_workers)
    work_share = num_self_play // num_workers
    worker_results = []

    # Send processes to generate self plays
    for i in range(num_workers):
        if i == num_workers - 1:
            work_share += (num_self_play % num_workers)

        # Send workers
        result_async = process_pool.apply_async(generate_self_play, args=(i + 1, model_path, work_share))
        worker_results.append(result_async)

    # Join processes and summarise the generated final list of games
    game_list = []
    for result in worker_results:
        game_list += result.get()

    process_pool.close()
    process_pool.join()

    return game_list



def train(model_path, board_x, pi_y, v_y, data_retention, version):
    # Set TF gpu limit
    import tensorflow as tf
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    from keras.backend.tensorflow_backend import set_session
    set_session(session=session)

    message = 'At {}, Training Version {}, Number of examples: {} (retaining {:.1f}%)' \
        .format(utils.cur_time(), version, len(board_x), data_retention * 100)

    utils.stress_message(message, True)


    # Make sure path is not null if we are not training from scratch
    if version > 0:
        assert model_path is not None

    cur_model = ResidualCNN()
    if model_path is not None:
        cur_model.load(model_path)

    # Train!
    for _ in range(EPOCHS):
        # Sample a *different* portion of training data in each epoch
        sampled_idx = np.random.choice(len(board_x), int(data_retention * len(board_x)), replace=False)
        sampled_board_x = board_x[sampled_idx]
        sampled_pi_y = pi_y[sampled_idx]
        sampled_v_y = v_y[sampled_idx]

        cur_model.model.fit(sampled_board_x, [sampled_pi_y, sampled_v_y],
            batch_size=BATCH_SIZE,
            epochs=1,
            shuffle=True)

    cur_model.save(SAVE_MODELS_DIR, version)



def evolve(cur_model_path):
    while True:
        global ITERATION_COUNT

        # print some useful message
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = 'At {}, Starting to generate self-plays for Version {}'.format(utils.cur_time(), ITERATION_COUNT)
        utils.stress_message(message, True)

        games = generate_self_play_in_parallel(cur_model_path, NUM_SELF_PLAY, NUM_WORKERS)

        # Convert self-play games to training data
        board_x, pi_y, v_y = convert_to_train_data(games)
        board_x, pi_y, v_y = augment_train_data(board_x, pi_y, v_y)
        
        # Numpyify and save for later iterations
        board_x, pi_y, v_y = np.array(board_x), np.array(pi_y), np.array(v_y)
        save_train_data(board_x, pi_y, v_y, version=ITERATION_COUNT)

        # Get prev iters training data
        board_x, pi_y, v_y, data_iters_used = combine_prev_iters_train_data(board_x, pi_y, v_y)
        assert len(board_x) == len(pi_y) == len(v_y)

        # Calculate training set retention rate including current iteration; use default if too high
        data_retention_rate = min(1. / data_iters_used, DEF_DATA_RETENTION_RATE)

        # Use a *new process* to train since we DONT want to load TF in the parent process
        training_process = mp.Process(
            target=train,
            args=(cur_model_path, board_x, pi_y, v_y, data_retention_rate, ITERATION_COUNT))
        training_process.start()
        training_process.join()

        # Update path variable since we made a new version
        cur_model_path = get_model_path_from_version(ITERATION_COUNT)

        # Update version number
        ITERATION_COUNT += 1



def combine_prev_iters_train_data(board_x, pi_y, v_y):
    # Read data from previous iterations
    all_board_x, all_pi_y, all_v_y = [board_x], [pi_y], [v_y]
    for i in range(ITERATION_COUNT - PAST_ITER_COUNT, ITERATION_COUNT):
        if i < 0:
            continue
        with h5py.File('{}/{}{}.h5'.format(SAVE_TRAIN_DATA_DIR, SAVE_TRAIN_DATA_PREF, i), 'r') as H:
            all_board_x.append(np.copy(H['board_x']))
            all_pi_y.append(np.copy(H['pi_y']))
            all_v_y.append(np.copy(H['v_y']))

    # Make a pool of training data from previous iterations
    board_x = np.vstack(all_board_x)
    pi_y = np.vstack(all_pi_y)
    v_y = np.hstack(all_v_y)                        # hstack as v_y is 1D array

    return board_x, pi_y, v_y, len(all_board_x)     # Last item is total iterations used



def save_train_data(board_x, pi_y, v_y, version):
    ''' Write current iteration training data to disk '''
    if not os.path.exists(SAVE_TRAIN_DATA_DIR):
        os.makedirs(SAVE_TRAIN_DATA_DIR)

    with h5py.File('{}/{}{}.h5'.format(SAVE_TRAIN_DATA_DIR, SAVE_TRAIN_DATA_PREF, version), 'w') as H:
        H.create_dataset('board_x', data=board_x)
        H.create_dataset('pi_y', data=pi_y)
        H.create_dataset('v_y', data=v_y)



def convert_to_train_data(self_play_games):
    ''' return python lists containing training data '''
    board_x, pi_y, v_y = [], [], []
    for game in self_play_games:
        history, reward = game
        curr_player = PLAYER_ONE
        for board, pi in history:
            board_x.append(Model.to_model_input(board, curr_player))
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



def get_model_path_from_version(version):
    return '{}/{}{:0>4}.h5'.format(SAVE_MODELS_DIR, MODEL_PREFIX, version)



if __name__ == '__main__':
    model_path = None
    if len(sys.argv) != 1:
        model_path = sys.argv[1]
        try:
            # Read the count from file name
            ITERATION_COUNT = int(re.search('{}(.+?)\.h5'.format(MODEL_PREFIX), model_path).group(1)) + 1
        except:
            ITERATION_COUNT = 0

    utils.stress_message('Start to training from version: {}'.format(ITERATION_COUNT), True)
    evolve(model_path)

