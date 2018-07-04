import gc
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
from data_generators import GreedyDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


def generate_self_play(worker_id, num_self_play):
    # Re-seed the generators: since the RNG was copied from parent process
    np.random.seed()        # None seed to source from /dev/urandom

    # Worker start generating self plays according to their workload
    worker_result = []
    num_normal_games = int(G_NORMAL_GAME_RATIO * num_self_play)
    num_rand_start_games = int(G_RAND_START_GAME_RATIO * num_self_play)
    num_randomised_games = num_self_play - num_normal_games - num_rand_start_games

    normal_gen = GreedyDataGenerator()
    random_start_gen = GreedyDataGenerator(random_start=True)
    randomised_gen = GreedyDataGenerator(randomised=True)

    def generate(num_games, gen):
        for i in range(num_games):
            worker_result.append(gen.generate_play())
            if len(worker_result) % 100 == 0:
                print('Worker {}: generated {} self-plays'.format(worker_id, len(worker_result)))

    generate(num_normal_games, normal_gen)
    generate(num_rand_start_games, random_start_gen)
    generate(num_randomised_games, randomised_gen)

    print('Worker {}: generated {} self-plays'.format(worker_id, len(worker_result)))
    return worker_result



def generate_self_play_in_parallel(num_self_play, num_workers):
    # Process pool for parallelism
    process_pool = mp.Pool(processes=num_workers)
    work_share = num_self_play // num_workers
    worker_results = []

    # Send processes to generate self plays
    for i in range(num_workers):
        if i == num_workers - 1:
            work_share += (num_self_play % num_workers)

        # Send workers
        result_async = process_pool.apply_async(generate_self_play, args=(i + 1, work_share))
        worker_results.append(result_async)

    # Join processes and summarise the generated final list of games
    game_list = []
    for result in worker_results:
        game_list += result.get()

    process_pool.close()
    process_pool.join()

    return game_list



def train(num_games, model, version):
    # print some useful message
    message = 'At {}, Starting to generate {} greedy self-play games for version {}'.format(utils.cur_time(), num_games, version)
    utils.stress_message(message, True)

    # Generate games
    games = generate_self_play_in_parallel(num_games, NUM_WORKERS)

    utils.stress_message('Preparing training examples from {} games'.format(len(games)))

    # Convert self-play games to training data
    board_x, pi_y, v_y = utils.convert_to_train_data(games)
    board_x, pi_y, v_y = utils.augment_train_data(board_x, pi_y, v_y)
    assert len(board_x) == len(pi_y) == len(v_y)

    print('\nNumber of training examples (Total): {}'.format(len(board_x)))

    # Sample a portion of training data
    num_train_data = int(G_DATA_RETENTION_RATE * len(board_x))

    sampled_idx = np.random.choice(len(board_x), num_train_data, replace=False)
    board_x_train = np.array([board_x[sampled_idx[i]] for i in range(num_train_data)])
    pi_y_train = np.array([pi_y[sampled_idx[i]] for i in range(num_train_data)])
    v_y_train = np.array([v_y[sampled_idx[i]] for i in range(num_train_data)])

    # board_x_val = np.array([board_x[sampled_idx[i]] for i in range(num_train_data, num_train_data + G_NUM_VAL_DATA)])
    # pi_y_val = np.array([pi_y[sampled_idx[i]] for i in range(num_train_data, num_train_data + G_NUM_VAL_DATA)])
    # v_y_val = np.array([v_y[sampled_idx[i]] for i in range(num_train_data, num_train_data + G_NUM_VAL_DATA)])

    assert len(board_x_train) == len(pi_y_train) == len(v_y_train)
    print('Number of training examples (Sampled): {}\n'.format(len(board_x_train)))

    # Make sure that the directory is available
    if not os.path.exists(SAVE_WEIGHTS_DIR):
        os.makedirs(SAVE_WEIGHTS_DIR)

    model.model.fit(board_x_train, [pi_y_train, v_y_train],
        # validation_data=((board_x_val, [pi_y_val, v_y_val]) if G_NUM_VAL_DATA > 0 else None),
        validation_split=G_VAL_SPLIT,
        batch_size=G_BATCH_SIZE,
        epochs=G_ITER_PER_EPOCH,
        shuffle=True)
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5),
        #            ModelCheckpoint(filepath=SAVE_WEIGHTS_DIR+'GreedyWeights-ep{epoch:02d}-val{val_loss:.2f}.h5',
        #                            save_best_only=True, save_weights_only=True)])

    model.save_weights(SAVE_WEIGHTS_DIR, G_MODEL_PREFIX, version=version)
    utils.stress_message('GreedyModel Weights version {} saved to {}'.format(version, SAVE_WEIGHTS_DIR), True)



if __name__ == '__main__':
    print('Initialising Model...')
    model = ResidualCNN()
    print('Model Initialised')

    version = 0
    if len(sys.argv) == 2:
        checkpoint = sys.argv[1]
        print('Continue training from version "{}"'.format(checkpoint))
        model.load_weights(checkpoint)
        version = utils.find_version_given_filename(checkpoint) + 1
        print('\nCurrent training version {}'.format(version))

    for i in range(G_EPOCHS):
        train(G_GAMES_PER_EPOCH, model, i + version)
