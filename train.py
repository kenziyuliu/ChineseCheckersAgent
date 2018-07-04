import re
import os
import sys
import h5py
import datetime
import threading
import argparse
import multiprocessing as mp

import utils
from config import *
from model import *
from MCTS import *
from selfplay import selfplay

"""
This file coordinates training procedure, including:

1. invoke self play
2. store result from self play & start training NN immediately based on that single example
3. evaluate how well the training is, via loss, (# games drawn due to 50 moves no progress?, ) etc.
4. save model checkpoints for each 'x' self play
5. allow loading saved model checkpoints given argument
"""


def generate_self_play(worker_id, model_path, num_self_play, model2_path=None):
    # Load the current model in the worker only for prediction and set GPU limit
    import tensorflow as tf
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    from keras.backend.tensorflow_backend import set_session
    set_session(session=session)

    # Re-seed the generators: since the RNG was copied from parent process
    np.random.seed()        # None seed to source from /dev/urandom
    random.seed()

    # Decide what model to use
    model = ResidualCNN()
    model2 = None
    if model_path is not None:
        print('Worker {}: loading model "{}"'.format(worker_id, model_path))
        model.load_weights(model_path)
        print('Worker {}: model load successful'.format(worker_id))

        if model2_path is not None:
            print('Worker {}: loading 2nd model "{}"'.format(worker_id, model2_path))
            model2 = ResidualCNN()
            model2.load_weights(model2_path)
            print('Worker {}: 2nd model load successful'.format(worker_id))
        else:
            print ('Worker {}: Model2 is None; using Model1 to generate selfplays'.format(worker_id))
    else:
        print('Worker {}: using un-trained model'.format(worker_id))

    # Worker start generating self plays according to their workload
    worker_result = []
    for i in range(num_self_play):
        play_history, p1_reward = selfplay(model, model2, randomised=False)
        if play_history is not None and p1_reward is not None:
            worker_result.append((play_history, p1_reward))
        print('Worker {}: generated {} self-plays'.format(worker_id, len(worker_result)))

    return worker_result



def generate_self_play_in_parallel(model_path, num_self_play, num_workers, model2_path=None):
    # Process pool for parallelism
    process_pool = mp.Pool(processes=num_workers)
    work_share = num_self_play // num_workers
    worker_results = []

    # Send processes to generate self plays
    for i in range(num_workers):
        if i == num_workers - 1:
            work_share += (num_self_play % num_workers)

        # Send workers
        result_async = process_pool.apply_async(
            generate_self_play,
            args=(i + 1, model_path, work_share, model2_path))
        worker_results.append(result_async)

    try:
        # Join processes and summarise the generated final list of games
        game_list = []
        for result in worker_results:
            game_list += result.get()

        process_pool.close()

    # Exit early if need
    except KeyboardInterrupt:
        utils.stress_message('SIGINT caught, exiting')
        process_pool.terminate()
        process_pool.join()
        exit()

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

    np.random.seed()
    random.seed()

    message = 'At {}, Training Version {}, Number of examples: {} (retaining {:.1f}%)' \
        .format(utils.cur_time(), version, len(board_x), data_retention * 100)

    utils.stress_message(message, True)

    # Make sure path is not null if we are not training from scratch
    cur_model = ResidualCNN()
    if version > 0:
        assert model_path is not None
        cur_model.load_weights(model_path)

    # Sample a portion of training data before training
    sampled_idx = np.random.choice(len(board_x), int(data_retention * len(board_x)), replace=False)
    sampled_board_x = board_x[sampled_idx]
    sampled_pi_y = pi_y[sampled_idx]
    sampled_v_y = v_y[sampled_idx]

    cur_model.model.fit(sampled_board_x, [sampled_pi_y, sampled_v_y],
        batch_size=BATCH_SIZE,
        validation_split=0.05,
        epochs=EPOCHS,
        shuffle=True)

    cur_model.save_weights(SAVE_WEIGHTS_DIR, MODEL_PREFIX, version)
    # cur_model.save(SAVE_MODELS_DIR, MODEL_PREFIX, version)



def evaluate(worker_id, best_model, cur_model, num_games):
    # Load the current model in the worker only for prediction and set GPU limit
    import tensorflow as tf
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    from keras.backend.tensorflow_backend import set_session
    set_session(session=session)

    # Re-seed the generators: since the RNG was copied from parent process
    np.random.seed()        # None seed to source from /dev/urandom
    random.seed()

    from ai_vs_ai import agent_match

    cur_model_wincount = 0
    best_model_wincount = 0
    draw_count = 0
    for i in range(num_games):
        # Alternate players
        if i % 2 == 0:
            winner = agent_match(best_model, cur_model, num_games=1, enforce_move_limit=True)
        else:
            winner = agent_match(cur_model, best_model, num_games=1, enforce_move_limit=True)

        if winner == cur_model:
            cur_model_wincount += 1
        elif winner == best_model:
            best_model_wincount += 1
        else:
            draw_count += 1

    print('Worker {}: cur_model "{}" wins {}/{} games'.format(worker_id, cur_model, cur_model_wincount, num_games))
    print('Worker {}: best_model "{}" wins {}/{} games'.format(worker_id, best_model, best_model_wincount, num_games))
    print('Worker {}: {}/{} games were draw'.format(worker_id, draw_count, num_games))
    return cur_model_wincount



def evaluate_in_parallel(best_model, cur_model, num_games, num_workers):
    utils.stress_message('Evaluating model "{}" against current best model "{}" on {} games'
        .format(cur_model, best_model, num_games), True)

    # Process pool for parallelism
    process_pool = mp.Pool(processes=num_workers)
    work_share = num_games // num_workers
    worker_results = []

    # Send processes to generate self plays
    for i in range(num_workers):
        if i == num_workers - 1:
            work_share += (num_games % num_workers)

        # Send workers
        result_async = process_pool.apply_async(
            evaluate,
            args=(i + 1, best_model, cur_model, work_share))
        worker_results.append(result_async)

    try:
        # Join processes and count games
        cur_model_wincount = 0
        for result in worker_results:
            cur_model_wincount += result.get()

        process_pool.close()

    # Exit early if need
    except KeyboardInterrupt:
        utils.stress_message('SIGINT caught, exiting')
        process_pool.terminate()
        process_pool.join()
        exit()


    process_pool.join()

    utils.stress_message('Overall, cur_model "{}" wins {}/{} against best_model "{}"'
        .format(cur_model, cur_model_wincount, num_games, best_model), True)

    return cur_model_wincount



def evolve(cur_model_path, other_opponent_for_selfplay, iteration_count, best_model):

    while True:
        # print some useful message
        message = 'At {}, Starting to generate self-plays for Version {}'.format(utils.cur_time(), iteration_count)
        utils.stress_message(message, True)

        ##########################
        ##### GENERATE PLAYS #####
        ##########################

        if other_opponent_for_selfplay is not None:
            print('(Generating games using current model {} and other model {})'.format(cur_model_path, other_opponent_for_selfplay))
            games = generate_self_play_in_parallel(cur_model_path, NUM_SELF_PLAY, NUM_WORKERS, model2_path=other_opponent_for_selfplay)
        elif best_model is not None:
            print('(Generating games using given best model: {})'.format(best_model))
            games = generate_self_play_in_parallel(best_model, NUM_SELF_PLAY, NUM_WORKERS)
        else:
            # # Use previous version to generate selfplay if necessary
            # model2_path = None
            # if SELF_PLAY_DIFF_MODEL and ITERATION_COUNT > 1:
            #     model2_version = get_rand_prev_version(ITERATION_COUNT)
            #     model2_path = get_weights_path_from_version(model2_version)
            #     utils.stress_message('.. and vs. Version {}'.format(model2_version))
            #
            # games = generate_self_play_in_parallel(cur_model_path, NUM_SELF_PLAY, NUM_WORKERS, model2_path)
            games = generate_self_play_in_parallel(cur_model_path, NUM_SELF_PLAY, NUM_WORKERS)


        ##########################
        ##### PREPARING DATA #####
        ##########################

        # Convert self-play games to training data
        board_x, pi_y, v_y = utils.convert_to_train_data(games)
        board_x, pi_y, v_y = utils.augment_train_data(board_x, pi_y, v_y)

        # Numpyify and save for later iterations
        board_x, pi_y, v_y = np.array(board_x), np.array(pi_y), np.array(v_y)
        if len(board_x) > 0 and len(pi_y) > 0 and len(v_y) > 0:
            utils.save_train_data(board_x, pi_y, v_y, version=iteration_count)

        # Get prev iters training data
        board_x, pi_y, v_y, data_iters_used = combine_prev_iters_train_data(board_x, pi_y, v_y, iteration_count)
        assert len(board_x) == len(pi_y) == len(v_y)

        # Train only if there were data
        if data_iters_used == 0:
            utils.stress_message('No training data for iteration {}! Re-iterating...'.format(iteration_count))
            continue

        # Calculate training set retention rate including current iteration; use default if too high
        data_retention_rate = min(1. / data_iters_used, DEF_DATA_RETENTION_RATE)

        #################
        ##### TRAIN #####
        #################

        # Use a *new process* to train since we DONT want to load TF in the parent process
        training_process = mp.Process(
            target=train,
            args=(cur_model_path, board_x, pi_y, v_y, data_retention_rate, iteration_count))
        training_process.start()
        training_process.join()

        # Update path variable since we made a new version
        # cur_model_path = get_model_path_from_version(ITERATION_COUNT)
        cur_model_path = get_weights_path_from_version(iteration_count)

        ####################
        ##### EVALUATE #####
        ####################

        if best_model is not None:
            cur_model_wincount = evaluate_in_parallel(best_model, cur_model_path, EVAL_GAMES, NUM_WORKERS)
            if cur_model_wincount > int(0.55 * EVAL_GAMES):
                best_model = cur_model_path
                utils.stress_message('Now using {} as the best model'.format(best_model))
            else:
                utils.stress_message('Output model of this iteration is not better; retaining {} as the best model'.format(best_model), True)

        # Update version number
        iteration_count += 1



def combine_prev_iters_train_data(board_x, pi_y, v_y, iteration_count):
    all_board_x, all_pi_y, all_v_y = [], [], []

    if len(board_x) > 0 and len(pi_y) > 0 and len(v_y) > 0:
        all_board_x.append(board_x)
        all_pi_y.append(pi_y)
        all_v_y.append(v_y)

    # Read data from previous iterations
    for i in range(iteration_count - PAST_ITER_COUNT, iteration_count):
        if i >= 0:
            filename = '{}/{}{}.h5'.format(SAVE_TRAIN_DATA_DIR, SAVE_TRAIN_DATA_PREF, i)

            if not os.path.exists(filename):
                utils.stress_message('{} does not exist!'.format(filename))
                continue

            with h5py.File(filename, 'r') as H:
                all_board_x.append(np.copy(H['board_x']))
                all_pi_y.append(np.copy(H['pi_y']))
                all_v_y.append(np.copy(H['v_y']))

    if len(all_board_x) > 0 and len(all_pi_y) > 0 and len(all_v_y) > 0:
        # Make a pool of training data from previous iterations
        board_x = np.vstack(all_board_x)
        pi_y = np.vstack(all_pi_y)
        v_y = np.hstack(all_v_y)                        # hstack as v_y is 1D array

        return board_x, pi_y, v_y, len(all_board_x)     # Last retval is total iterations used

    # If no data at all: return empty training data
    return [], [], [], 0



def get_model_path_from_version(version):
    return '{}/{}{:0>4}.h5'.format(SAVE_MODELS_DIR, MODEL_PREFIX, version)


def get_weights_path_from_version(version):
    return '{}/{}{:0>4}-weights.h5'.format(SAVE_WEIGHTS_DIR, MODEL_PREFIX, version)


def get_rand_prev_version(upper):
    return np.random.randint(upper // 2, upper)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cur_model_path', '-c',
                    dest='model_path',
                    help='current model_path of model to optimize on (newest model)')
    parser.add_argument('--best_model_path', '-b',
                    dest='best_model_path',
                    help='best model')
    parser.add_argument('--other_opponent_for_selfplay', '-p',
                    dest='other_opponent_for_selfplay',
                    help='boolean to specify selfplay generated by both current model and best model or not, \
                    if not generate with only best model')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    model_path = args.model_path
    best_model = args.best_model_path
    other_opponent_for_selfplay = args.other_opponent_for_selfplay

    try:
        # Read the count from file name
        iteration_count = utils.find_version_given_filename(model_path) + 1
    except:
        iteration_count = 0

    if best_model is not None:
        print('\nBest model {} specified!\n'.format(best_model))
    if other_opponent_for_selfplay is not None:
        print('\nOpponent {} for selfplay specified\n'.format(other_opponent_for_selfplay))

    utils.stress_message('Start to training from version: {}'.format(iteration_count), True)
    evolve(model_path, other_opponent_for_selfplay, iteration_count, best_model)
