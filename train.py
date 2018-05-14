import re
import imp
import sys
import datetime
import threading
import multiprocessing as mp

from constants import *
from model_configs import *
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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

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



def train(model_path, board_x, pi_y, v_y, iter_count):
    # Set TF gpu limit
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    from keras.backend.tensorflow_backend import set_session
    set_session(session=session)

    print('\n{0}\nAt {1}, Training Version {2}, Number of training examples: {3}\n{0}\n'.format(
        '*'*80, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iter_count, len(board_x)))

    # Make sure path is not null if we are not training from scratch
    if iter_count > 0:
        assert model_path is not None

    cur_model = ResidualCNN()
    if model_path is not None:
        cur_model.load(model_path)

    cur_model.model.fit(board_x, [pi_y, v_y], batch_size=BATCH_SIZE, epochs=EPOCHS)
    cur_model.save(SAVE_MODELS_DIR, iter_count)



def evolve(cur_model_path):
    while True:
        global ITERATION_COUNT
        print('\n{0}\nAt {1}, Starting to generate self-plays for Version {2}\n{0}\n'.format(
            '*'*80, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ITERATION_COUNT))

        training_data = generate_self_play_in_parallel(cur_model_path, NUM_SELF_PLAY, NUM_WORKERS)
        board_x, pi_y, v_y = preprocess_training_data(training_data)

        # Use a *new process* to train since we DONT want to load TF in the parent process
        training_process = mp.Process(target=train, args=(cur_model_path, board_x, pi_y, v_y, ITERATION_COUNT))
        training_process.start()
        training_process.join()

        # Update path variable since we made a new version
        cur_model_path = get_path_from_version(SAVE_MODELS_DIR, ITERATION_COUNT)

        # Update version number
        ITERATION_COUNT += 1



def preprocess_training_data(self_play_games):
    board_x = []
    pi_y = []
    v_y = []
    for game in self_play_games:
        history, reward = game
        curr_player = PLAYER_ONE
        for board, pi in history:
            board_x.append(Model.to_model_input(board, curr_player))
            pi_y.append(pi)
            v_y.append(reward)
            reward = -reward
            curr_player = PLAYER_ONE + PLAYER_TWO - curr_player

    return np.array(board_x), np.array(pi_y), np.array(v_y)



def get_path_from_version(path_pref, version):
    return path_pref + '/version{:0>4}.h5'.format(version)



if __name__ == '__main__':
    model_path = None
    if len(sys.argv) != 1:
        model_path = sys.argv[1]
        try:
            ITERATION_COUNT = int(re.search('version(.+?)\.h5', model_path).group(1)) + 1
        except:
            ITERATION_COUNT = 0

    print('\n{0}\nStart to training from version: {1}\n{0}\n'.format('*'*50, ITERATION_COUNT))
    evolve(model_path)

