import sys
from constants import *
from model_configs import *
import threading
from model import *
from MCTS import *
import tensorflow as tf
import keras
from keras import backend as K
import re
import datetime

"""
Coordinates training procedure, including:

1. invoke self play
2. store result from self play & start training NN immediately based on that single example
3. evaluate how well the training is, via loss, (# games drawn due to 50 moves no progress?, ) etc.
4. save model checkpoints for each 'x' self play
5. allow loading saved model checkpoints given argument
"""

# Count the number of training iterations done; used for naming models
ITERATION_COUNT = 0

class SelfPlayThread(threading.Thread):
    def __init__(self, id, game_list, model, num_self_play):
        threading.Thread.__init__(self)
        self.id = id
        self.game_list = game_list
        self.model = model
        self.num_self_play = num_self_play

    def run(self):
        thread_result = []
        for i in range(self.num_self_play):
            board = Board()
            node = Node(board, PLAYER_ONE)
            tree = MCTS(node, self.model)
            play_history, result = tree.selfPlay()
            thread_result.append((play_history, result))
            print('Thread {} generated {} self-play games'.format(self.id, len(thread_result)))

        self.game_list += thread_result


def generate_self_play_in_parallel(model, num_self_play=NUM_SELF_PLAY, num_threads=NUM_THREADS):
    game_list = []
    thread_list = []
    for i in range(num_threads):
        thread = SelfPlayThread(i + 1, game_list, model, num_self_play // num_threads)
        thread_list.append(thread)
        thread.start()
    for thread in thread_list:
        thread.join()
    return game_list


def evolve(curr_model, num_self_play = NUM_SELF_PLAY):
    while True:
        global ITERATION_COUNT
        print('\n{}\nAt {} Version {}, starting to generate self-plays\n{}\n'.format(
            '*'*80, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ITERATION_COUNT, '*'*80))
        # Make the model ready for prediction before concurrent access of `predict()`
        curr_model.model._make_predict_function()
        training_data = generate_self_play_in_parallel(curr_model, num_self_play, NUM_THREADS)
        board_x, pi_y, v_y = preprocess_training_data(training_data)
        print('\n{}\nAt {} Version {}, Number of training examples: {}\n{}\n'.format(
            '*'*80, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ITERATION_COUNT, len(board_x), '*'*80))
        curr_model.model.fit(board_x, [pi_y, v_y], validation_split=0.05, batch_size=BATCH_SIZE, epochs=EPOCHS)
        curr_model.save(ITERATION_COUNT)
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



if __name__ == '__main__':
    model = ResidualCNN()
    if len(sys.argv) != 1:
        filename = sys.argv[1]
        model.load(filename)
        try:
            ITERATION_COUNT = int(re.search('version(.+?)\.h5', filename).group(1)) + 1
        except:
            ITERATION_COUNT = 0

    print('\n{}\nStart to training from version: {}\n{}\n'.format('*'*50, ITERATION_COUNT, '*'*50))
    evolve(model)
