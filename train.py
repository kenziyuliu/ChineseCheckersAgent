import sys
from constants import *
from model_configs import *
import threading
from model import *
from MCTS import *
import tensorflow as tf
import keras
from keras import backend as K

"""
Coordinates training procedure, including:

1. invoke self play
2. store result from self play & start training NN immediately based on that single example
3. evaluate how well the training is, via loss, (# games drawn due to 50 moves no progress?, ) etc.
4. save model checkpoints for each 'x' self play
5. allow loading saved model checkpoints given argument
"""

class SelfPlayThread (threading.Thread):

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


def generate_self_play_in_parallel(model, num_self_play=NUM_SELF_PLAY, num_CPU=NUM_THREADS):
    game_list = []
    thread_list = []
    for i in range(num_CPU):
        thread = SelfPlayThread(i + 1, game_list, model, num_self_play // num_CPU)
        thread_list.append(thread)
        thread.start()
    for thread in thread_list:
        thread.join()
    return game_list


def evolve(curr_model, num_self_play = NUM_SELF_PLAY):
    count = 1
    while True:
        # Make the model ready for prediction before concurrent access of `predict()`
        curr_model.model._make_predict_function()
        training_data = generate_self_play_in_parallel(curr_model, num_self_play, NUM_THREADS)
        print('Number of training examples: {}'.format(len(training_data)))
        board_x, pi_y, v_y = preprocess_training_data(training_data)
        curr_model.model.fit(board_x, [pi_y, v_y], batch_size=BATCH_SIZE, epochs=EPOCHS)
        if count % 10 == 0:
            curr_model.save(count / 10)
        count += 1


def preprocess_training_data(self_play_games):
    board_x = []
    pi_y = []
    v_y = []
    for game in self_play_games:
        history, winner = game
        curr_player = PLAYER_ONE
        for board, pi in history:
            board_x.append(Model.to_model_input(board, curr_player))
            pi_y.append(pi)
            if winner == 0:
                v_y.append(0)
            elif winner == curr_player:
                v_y.append(1)
            else:
                v_y.append(-1)

            curr_player = PLAYER_ONE + PLAYER_TWO - curr_player

    return np.array(board_x), np.array(pi_y), np.array(v_y)



if __name__ == '__main__':
    model = ResidualCNN()
    if len(sys.argv) != 1:
        model.load(sys.argv[1])
    evolve(model)



