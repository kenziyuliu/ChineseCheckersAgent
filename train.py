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
    
    def __init__(self, game_list, lock, model, num_self_play):
        threading.Thread.__init__(self)
        self.game_list = game_list
        self.lock = lock
        self.model = model
        self.num_self_play = num_self_play
    
    def run(self):
        for i in range(self.num_self_play):
            board = Board()
            node = Node(board, PLAYER_ONE)
            tree = MCTS(node, self.model)
            play_history, result = tree.selfPlay()
            self.lock.acquire()
            self.game_list.append((play_history, result))
            self.lock.release()


def generate_self_play_in_parallel(model, num_self_play = NUM_SELF_PLAY, num_CPU = NUM_THREADS):
    game_list = []
    thread_list = []
    threadLock = threading.Lock()
    for i in range(num_CPU):
        thread = SelfPlayThread(game_list, threadLock, model, num_self_play // num_CPU)
        thread_list.append(thread)
        thread.start()
    for thread in thread_list:
        thread.join()
    return game_list


def evolve(curr_model, num_self_play = NUM_SELF_PLAY):
    count = 1
    while True:

        model_copy = ResidualCNN()
        model_copy.model = keras.models.clone_model(curr_model.model)
        model_copy.model.set_weights(curr_model.model.get_weights())
        model_copy.model._make_predict_function()

        training_data = generate_self_play_in_parallel(model_copy, num_self_play, NUM_THREADS)
        print(len(training_data))
        board_x, pi_y, v_y = preprocess_training_data(training_data)
        curr_model.model.fit(board_x, [pi_y, v_y], batch_size=BATCH_SIZE, epochs = EPOCHS)
        if count % 10 == 0:
            curr_model.save(cont/10)
        count += 1

def preprocess_training_data(raw_data):
    board_x = []
    pi_y = []
    v_y = []
    for game in raw_data:
        curr_player = 1
        for board, pi in game[0]:
            board_x.append(Model.to_model_input(board, curr_player))
            pi_y.append(pi)
            if game[1] == 0:
                v_y.append(0)
            elif game[1] == curr_player:
                v_ya.append(1)
            else:
                v_ya.append(-1)
    board_x = np.array(board_x)
    print(board_x.shape)
    pi_y = np.array(pi_y)
    v_y = np.array(v_y)
    return board_x, pi_y, v_y


if __name__ == '__main__':
    model = ResidualCNN()
    if len(sys.argv) != 1:
        model.load(sys.argv[1])
    evolve(model)


    
