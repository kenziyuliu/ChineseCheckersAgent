from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.models import Model as KerasModel
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU, add
from board import *
from constants import *
from loss import softmax_cross_entropy_with_logits
import model_configs
import os

class Model:
    def __init__(self, input_dim, filters):
        self.input_dim = input_dim
        self.filters = filters

    def predict(self, input_board):
        return self.model.predict(np.expand_dims(input_board, axis=0))

    def save(self, version):
        if not os.path.exists(SAVE_MODELS_DIR):
            os.makedirs(SAVE_MODELS_DIR)
        self.model.save('{0}version{1:0>4}'.format(SAVE_MODELS_DIR, version) + '.h5')

    def load(self, filepath):
        self.model = load_model(filepath)
        return self.model

    def visualise_layers(self):
        """
        Referenced from
        https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning
        Not sure if it works
        """
        layers = self.model.layers
        for i, l in enumerate(layers):
            x = l.get_weights()
            print('LAYER ' + str(i))

            try:
                weights = x[0]
                s = weights.shape

                fig = plt.figure(figsize=(s[2], s[3]))  # width, height in inches
                channel = 0
                filter = 0
                for i in range(s[2] * s[3]):
                    sub = fig.add_subplot(s[3], s[2], i + 1)
                    sub.imshow(weights[:,:,channel,filter], cmap='coolwarm', clim=(-1, 1),aspect="auto")
                    channel = (channel + 1) % s[2]
                    filter = (filter + 1) % s[3]

            except:
                try:
                    fig = plt.figure(figsize=(3, len(x)))  # width, height in inches
                    for i in range(len(x)):
                        sub = fig.add_subplot(len(x), 1, i + 1)
                        if i == 0:
                            clim = (0,2)
                        else:
                            clim = (0, 2)
                        sub.imshow([x[i]], cmap='coolwarm', clim=clim,aspect="auto")

                    plt.show()
                except:
                    try:
                        fig = plt.figure(figsize=(3, 3))  # width, height in inches
                        sub = fig.add_subplot(1, 1, 1)
                        sub.imshow(x[0], cmap='coolwarm', clim=(-1, 1),aspect="auto")

                        plt.show()
                    except:
                        pass

            plt.show()


    @staticmethod
    def to_model_input(board, cur_player):
        """
        Input:
            board: 7 x 7 x 3 board._board. Each channel contains positions of both players' checkers.
            cur_player: player number of the current player
        Output:
            7 x 7 x 7. First 3 channel is player 1, next 3 channel is player 2, last channel is all 0 if player 1 is to play.
        """
        # initialise the model input
        model_input = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, NUM_HIST_MOVES * 2 + 1), dtype="uint8") # may change dtype afterwards
        # get np array board
        new_board = board.board
        # get history moves
        moves = board.hist_moves
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
        hist_index = len(moves) - 1
        for channel in range(1, NUM_HIST_MOVES):
            if not np.any(new_board[:, :, channel]): # timestep < 0
                break
            move = moves[hist_index]
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

        if cur_player == 2: # player 2 to play
            model_input[:, :, NUM_HIST_MOVES * 2] = np.ones((BOARD_WIDTH, BOARD_HEIGHT), dtype="uint8")

        return model_input


    @staticmethod
    def encode_checker_index(checker_id, coord):
        """
        Convert a checker and its destination
        to the model's output encoding.
        """
        region = checker_id * BOARD_WIDTH * BOARD_HEIGHT # get the element-block in the model's output
        offset = coord[0] * BOARD_WIDTH + coord[1]          # offset in this region
        return region + offset


    @staticmethod
    def decode_checker_index(model_output_index):
        """
        Convert the index in the model's output vector
        to the checker number and its destination on board
        """
        checker_id = model_output_index // (BOARD_WIDTH * BOARD_HEIGHT)
        offset = model_output_index % (BOARD_WIDTH * BOARD_HEIGHT)
        dest = offset // BOARD_WIDTH, offset % BOARD_WIDTH
        return checker_id, dest


class ResidualCNN(Model):
    def __init__(self, input_dim=model_configs.INPUT_DIM, filters=model_configs.NUM_FILTERS):
        Model.__init__(self, input_dim, filters)
        self.model = self.build_model()


    def build_model(self):
        main_input = Input(shape=self.input_dim)
        x = self.conv_block(main_input, self.filters, 3, model_configs.REGULARIZER)

        for _ in range(model_configs.NUM_RESIDUAL_BLOCKS):
            x = self.residual_block(x, self.filters, 3, model_configs.REGULARIZER)

        value = self.value_head(x, model_configs.REGULARIZER)
        policy = self.policy_head(x, model_configs.REGULARIZER)

        model = KerasModel(inputs=[main_input], outputs=[policy, value])
        model.compile(loss={"value_head":"mean_squared_error", "policy_head":softmax_cross_entropy_with_logits},
                        optimizer=Adam(lr=model_configs.LEARNING_RATE),
                        loss_weights={"value_head": 0.5, "policy_head": 0.5})
        return model


    def conv_layer(self, layer_input, filters, kernel_size, regularizer):
        return Conv2D(filters=filters,
                        kernel_size=kernel_size,
                        padding="same",
                        use_bias=False,
                        activation="linear",
                        kernel_regularizer=regularizer)(layer_input)


    def residual_block(self, block_input, filters, kernel_size, regularizer):
        x = self.conv_layer(block_input, filters, kernel_size, regularizer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = self.conv_layer(x, filters, kernel_size, regularizer)
        x = BatchNormalization()(x)
        x = add([block_input, x])
        x = LeakyReLU()(x)
        return x


    def conv_block(self, block_input, filters, kernel_size, regularizer):
        x = self.conv_layer(block_input, filters, kernel_size, regularizer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x


    def value_head(self, head_input, regularizer):
        x = self.conv_layer(head_input, filters=1, kernel_size=1, regularizer=regularizer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(self.filters,
                use_bias=False,
                activation="linear",
                kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(1,
                use_bias=False,
                activation="tanh",
                kernel_regularizer=regularizer,
                name="value_head")(x)
        return x


    def policy_head(self, head_input, regularizer):
        x = self.conv_layer(head_input, filters=16, kernel_size=1, regularizer=regularizer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(NUM_CHECKERS * BOARD_WIDTH * BOARD_WIDTH,
                use_bias=False,
                activation='linear',
                kernel_regularizer=regularizer,
                name="policy_head")(x)
        return x


if __name__ == '__main__':
    """
    Test cases here
    """
    checker_pos = []
    for i in range(6 * 49 + 1):
        checker_pos.append(Model.decode_checker_index(i))
        # print(Model.decode_checker_index(i))

    count = 0
    for checker_id, pos in checker_pos:
        assert count == Model.encode_checker_index(checker_id, pos)
        # print(Model.encode_checker_index(checker_id, pos))
        count += 1

    # Test `to_model_input`
    gameboard = Board()
    gameboard.place(1, (5, 0), (3, 0))
    board = gameboard.board
    for i in range(board.shape[2]):
        print(board[:, :, i])
    model_input = Model.to_model_input(board, 1)
    print('\n\n')
    for i in range(model_input.shape[2]):
        print(model_input[:, :, i])

    model = ResidualCNN()
    # test for saving
    model.save(1)
