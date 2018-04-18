from keras import regularizers
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Model as KerasModel
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU, add
from board import *
from constants import *
from loss import softmax_cross_entropy_with_logits
import model_configs


class Model:
    def __init__(self, input_dim, filters):
        self.input_dim = input_dim
        self.filters = filters

    def predict(self, input_board):
        return self.model.predict(input_board)

    def save(self, version):
        self.model.save('version{0:0>4}'.format(version) + '.h5')

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
        model_input = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, NUM_HIST_MOVES * 2 + 1), dtype="uint8") # may change dtype afterwards
        op_player = PLAYER_ONE + PLAYER_TWO - cur_player
        for channel in range(NUM_HIST_MOVES):
            if not np.any(board[:, :, channel]): # timestep < 0
                break

            op_layer = np.copy(board[:, :, channel])
            cur_layer = np.copy(board[:, :, channel])

            np.putmask(cur_layer, cur_layer != cur_player, 0)
            np.putmask(cur_layer, cur_layer == cur_player, 1)

            np.putmask(op_layer, op_layer != op_player, 0)
            np.putmask(op_layer, op_layer == op_player, 1)

            model_input[:, :, channel * 2] = cur_layer
            model_input[:, :, channel * 2 + 1] = op_layer

        if cur_player == 2: # player 2 to play
            model_input[:, :, NUM_HIST_MOVES * 2] = np.ones((BOARD_WIDTH, BOARD_HEIGHT), dtype="uint8")

        return model_input

    @staticmethod
    def encode_checker_index(checker_id, coord):
        """
        Convert a checker and its current position
        to the model's output encoding.
        """
        region = checker_id * (BOARD_WIDTH * BOARD_HEIGHT) # get the element-block in the model's output
        offset = coord[0] * BOARD_WIDTH + coord[1]          # offset in this region
        return region + offset


    @staticmethod
    def decode_checker_index(model_output_index):
        """
        Convert the index in the model's output vector
        to the checker number and its position on board
        """
        checker_id = model_output_index // (BOARD_WIDTH * BOARD_HEIGHT)
        offset = model_output_index % (BOARD_WIDTH * BOARD_HEIGHT)
        pos = offset // BOARD_WIDTH, offset % BOARD_WIDTH
        return checker_id, pos

class ResidualCNN(Model):
    def __init__(self, input_dim, filters):
        Model.__init__(self, input_dim, filters)
        self.model = self.build_model()

    def build_model(self):
        main_input = Input(shape=self.input_dim)
        x = self.conv_block(main_input, self.filters, 3, model_configs.REGULARIZER)

        for _ in range(model_configs.NUM_RESIDUAL_BLOCKS):
            x = self.residual_block(x, self.filters, 3, model_configs.REGULARIZER)

        vh = self.value_head(x, model_configs.REGULARIZER)
        ph = self.policy_head(x, model_configs.REGULARIZER)

        model = KerasModel(inputs=[main_input], outputs=[vh, ph])
        model.compile(loss={"value_head":"mean_squared_error", "policy_head":softmax_cross_entropy_with_logits},
                        optimizer=SGD(lr=model_configs.LEARNING_RATE),
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
        x = Dense( self.filters,
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
        x = self.conv_layer(head_input, filters=2, kernel_size=1, regularizer=regularizer)
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

    model = ResidualCNN(input_dim=(7,7,7), filters=24)

