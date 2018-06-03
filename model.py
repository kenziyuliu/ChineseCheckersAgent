import os

from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.models import Model as KerasModel
from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, LeakyReLU, add

import utils
from board import *
from config import *
from loss import softmax_cross_entropy_with_logits


class Model:
    def __init__(self, input_dim, filters, version=0):
        self.input_dim = input_dim
        self.filters = filters
        self.version = version

    def predict(self, input_board):
        logits, v = self.model.predict(np.expand_dims(input_board, axis=0).astype('float64'))
        p = utils.softmax(logits)           # Apply softmax on the logits after prediction
        return p.squeeze(), v.squeeze()     # Remove the extra batch dimension

    def save(self, save_dir, model_prefix, version):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.version = version
        self.model.save('{}/{}{:0>4}.h5'.format(save_dir, model_prefix, version))
        print('\nSaved model "{}{:0>4}.h5" to "{}"\n'.format(model_prefix, version, save_dir))

    def save_weights(self, save_dir, prefix, version):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights('{}/{}{:0>4}-weights.h5'.format(save_dir, prefix, version))
        utils.stress_message('Saved model weights "{}{:0>4}-weights" to "{}"'.format(prefix, version, save_dir))

    def load(self, filepath):
        self.model = load_model(
            filepath,
            custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits}
        )
        return self.model

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        return self.model   # Return reference to model just in case



class ResidualCNN(Model):
    def __init__(self, input_dim=INPUT_DIM, filters=NUM_FILTERS):
        Model.__init__(self, input_dim, filters)
        self.model = self.build_model()


    def build_model(self):
        main_input = Input(shape=self.input_dim)
        x = self.conv_block(main_input, self.filters, 3, REGULARIZER)

        for _ in range(NUM_RESIDUAL_BLOCKS):
            x = self.residual_block(x, self.filters, 3, REGULARIZER)

        value = self.value_head(x, REGULARIZER)
        policy = self.policy_head(x, REGULARIZER)

        model = KerasModel(inputs=[main_input], outputs=[policy, value])
        model.compile(loss={"policy_head":softmax_cross_entropy_with_logits, "value_head":"mean_squared_error"},
                        optimizer=Adam(lr=LEARNING_RATE),
                        loss_weights=LOSS_WEIGHTS)
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
        checker_pos.append(utils.decode_checker_index(i))
        # print(utils.decode_checker_index(i))

    count = 0
    for checker_id, pos in checker_pos:
        assert count == utils.encode_checker_index(checker_id, pos)
        # print(utils.encode_checker_index(checker_id, pos))
        count += 1

    # Test `to_model_input`
    gameboard = Board()
    gameboard.place(1, (5, 0), (3, 0))
    board = gameboard.board
    for i in range(board.shape[2]):
        print(board[:, :, i])
    model_input = utils.to_model_input(board, 1)
    print('\n\n')
    for i in range(model_input.shape[2]):
        print(model_input[:, :, i])

    model = ResidualCNN()
    # test for saving
    model.save(SAVE_MODELS_DIR, None, 1)
