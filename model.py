from board import *
from constants import *

class Model:
    def __init__(self):
        # TODO
        pass

    @staticmethod
    def to_model_input(board, cur_player, timestep):
        """
        Input:
            board: 7 x 7 x 3 board._board. Each channel contains positions of both players' checkers.
            cur_player: player number of the current player
            timestep: the layer of model_input is all zeros if t<0
        Output:
            7 x 7 x 7. First 3 channel is player 1, next 3 channel is player 2, last channel is all 0 if player 1 is to play.
        """
        model_input = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, NUM_HIST_MOVES * 2 + 1), dtype="uint8") # may change dtype afterwards
        op_player = PLAYER_ONE + PLAYER_TWO - cur_player
        for channel in range(NUM_HIST_MOVES):
            if timestep - channel < 0:
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


if __name__ == '__main__':
    """
    Test cases here
    """
    checker_pos = []
    for i in range(6 * 49 + 1):
        checker_pos.append(Model.decode_checker_index(i))
        print(Model.decode_checker_index(i))

    count = 0
    for checker_id, pos in checker_pos:
        assert count == Model.encode_checker_index(checker_id, pos)
        print(Model.encode_checker_index(checker_id, pos))
        count += 1

        