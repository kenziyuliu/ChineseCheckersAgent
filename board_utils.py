from constants import *

def np_index_to_human_coord(coord):
    np_i, np_j = coord
    human_row = np_i - np_j + BOARD_WIDTH
    human_col = min(np_i, np_j) + 1
    return human_row, human_col

def human_coord_to_np_index(coord):
    human_row, human_col = coord
    np_i = human_col - 1 + max(0, human_row - BOARD_WIDTH)
    np_j = human_col - 1 - min(0, human_row - BOARD_WIDTH)
    return np_i, np_j

def is_valid_pos(i, j):
    return i >= 0 and i < BOARD_HEIGHT and j >= 0 and j < BOARD_WIDTH

def convert_np_to_human_moves(np_moves):
    return { np_index_to_human_coord(key) : \
            [np_index_to_human_coord(to) for to in np_moves[key]] \
             for key in np_moves }


if __name__ == '__main__':
    """
    Put board_utils.py test cases here.
    """
    print(human_coord_to_np_index((12, 1)))
    print(human_coord_to_np_index((10, 1)))
    print(human_coord_to_np_index((5, 3)))
