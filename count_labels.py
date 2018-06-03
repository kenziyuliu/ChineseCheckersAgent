import sys
import h5py
import numpy as np

def count_items(v_y):
    count = dict()
    for val in v_y:
        if val in count:
            count[val] += 1
        else:
            count[val] = 1
    return count


def get_train_label_count(path):
    with h5py.File(path, 'r') as H:
        board_x = np.copy(H['board_x'])
        pi_y = np.copy(H['pi_y'])
        v_y = np.copy(H['v_y'])

    return count_items(v_y)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Missing training data file')
        exit()

    path = sys.argv[1]
    print(get_train_label_count(path))

