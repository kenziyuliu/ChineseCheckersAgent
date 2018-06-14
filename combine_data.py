import os
import h5py
import numpy as np

def combine_train_data(board_x, pi_y, v_y, first_version, last_version, save_dir, pref):
    all_board_x, all_pi_y, all_v_y = [], [], []

    if len(board_x) > 0 and len(pi_y) > 0 and len(v_y) > 0:
        all_board_x.append(board_x)
        all_pi_y.append(pi_y)
        all_v_y.append(v_y)

    # Read data from previous iterations
    for i in range(first_version, last_version + 1):
        if i >= 0:
            filename = '{}/{}{}.h5'.format(save_dir, pref, i)

            if not os.path.exists(filename):
                utils.stress_message('{} does not exist!'.format(filename))
                continue

            with h5py.File(filename, 'r') as H:
                all_board_x.append(np.copy(H['board_x']))
                all_pi_y.append(np.copy(H['pi_y']))
                all_v_y.append(np.copy(H['v_y']))

    if len(all_board_x) > 0 and len(all_pi_y) > 0 and len(all_v_y) > 0:
        # Make a pool of training data from previous iterations
        board_x = np.vstack(all_board_x)
        pi_y = np.vstack(all_pi_y)
        v_y = np.hstack(all_v_y)                        # hstack as v_y is 1D array

        return board_x, pi_y, v_y, len(all_board_x)     # Last retval is total iterations used

    # If no data at all: return empty training data
    return [], [], [], 0


def save_train_data(board_x, pi_y, v_y):
    ''' Write current iteration training data to disk '''
    with h5py.File('combined.h5', 'w') as H:
        H.create_dataset('board_x', data=board_x)
        H.create_dataset('pi_y', data=pi_y)
        H.create_dataset('v_y', data=v_y)



if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('\nUsage: python3 combine_data.py <dir> <pref> <1st version num> <last version num>\n')
        exit()
    
    board_x, pi_y, v_y, num = combine_train_data([], [], [], int(sys.argv[3]), int(sys.argv[4]), sys.argv[1], sys.argv[2])
    save_train_data(board_x, pi_y, v_y)

