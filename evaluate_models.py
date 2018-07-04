import sys
import random
import numpy as np
import multiprocessing as mp

import utils
from config import *


def evaluate(worker_id, model1, model2, num_games):
    # Load the current model in the worker only for prediction and set GPU limit
    import tensorflow as tf
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    from keras.backend.tensorflow_backend import set_session
    set_session(session=session)

    # Re-seed the generators: since the RNG was copied from parent process
    np.random.seed()        # None seed to source from /dev/urandom
    random.seed()

    print('Worker {}: matching between {} and {} with {} games'.format(worker_id, model1, model2 or 'Greedy', num_games))
    if model2 is not None:
        from ai_vs_ai import agent_match as match
    else:
        from ai_vs_greedy import agent_greedy_match as match

    model1_wincount = model2_wincount = draw_count = 0

    for i in range(num_games):
        winner = None
        if model2 is None:
            winner = match(model1, num_games=1)
        else:
            # Alternate players
            if i % 2 == 0:
                winner = match(model1, model2, num_games=1)
            else:
                winner = match(model2, model1, num_games=1)

        if winner is None:
            draw_count += 1
        elif winner == model1:
            model1_wincount += 1
        else:
            model2_wincount += 1

    print('Worker {}: model1 "{}" wins {}/{} games'.format(worker_id, model1, model1_wincount, num_games))
    print('Worker {}: model2/greedy wins {}/{} games'.format(worker_id, model2_wincount, num_games))
    print('Worker {}: {}/{} games were draw'.format(worker_id, draw_count, num_games))
    return model1_wincount, model2_wincount, draw_count



def evaluate_in_parallel(model1, model2, num_games, num_workers):
    if model2 is not None:
        utils.stress_message('Evaluating model "{}" against model "{}" on {} games'
            .format(model1, model2, num_games), True)

    # Process pool for parallelism
    process_pool = mp.Pool(processes=num_workers)
    work_share = num_games // num_workers
    worker_results = []

    # Send processes to generate self plays
    for i in range(num_workers):
        if i == num_workers - 1:
            work_share += (num_games % num_workers)

        # Send workers
        result_async = process_pool.apply_async(
            evaluate,
            args=(i + 1, model1, model2, work_share))
        worker_results.append(result_async)

    try:
        # Join processes and count games
        model1_wincount = model2_wincount = draw_count = 0
        for result in worker_results:
            game_stats = result.get()
            model1_wincount += game_stats[0]
            model2_wincount += game_stats[1]
            draw_count += game_stats[2]

        process_pool.close()

    # Exit early if need
    except KeyboardInterrupt:
        utils.stress_message('SIGINT caught, exiting')
        process_pool.terminate()
        process_pool.join()
        exit()


    process_pool.join()

    utils.stress_message('Overall, model1 "{}" wins {}/{} against model2 "{}"'
        .format(model1, model1_wincount, num_games, model2), True)

    return model1_wincount, model2_wincount, draw_count



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('\nUsage: python3 evaluate_models.py <Model1> [<Model2> <Number of games>]')
        exit()

    path1 = sys.argv[1]
    path2 = None
    num_games = 100

    if len(sys.argv) > 2:
        path2 = sys.argv[2]
    if len(sys.argv) > 3:
        num_games = int(sys.argv[3])

    if path2 is None:
        print('\nModel 2 is not given, evaluating model1 against greedy player')

    p1_wincount, p2_wincount, draw_count = evaluate_in_parallel(path1, path2, num_games, NUM_WORKERS)

    num_nondraws = num_games - draw_count

    message = '''
    With {5} games in total:
        Model 1 wins {0}/{1} games ({6}%)
        {2} wins {3}/{1} games     ({7}%)
        {4} Games were draw
    '''.format(p1_wincount, num_nondraws, (path2 or 'Greedy')
             , p2_wincount, draw_count, num_games
             , 100*p1_wincount/num_nondraws, 100*p2_wincount/num_nondraws)

    print(message)


