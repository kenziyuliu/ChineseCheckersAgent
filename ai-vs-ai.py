import re
from game import Game
from config import *
import sys
from model import *
"""
Run this file with argument specifying the models from terminal if you
want to play ai-vs-ai game
e.g. python3 ai-vs-ai.py saved-models/version0000.h5 saved-models/version0033.h5
"""

if __name__ == '__main__':
    count = { PLAYER_ONE : 0, PLAYER_TWO : 0 }
    if len(sys.argv) == 3:
        model1 = ResidualCNN()
        filename1 = sys.argv[1]
        version_num1 = int(re.search('{}(.+?)\.h5'.format(MODEL_PREFIX), filename1).group(1))
        model1.version = version_num1
        print("\nLoading model1 from path {}".format(filename1))
        model1.load(filename1)
        print("Model1 is loaded sucessfully\n")

        model2 = ResidualCNN()
        filename2 = sys.argv[2]
        version_num2 = int(re.search('{}(.+?)\.h5'.format(MODEL_PREFIX), filename2).group(1))
        model2.version = version_num2
        print("Loading model2 from path {}".format(filename2))
        model2.load(filename2)
        print("Model2 is loaded sucessfully\n")
        for i in range(1):
            print(i, end=' ')
            game = Game(p1_type='ai', p2_type='ai', verbose=True, model1 = model1, model2 = model2)
            winner = game.start()
            count[winner] += 1

        print('AiPlayer {} wins {} matches'.format(PLAYER_ONE, count[PLAYER_ONE]))
        print('AiPlayer {} wins {} matches'.format(PLAYER_TWO, count[PLAYER_TWO]))
