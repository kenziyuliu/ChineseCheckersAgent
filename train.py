"""
Coordinates training procedure, including:

1. invoke self play
2. store result from self play & start training NN immediately based on that single example
3. evaluate how well the training is, via loss, (# games drawn due to 50 moves no progress?, ) etc.
4. save model checkpoints for each 'x' self play
5. allow loading saved model checkpoints given argument
"""
def generate_self_play_data():
    # TODO
    pass

def load_self_play_data(filename):
    # TODO
    pass

def train(model, data):
    # TODO
    pass
