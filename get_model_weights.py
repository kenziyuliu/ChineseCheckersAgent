import sys
from keras.models import load_model
from loss import softmax_cross_entropy_with_logits

def get_weights(filename):
    print('Loading model...')
    model = load_model(filename, custom_objects={'softmax_cross_entropy_with_logits': softmax_cross_entropy_with_logits})
    savename = '{}_weights.h5'.format(filename)
    print('Saving model weights to "{}"'.format(savename))
    model.save_weights(savename)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nUsage: python3 get_model_weights.py <model path>\n')
        exit()

    filename = sys.argv[1]
    get_weights(filename)