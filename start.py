import Cnn
import Data
import sys
import tensorflow as tf
from Cnn import Cnn

FLAGS = tf.compat.v1.app.flags.FLAGS

def main(**kwargs):

    for key, value in kwargs.iteritems():
        print (' key:{}  value: {}'.format(key, value))
    print("Mode: %s " % Cnn.mode)
    cnn = Cnn()
    if FLAGS.mode == 'recognize_image':
        image_file_name = 'images/onlineCharacter5.png'
        characters, labels, probabilities = cnn.recognize_image(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif FLAGS.mode == 'recognize_image_with_model':
        image_file_name = 'images/onlineCharacter.png'
        characters, labels, probabilities = cnn.recognize_image_with_model(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif FLAGS.mode == "training":
        cnn.training()

    elif FLAGS.mode == "convert_to_tensor_lite":
        cnn.convert_to_tensor_lite()

if __name__ == "__main__":  # Learning mode

    arg = sys.argv[1:]
    mode = arg[0].split('=')[1]
    print("Mode: %s " % mode)
    print('Python version:', sys.version)
    print('Tensorflow version:', tf.version.VERSION)
    var = tf.Variable([3, 3])
    #if tf.test.is_gpu_available():
    if tf.config.list_physical_devices('GPU'):
        print('Running on GPU')
        print('GPU #0?')
        print(var.device.endswith('GPU:0'))
    else:
        print('Running on CPU')
    cnn = Cnn()
    if mode == 'recognize_image':
        image_file_name = 'images/onlineCharacter5.png'
        characters, labels, probabilities = cnn.recognize_image(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif mode == 'recognize_image_with_model':
        image_file_name = 'images/onlineCharacter.png'
        characters, labels, probabilities = cnn.recognize_image_with_model(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif mode == "training":
        cnn.training()

    elif mode == "convert_to_tensor_lite":
        cnn.convert_to_tensor_lite()

else:  # webServer mode
    print('Web server mode')
