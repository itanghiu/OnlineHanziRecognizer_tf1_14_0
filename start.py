import cnn
import Data
import tensorflow as tf
from cnn import Cnn

FLAGS = tf.app.flags.FLAGS

def main(_):
    print("Mode: %s " % FLAGS.mode)
    if FLAGS.mode == 'recognize_image':
        image_file_name = 'onlineCharacter5.png'
        characters, labels, probabilities = Cnn.recognize_image(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif FLAGS.mode == 'recognize_with_model':
        image_file_name = 'onlineCharacter5.png'
        characters, labels, probabilities = Cnn.recognize_image_with_model(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif FLAGS.mode == "training":
        Cnn.training()


if __name__ == "__main__":  # Learning mode
    Cnn.start_app()
else:  # webServer mode
    print('Web server mode')
