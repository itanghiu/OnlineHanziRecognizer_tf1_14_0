import Cnn
import Data
import tensorflow as tf
from Cnn import Cnn

FLAGS = tf.compat.v1.app.flags.FLAGS

def main(_):
    print("Mode: %s " % FLAGS.mode)
    if FLAGS.mode == 'recognize_image':
        image_file_name = 'onlineCharacter5.png'
        characters, labels, probabilities = Cnn.recognize_image(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif FLAGS.mode == 'recognize_image_with_model':
        image_file_name = 'onlineCharacter.png'
        characters, labels, probabilities = Cnn.recognize_image_with_model(image_file_name)
        print('Label: %s  Probability: %s  Character: %s' % (labels[0], probabilities[0], characters[0]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[1], probabilities[1], characters[1]))
        print('Label: %s  Probability: %s  Character: %s' % (labels[2], probabilities[2], characters[2]))

    elif FLAGS.mode == "training":
        Cnn.training()

    elif FLAGS.mode == "convert_to_tensor_lite":
        Cnn.convert_to_tensor_lite()

if __name__ == "__main__":  # Learning mode
    Cnn.start_app()
else:  # webServer mode
    print('Web server mode')
