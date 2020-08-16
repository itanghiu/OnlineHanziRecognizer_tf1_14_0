from flask import Flask, render_template, send_from_directory
from flask import request
import base64
from PIL import Image
from io import BytesIO
import cnn
import numpy
import utils
import logging
import json
from Data import Data
import tensorflow as tf
from cnn import Cnn

logger = logging.getLogger('app.py')
logging.basicConfig(filename='webApp.log', level=logging.DEBUG)
app = Flask(__name__)
batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
images_ph = tf.placeholder(dtype=tf.string, name='images_ph')
labels_ph = tf.placeholder(dtype=tf.int32, name='labels_ph')
data = Data(images_ph=images_ph, labels_ph=labels_ph, batch_size_ph=batch_size_ph)
data.image_file_paths = [utils.HAND_WRITTEN_CHAR_FILE_NAME]
init_iterator_operation = data.get_batch(aug=True)
data_sample = data.get_next_element()
image_tensor = data_sample[0]
labels_tensor = data_sample[1]
cnn = Cnn(image_tensor)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/addCharImage/', methods=['POST'])
def add_char_image():
        charBase64 = request.get_json().get("value")
        image = Image.open(BytesIO(base64.b64decode(charBase64))) #converts from base64 to png
        # image is RGB
        image = utils.set_image_background_to_white(image)
        image = utils.crop_image(image)
        hand_written_char_file_name = utils.HAND_WRITTEN_CHAR_FILE_NAME
        image.save(hand_written_char_file_name, 'PNG')
        labels = numpy.array([0])
        cnn.sess.run(init_iterator_operation, feed_dict={images_ph: [hand_written_char_file_name], labels_ph: labels, batch_size_ph: 1})
        predicted_chars, predicted_indexes, predicted_probabilities = cnn.recognize()
        logger.info('Predicted chars: ' + ":".join(predicted_chars))
        logger.info('Predicted probabilities: ' + ", ".join(predicted_probabilities))
        result = dict(chars=predicted_chars, probabilities=predicted_probabilities)
        jsonResult = json.dumps(result)
        return jsonResult


@app.route('/getCharImage/', methods=['GET'])
def get_char_image():
        return "Hi everyone !"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
    cnn.tf.app.run()