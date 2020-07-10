from flask import Flask, render_template, send_from_directory
from flask import request
import base64
from PIL import Image
from io import BytesIO
import cnn

import utils
import logging
import json

logger = logging.getLogger('app.py')
logging.basicConfig(filename='webApp.log', level=logging.DEBUG)
app = Flask(__name__)
recognizer = cnn.get_predictor()
sess, graph, training_init_op, saver, data = cnn.init_graph(utils.HAND_WRITTEN_CHAR_FILE_NAME)

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
        predicted_chars, predicted_indexes, predicted_probabilities = recognizer(sess, graph,
                                                                                 training_init_op, saver)
        logger.info('Predicted chars: ' + ":".join(predicted_chars))
        logger.info('Predicted probabilities: ' + ", ".join(predicted_probabilities))
        result = dict(chars=predicted_chars, probabilities=predicted_probabilities)
        jsonResult = json.dumps(result)
        #return ":".join(predicted_chars)
        return jsonResult


@app.route('/getCharImage/', methods=['GET'])
def get_char_image():
        return "Hi everyone !"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
    cnn.tf.app.run()