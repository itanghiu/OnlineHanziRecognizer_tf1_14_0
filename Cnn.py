from Data import Data
import utils
import time
import os
import logging
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy
import sys
import shutil
from datetime import datetime
import os.path
from os import path
import ImageDatasetGeneration
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, callbacks,optimizers, utils
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from TimeHistory import TimeHistory


logger = logging.getLogger('Cnn.py')
logger.setLevel(logging.INFO)

LOG_DIR = './log'
LEARNING_RATE = 2e-4
# builds the map whose keys are labels and values characters
label_char_dico = Data.get_label_char_dico(ImageDatasetGeneration.CHAR_LABEL_DICO_FILE_NAME)
MODEL_NAME = 'hanzi_recog_model'
saving_step_frequency = 100
max_steps = 250
STEPS_PER_EPOCH = 25
EPOCHS = 10
root_logdir = os.path.join(os.curdir, "hanzi_recog_logs")
batch_size =100

class Cnn:

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    checkpoint_dir = Data.CHECKPOINT
    evaluation_step_frequency = 250 # Evaluates every 'evaluation_step_frequency' step
    mode = 'training' # Running mode: {"training", "test"}
    batch_size = 100
    saving_step_frequency = 250
    epoch = 15 # Number of epoches
    max_steps = 250 # the max number of steps for training
    restore = True # whether to restore from checkpoint
    random_flip_up_down = False # Whether to random flip up down
    random_brightness = False # whether to adjust brightness
    random_contrast = True # whether to random contrast
    gray = True # whether to change the rbg to gray
    KEEP_NODES_PROBABILITY = 0.5

    # LEARNING_RATE = 2e-5
    # LEARNING_RATE = 7.e-4 does not work : loss stalls at 8.23

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
    FLAGS = tf.compat.v1.app.flags.FLAGS

    def __init__(self):
        self.model = models.Sequential()

    """
    def __init__(self, image_tensor, labels):
        print('Instantiating...')
        self.build_graph(images=image_tensor, labels=labels)
        # Initializing graph...
        self.saver = tf.compat.v1.train.Saver()
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)
    """

    def _loss(self, y, y_pred):
        return tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y))

    def get_run_logdir(self):
        import time
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    #@staticmethod
    def training(self):

        def _loss(y_true, y_pred):
            print('y_pred.shape:', y_pred.shape)
            print('y_true.shape: ', y_true.shape)
            labels = tf.cast(tf.squeeze(y_true), tf.int64)
            return tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels))

        print('Executing eagerly? ', tf.executing_eagerly())

        self.build_graph()
        learning_rate = ExponentialDecay(initial_learning_rate=LEARNING_RATE, decay_steps=2000, decay_rate=0.97, staircase=True)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        #self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        self.model.compile(loss=_loss, optimizer=optimizer, metrics=["accuracy"])
        self.model.run_eagerly = True
        self.model.summary()
        #keras.utils.plot_model(self.model, show_shapes=True)

        checkpoint_callback = callbacks.ModelCheckpoint(MODEL_NAME+".h5", save_freq='epoch')
        time_callback = TimeHistory()
        run_logdir = self.get_run_logdir()
        tensorboard_callback = callbacks.TensorBoard(run_logdir)

        data = Data()
        train_dataset = data.prepare_data_for_training(Data.DATA_TRAINING, batch_size)
        history = self.model.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[time_callback])
        #score = self.model.evaluate(X_test, y_test)
        self.save_model()

    def load_model(self):
        print('Loading model...')
        savedmodel_dir = os.getcwd() + os.sep + MODEL_NAME
        # Restore the model's state
        self.model.load_weights(savedmodel_dir)
        print('End Loading model.')

    #@staticmethod
    def save_model(self):

        # --- Saving model
        logger.info('Saving model %s...'%(MODEL_NAME))
        savedmodel_dir = os.getcwd() + os.sep + MODEL_NAME
        # delete the model directory
        if (path.exists(MODEL_NAME)):
            try:
                shutil.rmtree(savedmodel_dir)
            except OSError as e:
                print("Error: %s : %s" % (savedmodel_dir, e.strerror))

        # Save weights to a HDF5 file
        self.model.save_weights(MODEL_NAME + '.h5', save_format='h5')
        logger.info('End saving model.')

        """"
        probability_ts = graph.get_tensor_by_name('predicted_probability_ts:0')
        index_ts = graph.get_tensor_by_name('predicted_index_ts:0')
        tf.compat.v1.saved_model.simple_save(sess, savedmodel_dir,
                                   #inputs={"image_paths_ph": data.image_paths_ph, "labels_ph": data.labels_ph,"batch_size_ph": data.batch_size_ph},
                                   inputs={"image_paths_ph": data.image_paths_ph},
                                   outputs={"probability": probability_ts, "index": index_ts})
        """

    #@staticmethod
    def build_graph(self):

        input_shape = (Data.IMAGE_SIZE, Data.IMAGE_SIZE, 1)
        # hypothesis: slim.conv2d.num_outputs= Conv2D.filters
        conv3_1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape)
        max_pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        conv3_2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')
        max_pool_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        conv3_3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')
        max_pool_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        conv3_4 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu')
        max_pool_4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        flatten = Flatten()
        fc1 = Dense(1024, activation='relu')
        dropout1 = Dropout(Cnn.KEEP_NODES_PROBABILITY)
        #fc2 = Dense(Data.CHARSET_SIZE, activation='relu')
        #dropout2 = Dropout(Cnn.KEEP_NODES_PROBABILITY)
        logits = Dense(Data.CHARSET_SIZE, activation='softmax')

        self.model.add(conv3_1)
        self.model.add(max_pool_1)
        self.model.add(conv3_2)
        self.model.add(max_pool_2)
        self.model.add(conv3_3)
        self.model.add(max_pool_3)
        self.model.add(conv3_4)
        self.model.add(max_pool_4)
        self.model.add(flatten)
        self.model.add(fc1)
        self.model.add(dropout1)
        self.model.add(logits)
        #self.model.add(dropout2)
        return logits

    def recognize_image_with_model(self, image_file_name):

        graph = tf.Graph()
        with graph.as_default():
            with tf.compat.v1.Session(graph=graph) as sess:
                savedmodel_dir = os.getcwd() + os.sep + 'savedModel'
                tf.compat.v1.saved_model.loader.load(sess, ["serve"], savedmodel_dir)
                labels_ph = graph.get_tensor_by_name('labels_ph:0')
                image_paths_ph = graph.get_tensor_by_name('image_paths_ph:0')
                batch_size_ph = graph.get_tensor_by_name('batch_size_ph:0')

                # initialize data
                init_iterator_op = graph.get_operation_by_name('init_iterator_op')
                image_file_paths = [image_file_name]
                labels = numpy.array([0])
                sess.run(init_iterator_op, feed_dict={image_paths_ph: image_file_paths, labels_ph: labels, batch_size_ph: 1})

                # Compute inference for dataset
                #predicted_probability_ts = graph.get_tensor_by_name('predicted_probability_op:0')
                #predicted_index_ts = graph.get_tensor_by_name('predicted_index_op:0')
                predicted_probability_ts = graph.get_tensor_by_name('predicted_probability_ts:0')
                predicted_index_ts = graph.get_tensor_by_name('predicted_index_ts:0')

                keep_nodes_probabilities_ph = graph.get_tensor_by_name('keep_nodes_probabilities_ph:0')
                is_training_ph = graph.get_tensor_by_name('is_training_ph:0')
                top_k_ph = graph.get_tensor_by_name('top_k_ph:0')

                predicted_probability, predicted_index = sess.run(
                    [predicted_probability_ts, predicted_index_ts],
                    feed_dict={keep_nodes_probabilities_ph: 1.0, is_training_ph: False, top_k_ph: 3})
                cols, rows = predicted_index.shape
                list_length = rows if (rows < 6) else 6
                predicted_indexes2 = predicted_index[0, :list_length]
                predicted_chars = [label_char_dico.get(index) for index in predicted_indexes2]
                predicted_probabilities2 = ["%.1f" % (proba * 100) for proba in predicted_probability[0, :list_length]]
        return predicted_chars, predicted_indexes2, predicted_probabilities2

    def recognize(self):

        ckpt = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt)

        graph = self.sess.graph
        keep_nodes_probabilities_ph = graph.get_tensor_by_name('keep_nodes_probabilities_ph:0')
        is_training_ph = graph.get_tensor_by_name('is_training_ph:0')
        top_k_ph = graph.get_tensor_by_name('top_k_ph:0')
        predicted_probability_op = graph.get_tensor_by_name('probability_and_index_op:0')
        predicted_index_op = graph.get_tensor_by_name('probability_and_index_op:1')

        predicted_probabilities, predicted_indexes = self.sess.run(
            [predicted_probability_op, predicted_index_op],
            feed_dict={keep_nodes_probabilities_ph: 1.0, is_training_ph: False, top_k_ph: 3})

        cols, rows = predicted_indexes.shape
        list_length = rows if (rows < 6) else 6
        predicted_indexes2 = predicted_indexes[0, :list_length]
        print(" ".join(map(str, predicted_indexes2)))
        predicted_chars = [label_char_dico.get(index) for index in predicted_indexes2]
        predicted_probabilities2 = ["%.1f" % (proba * 100) for proba in predicted_probabilities[0, :list_length]]
        return predicted_chars, predicted_indexes2, predicted_probabilities2

    @staticmethod
    def recognize_image(image_file_name):

        cnn = Cnn()
        cnn.load_model()
        data = Data()
        image_file_paths = [image_file_name]
        #X_new = X_test[:10]  # pretend we have new images
        predicted_indexes = cnn.model.predict(image_file_paths)
        predicted_chars = [label_char_dico.get(index) for index in predicted_indexes]
        labels = "x"
        probabilities = "0"
        return predicted_chars, labels, probabilities

        """"
        
        init_iterator_operation = data.get_batch(aug=True)
        data_sample = data.get_next_element()
        image_tensor = data_sample[0]
        labels = numpy.array([0])
        cnn = Cnn(image_tensor=image_tensor, labels=labels)
        cnn.sess.run(init_iterator_operation, feed_dict={data.image_paths_ph: image_file_paths, data.labels_ph: labels, data.batch_size_ph: 1})
        return cnn.recognize()
        """

    @staticmethod
    def start_app():
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
        tf.compat.v1.app.run()

    @staticmethod
    def convert_to_tensor_lite():
        logger.info('Converting model...')
        saved_model_dir = os.getcwd() + os.sep + 'savedModel'
        logger.info('Converting model located at ' + saved_model_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()
        open("converted_model.tflite", "wb").write(tflite_model)


""""
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=Cnn.gpu_options, allow_soft_placement=True,
                                              log_device_placement=True)) as sess:

            sess.run(training_init_op, feed_dict={data.image_paths_ph: images, data.labels_ph: labels, data.batch_size_ph: FLAGS.batch_size})
            sess.run(tf.compat.v1.global_variables_initializer())

            coordinator = tf.train.Coordinator()
            threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coordinator)
            saver = tf.compat.v1.train.Saver()
            train_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + '/training', sess.graph)
            test_writer = tf.compat.v1.summary.FileWriter(LOG_DIR + '/test')
            start_step = 0
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    start_step += int(ckpt.split('-')[-1])
                    print("Restoring from the checkpoint %s at step %d" % (ckpt, start_step))
            logger.info('Start training')
            #logger.info("Training data size: %d" % training_data.size)
            #logger.info("Test data size: %d" % test_data.size)
            print("Getting training data...")
            graph = sess.graph
            keep_nodes_probabilities_ph = graph.get_tensor_by_name('keep_nodes_probabilities_ph:0')
            is_training_ph = graph.get_tensor_by_name('is_training_ph:0')
            top_k_ph = graph.get_tensor_by_name('top_k_ph:0')
            step_tsr = graph.get_tensor_by_name('step:0')
            train_op = graph.get_tensor_by_name('train_op/control_dependency:0')
            loss_tsr = graph.get_tensor_by_name('loss/Mean:0')
            merged_summary_op = graph.get_tensor_by_name('Merge/MergeSummary:0')

            def train():
                start_time = datetime.now()
                # print("Getting training data took %s " % utils.r(start_time))
                feed_dict = {keep_nodes_probabilities_ph: 0.8, is_training_ph: True, top_k_ph: 3}
                _, loss, train_summary, step = sess.run([train_op, loss_tsr, merged_summary_op, step_tsr], feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                logger.info("Step #%s took %s. Loss: %.3f \n" % (step, utils.r(start_time), loss))
                return step

            start_time = datetime.now()
            eval_frequency = FLAGS.evaluation_step_frequency
            # starts training
            while not coordinator.should_stop():

                step = train()
                if step > FLAGS.max_steps:
                    break

                if (step % eval_frequency == 0) and (step >= eval_frequency):
                    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
                    feed_dict = {keep_nodes_probabilities_ph: 1.0, is_training_ph: False}
                    accuracy_test, test_summary = sess.run([accuracy, merged_summary_op], feed_dict=feed_dict)

                    test_writer.add_summary(test_summary, step)
                    logger.info('---------- Step #%d   Test accuracy: %.2f ' % (int(step), accuracy_test))

                if ((step % FLAGS.saving_step_frequency == 0) and (step != 0)):
                    logger.info('Saving checkpoint of step %s' % step)
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'online_hanzi_recog'),
                               global_step=step_tsr)

            logger.info('Training Completed in  %s ' % utils.r(start_time))
            coordinator.request_stop()
            train_writer.close()
            test_writer.close()
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'online_hanzi_recog'), global_step=step_tsr)
            coordinator.join(threads)

            Cnn.save_model(graph, data, sess)
            sess.close()
"""
