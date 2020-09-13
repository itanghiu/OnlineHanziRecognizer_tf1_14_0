import ImageDatasetGeneration
from Data import Data
import utils

import os
import tf_slim as slim
import logging
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy
import sys
import shutil
from tensorflow.python.ops import control_flow_ops
from datetime import datetime
import os.path
from os import path

logger = logging.getLogger('Cnn.py')
logger.setLevel(logging.INFO)

LOG_DIR = './log'
LEARNING_RATE = 2e-4
# builds the map whose keys are labels and values characters
label_char_dico = Data.get_label_char_dico(ImageDatasetGeneration.CHAR_LABEL_DICO_FILE_NAME)

class Cnn:

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    # LEARNING_RATE = 2e-5
    # LEARNING_RATE = 7.e-4 does not work : loss stalls at 8.23
    tf.compat.v1.app.flags.DEFINE_string('checkpoint_dir', Data.CHECKPOINT, 'the checkpoint dir')
    tf.compat.v1.app.flags.DEFINE_integer('evaluation_step_frequency', 250,
                                "Evaluates every 'evaluation_step_frequency' step")  # 30
    tf.compat.v1.app.flags.DEFINE_string('mode', 'training', 'Running mode: {"training", "test"}')
    tf.compat.v1.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')  # 20
    tf.compat.v1.app.flags.DEFINE_integer('saving_step_frequency', 250, "Save the network every 'saving_step_frequency' steps")
    tf.compat.v1.app.flags.DEFINE_integer('epoch', 15, 'Number of epoches')
    tf.compat.v1.app.flags.DEFINE_integer('max_steps', 250, 'the max number of steps for training')  # 300
    tf.compat.v1.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')

    tf.compat.v1.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
    tf.compat.v1.app.flags.DEFINE_boolean('random_brightness', False, "whether to adjust brightness")
    tf.compat.v1.app.flags.DEFINE_boolean('random_contrast', True, "whether to random contrast")
    tf.compat.v1.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
    FLAGS = tf.compat.v1.app.flags.FLAGS

    def __init__(self, image_tensor, labels):
        print('Instantiating...')
        self.build_graph(images=image_tensor, labels=labels)
        # Initializing graph...
        self.saver = tf.compat.v1.train.Saver()
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)

    @staticmethod
    def training():

        FLAGS = Cnn.FLAGS
        data = Data()
        test_data = Data()
        images, labels = Data.prepare_data_for_training(Data.DATA_TRAINING)
        training_init_op = data.get_batch(aug=True)
        next_training_sample = data.get_next_element()
        images_tensor = next_training_sample[0]
        labels_tensor = next_training_sample[1]
        Cnn.build_graph(images=images_tensor, labels=labels_tensor)

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

    @staticmethod
    def save_model(graph, data, sess):

        # --- Saving model
        logger.info('Saving model...')
        savedmodel_dir = os.getcwd() + os.sep + 'savedModel'
        # delete the model directory
        if (path.exists('savedModel')):
            try:
                shutil.rmtree(savedmodel_dir)
            except OSError as e:
                print("Error: %s : %s" % (savedmodel_dir, e.strerror))
        # output = graph.get_operation_by_name('probability_and_index_op')
        #probability = graph.get_tensor_by_name('probability_and_index_op:0')
        #index = graph.get_tensor_by_name('probability_and_index_op:1')

        probability_ts = graph.get_tensor_by_name('predicted_probability_ts:0')
        index_ts = graph.get_tensor_by_name('predicted_index_ts:0')

        tf.compat.v1.saved_model.simple_save(sess, savedmodel_dir,
                                   #inputs={"image_paths_ph": data.image_paths_ph, "labels_ph": data.labels_ph,"batch_size_ph": data.batch_size_ph},
                                   inputs={"image_paths_ph": data.image_paths_ph},
                                   outputs={"probability": probability_ts, "index": index_ts})
        logger.info('End saving model.')

    @staticmethod
    def build_graph(images, labels):

        keep_nodes_probabilities_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name='keep_nodes_probabilities_ph')
        is_training_ph = tf.compat.v1.placeholder(dtype=tf.bool, shape=[], name='is_training_ph')

        with tf.compat.v1.variable_scope("convolutional_layer"):
            # stride = 1
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')

            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')

            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')

            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

        flatten = slim.flatten(max_pool_4)

        with tf.compat.v1.variable_scope("fc_layer"):  # fully connected layer
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_nodes_probabilities_ph), 1024,
                                       activation_fn=tf.nn.relu,
                                       scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_nodes_probabilities_ph), Data.CHARSET_SIZE,
                                               activation_fn=None, scope='fc2')

        with tf.compat.v1.variable_scope("loss"):
            loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        with tf.compat.v1.variable_scope("accuracy"):
            math_ops = tf.cast(tf.argmax(input=logits, axis=1), tf.int32)
            tensor_flag = tf.equal(math_ops, labels)
            # compare result to actual label to get accuracy
            accuracy = tf.reduce_mean(input_tensor=tf.cast(tensor_flag, tf.float32), name='accuracy')

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        #if update_ops:
        #     updates = tf.group(*update_ops)
        #     loss = control_flow_ops.with_dependencies([updates], loss)
        tf.compat.v1.disable_resource_variables() # to ensure v2 compatibility
        step = tf.compat.v1.get_variable("step", [], initializer=tf.compat.v1.constant_initializer(0.0), trainable=False)
        learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=step, decay_rate=0.97,
                                                   decay_steps=2000, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        # the step will be incremented after the call to optimizer.minimize()
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)

        probabilities = tf.nn.softmax(logits)

        tf.compat.v1.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.compat.v1.summary.merge_all()

        top_k_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[], name='top_k_ph')
        predicted_probability_ts, predicted_index_ts = tf.nn.top_k(probabilities, k=top_k_ph, name='probability_and_index_op')
        tf.identity(predicted_probability_ts, name="predicted_probability_ts")
        tf.identity(predicted_index_ts, name="predicted_index_ts")
        # To retrieve predicted_probability_ts, predicted_index_ts using get_tensor_from_name() :
        # predicted_probability_ts = graph.get_tensor_by_name('probability_and_index_op:0')
        # predicted_index_ts = graph.get_tensor_by_name('probability_and_index_op:1')

    def recognize_image_with_model(image_file_name):

        print('Loading model...')
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

        data = Data()
        image_file_paths = [image_file_name]
        init_iterator_operation = data.get_batch(aug=True)
        data_sample = data.get_next_element()
        image_tensor = data_sample[0]
        labels = numpy.array([0])
        cnn = Cnn(image_tensor=image_tensor, labels=labels)
        cnn.sess.run(init_iterator_operation, feed_dict={data.image_paths_ph: image_file_paths, data.labels_ph: labels, data.batch_size_ph: 1})
        return cnn.recognize()

    @staticmethod
    def start_app():
        print('Python version:', sys.version)
        print('Tensorflow version:', tf.version.VERSION)
        var = tf.Variable([3, 3])
        if tf.test.is_gpu_available():
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
