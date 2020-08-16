import os
import tensorflow.contrib.slim as slim
import logging
import tensorflow as tf
import utils
import numpy
import shutil
import ImageDatasetGeneration
from Data import Data
from tensorflow.python.ops import control_flow_ops
from datetime import datetime
import os.path
from os import path

logger = logging.getLogger('cnn.py')
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
    tf.app.flags.DEFINE_string('checkpoint_dir', Data.CHECKPOINT, 'the checkpoint dir')
    tf.app.flags.DEFINE_integer('evaluation_step_frequency', 250,
                                "Evaluates every 'evaluation_step_frequency' step")  # 30
    tf.app.flags.DEFINE_string('mode', 'training', 'Running mode: {"training", "test"}')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')  # 20
    tf.app.flags.DEFINE_integer('saving_step_frequency', 250, "Save the network every 'saving_step_frequency' steps")
    tf.app.flags.DEFINE_integer('epoch', 15, 'Number of epoches')
    tf.app.flags.DEFINE_integer('max_steps', 251, 'the max number of steps for training')  # 300
    tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')

    tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
    tf.app.flags.DEFINE_boolean('random_brightness', False, "whether to adjust brightness")
    tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random contrast")
    tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    FLAGS = tf.app.flags.FLAGS

    def __init__(self, image_tensor=None):

        print('Initializing graph...')
        self.graph_map = self.build_graph_for_recognition(top_k=3, images=image_tensor)
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('End of graph initialization.')

    @staticmethod
    def training():

        FLAGS = Cnn.FLAGS
        batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
        images_ph = tf.placeholder(dtype=tf.string,  name='images_ph')
        labels_ph = tf.placeholder(dtype=tf.int32, name='labels_ph')
        training_data = Data(images_ph=images_ph, labels_ph=labels_ph, batch_size_ph=batch_size_ph)
        test_data = Data()
        images, labels = Data.prepare_data_for_training(Data.DATA_TRAINING)
        training_init_op = training_data.get_batch(aug=True)
        next_training_sample = training_data.get_next_element()
        input_tensor = next_training_sample[0]
        labels_tensor = next_training_sample[1]

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        with tf.Session(config=tf.ConfigProto(gpu_options=Cnn.gpu_options, allow_soft_placement=True,
                                              log_device_placement=True)) as sess:

            sess.run(training_init_op, feed_dict={images_ph: images, labels_ph: labels, batch_size_ph: FLAGS.batch_size})
            graph_map = Cnn.build_graph_for_training(top_k=1, images=input_tensor, labels=labels_tensor)# returned value graph is a dictionary
            sess.run(tf.global_variables_initializer())

            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
            saver = tf.train.Saver()
            train_writer = tf.summary.FileWriter(LOG_DIR + '/training', sess.graph)
            test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
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
            print("nodes:", [n.name for n in graph.as_graph_def().node])

            def train():
                start_time = datetime.now()
                # print("Getting training data took %s " % utils.r(start_time))
                #loss = graph.get_tensor_by_name('loss/loss:0')

                feed_dict = {keep_nodes_probabilities_ph: 0.8, is_training_ph: True}
                _, loss, train_summary, step = sess.run([graph_map['train_op'], graph_map['loss'], graph_map['merged_summary_op'], graph_map['step']], feed_dict=feed_dict)
                #_, loss, train_summary, step = sess.run([graph_map['train_op'], loss, graph_map['merged_summary_op'], graph_map['step']], feed_dict=feed_dict)

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
                    feed_dict = {keep_nodes_probabilities_ph: 1.0, is_training_ph: False}
                    accuracy_test, test_summary = sess.run([graph_map['accuracy'], graph_map['merged_summary_op']],
                                                           feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    logger.info('---------- Step #%d   Test accuracy: %.2f ' % (int(step), accuracy_test))

                if ((step % FLAGS.saving_step_frequency == 0) and (step != 0)):
                    logger.info('Saving checkpoint of step %s' % step)
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'online_hanzi_recog'),
                               global_step=graph_map['step'])

            logger.info('Training Completed in  %s ' % utils.r(start_time))
            coordinator.request_stop()
            train_writer.close()
            test_writer.close()
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'online_hanzi_recog'), global_step=graph_map['step'])
            coordinator.join(threads)

            # --- Saving model
            logger.info('Saving model...')
            savedmodel_dir = os.getcwd() + os.sep + 'savedModel'
            # delete the model directory
            if (path.exists('savedModel')):
                try:
                    shutil.rmtree(savedmodel_dir)
                except OSError as e:
                    print("Error: %s : %s" % (savedmodel_dir, e.strerror))
            output = graph_map['predicted_index_top_k']
            tf.identity(output, name='output')
            tf.saved_model.simple_save(sess, savedmodel_dir, inputs={"images_ph": images_ph, "labels_ph": labels_ph, "batch_size_ph": batch_size_ph},
                                       outputs={"output": output})
            logger.info('End saving model.')
            sess.close()

    @staticmethod
    def build_main_graph(images):

        keep_nodes_probabilities_ph = tf.placeholder(dtype=tf.float32, shape=[], name='keep_nodes_probabilities_ph')
        is_training_ph = tf.placeholder(dtype=tf.bool, shape=[], name='is_training_ph')

        with tf.variable_scope("convolutional_layer"):
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

        with tf.variable_scope("fc_layer"):  # fully connected layer
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_nodes_probabilities_ph), 1024, activation_fn=tf.nn.relu,
                                       scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_nodes_probabilities_ph), Data.CHARSET_SIZE,
                                          activation_fn=None, scope='fc2')
            tf.identity(logits, name='output')

        return logits

    @staticmethod
    def build_graph_for_recognition(top_k, images):

        logits = Cnn.build_main_graph(images)
        probabilities = tf.nn.softmax(logits)
        merged_summary_op = tf.summary.merge_all()
        predicted_probabilities_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k, name='predicted_index_top_k')

        return { 'predicted_probabilities_top_k': predicted_probabilities_top_k,
                'predicted_index_top_k': predicted_index_top_k,
                'merged_summary_op': merged_summary_op
                }

    @staticmethod
    def build_graph_for_training(top_k, images, labels):

        logits = Cnn.build_main_graph(images)
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        with tf.variable_scope("accuracy"):
            math_ops = tf.cast(tf.argmax(logits, 1), tf.int32)
            #math_ops = tf.cast(tf.argmax(logits, 1), tf.int64)
            tensor_flag = tf.equal(math_ops, labels)
            # compare result to actual label to get accuracy
            accuracy = tf.reduce_mean(tf.cast(tensor_flag, tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=step, decay_rate=0.97,
                                                   decay_steps=2000, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # the step will be incremented after the call to optimizer.minimize()
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)

        probabilities = tf.nn.softmax(logits)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        predicted_probabilities_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        tf.identity(predicted_probabilities_top_k, name='predicted_probabilities_top_k')
        tf.identity(predicted_index_top_k, name='predicted_index_top_k')
        tf.identity(loss, name='loss')

        return {
                'step': step,
                "optimizer": optimizer,
                'loss': loss,
                'accuracy': accuracy,
                'predicted_probabilities_top_k': predicted_probabilities_top_k,
                'predicted_index_top_k': predicted_index_top_k,
                'merged_summary_op': merged_summary_op,
                'train_op': train_op,
                'logits': logits
                }


    def recognize(self):

        ckpt = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt)

        graph_map = self.graph_map
        graph = self.sess.graph
        keep_nodes_probabilities_ph = graph.get_tensor_by_name('keep_nodes_probabilities_ph:0')
        is_training_ph = graph.get_tensor_by_name('is_training_ph:0')
        predicted_probabilities, predicted_indexes = self.sess.run(
            [graph_map['predicted_probabilities_top_k'], graph_map['predicted_index_top_k']],
            feed_dict={keep_nodes_probabilities_ph: 1.0, is_training_ph: False})

        cols, rows = predicted_indexes.shape
        list_length = rows if (rows < 6) else 6
        predicted_indexes2 = predicted_indexes[0, :list_length]
        print(" ".join(map(str, predicted_indexes2)))
        predicted_chars = [label_char_dico.get(index) for index in predicted_indexes2]
        predicted_probabilities2 = ["%.1f" % (proba * 100) for proba in predicted_probabilities[0, :list_length]]
        return predicted_chars, predicted_indexes2, predicted_probabilities2

    @staticmethod
    def recognize_image_with_model(image_file_name):

        print('Loading model...')
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session(graph=graph) as sess:
                savedmodel_dir = os.getcwd() + os.sep + 'savedModel'
                tf.saved_model.loader.load(sess, ["serve"], savedmodel_dir)

                #graph = tf.get_default_graph()
                print(graph.get_operations())
                sess = tf.Session()

                images_ph = graph.get_tensor_by_name('images_ph:0')
                labels_ph = graph.get_tensor_by_name('labels_ph:0')
                batch_size_ph = graph.get_tensor_by_name('batch_size_ph:0')

                # Get restored model output
                #restored_logits = graph.get_tensor_by_name('predicted_index:0')

                init_iterator_operation = graph.get_operation_by_name('dataset_init')
                image_file_paths = [image_file_name]
                labels = numpy.array([0])
                # initialize data
                sess.run(init_iterator_operation, feed_dict={images_ph: image_file_paths, labels_ph: labels, batch_size_ph:1})

                # Compute inference for dataset
                predicted_probabilities_top_k = graph.get_operation_by_name('predicted_probabilities_top_k')
                predicted_index_top_k = graph.get_operation_by_name('predicted_index_top_k')
                keep_nodes_probabilities = graph.get_tensor_by_name('keep_nodes_probabilities:0')
                is_training = graph.get_tensor_by_name('is_training:0')

                #restored_logits = sess.run(restored_logits, feed_dict={graph['keep_nodes_probabilities']: 1.0, graph['is_training']: False})

                predicted_probabilities, predicted_indexes = sess.run(
                 #    [restored_logits, graph['predicted_probabilities_top_k'], graph['predicted_index_top_k']],
                    [predicted_probabilities_top_k, predicted_index_top_k],
                    feed_dict={keep_nodes_probabilities: 1.0, is_training: False})

                cols, rows = predicted_indexes.shape
                list_length = rows if (rows < 6) else 6
                predicted_indexes2 = predicted_indexes[0, :list_length]
                # print(" ".join(map(str, predicted_indexes2)))
                predicted_chars = [label_char_dico.get(index) for index in predicted_indexes2]
                predicted_probabilities2 = ["%.1f" % (proba * 100) for proba in predicted_probabilities[0, :list_length]]
                return predicted_chars, predicted_indexes2, predicted_probabilities2

    @staticmethod
    def recognize_image(image_file_name):

        batch_size_ph = tf.placeholder(tf.int64, name='batch_size_ph')
        images_ph = tf.placeholder(dtype=tf.string,  name='images_ph')
        labels_ph = tf.placeholder(dtype=tf.int32, name='labels_ph')
        data = Data(images_ph=images_ph, labels_ph=labels_ph, batch_size_ph=batch_size_ph)
        image_file_paths = [image_file_name]
        init_iterator_operation = data.get_batch(aug=True)
        data_sample = data.get_next_element()
        image_tensor = data_sample[0]
        labels = numpy.array([0])
        cnn = Cnn(image_tensor)
        cnn.sess.run(init_iterator_operation, feed_dict={images_ph: image_file_paths, labels_ph: labels, batch_size_ph: 1})
        return cnn.recognize()

    @staticmethod
    def start_app():
        tf.app.run()

    def define_string(key, value, comment):
        tf.app.flags.DEFINE_string(key, value, comment)

