# top 1 accuracy 0.9249791286257038 top k accuracy 0.9747623788455786
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import tensorflow as tf
import pickle
from PIL import Image
import utils
from tensorflow.python.ops import control_flow_ops
from datetime import datetime

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#DATA_ROOT_DIR = '/DATA/CASIA/onlineHanziRecognizer'
DATA_ROOT_DIR = '/TEMP_DATA_SET'
DATA_TRAINING = DATA_ROOT_DIR + '/training'
DATA_TEST = DATA_ROOT_DIR + '/test'

tf.app.flags.DEFINE_integer('max_steps', 300, 'the max training steps ')# initVal = 16002
tf.app.flags.DEFINE_integer('evaluation_step', 30, "the step num to eval") # initVal = 100
tf.app.flags.DEFINE_integer('batch_size', 20, 'Validation batch size')# originalValue=128
tf.app.flags.DEFINE_integer('save_steps', 500, "the steps to save")
tf.app.flags.DEFINE_integer('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")

tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'training', 'Running mode: {"training", "validation", "test"}')

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', False, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
FLAGS = tf.app.flags.FLAGS

# self.labels = {'00000', '00001', '00002', ...}

class DataIterator:

    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + os.sep + ('%05d' % FLAGS.charset_size) #display number with 5 leading 0
        print(truncate_path)
        self.image_file_paths = []
        for root, sub_folder, image_file_names in os.walk(data_dir):
            if root < truncate_path:
                self.image_file_paths += [os.path.join(root, image_file_name) for image_file_name in image_file_names]
        random.shuffle(self.image_file_paths)
        # the labels are the name of directories converted to int
        #self.labels = [int(image_file_path[len(data_dir)+1:].split(os.sep)[0]) for image_file_path in self.image_file_paths]
        self.labels = []
        for image_file_path in self.image_file_paths:
            #images_dir_name example : '00000', '00001', '00002'
            images_dir_name = image_file_path[len(data_dir)+1:].split(os.sep)[0]
            self.labels.append(int(images_dir_name))
        print("self.labels size: %d" % len(self.labels))

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):

        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.9, 1.1)# ioriginal Val: 0.8 1.2
        return images

    # converts image_file_paths and labels_tensor to tensors.
    # suffle them
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):

        # convert data into tensor
        images_tensor = tf.convert_to_tensor(self.image_file_paths, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)

        # Instead of loading the entire dataset in RAM, which is very RAM consuming,
        # slice the dataset in batch and read each batch one by one.
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        # Convert data to grey scale
        img = tf.read_file(input_queue[0])
        imgs = tf.image.convert_image_dtype(tf.image.decode_png(img, channels=1), tf.float32)
        if aug:
            imgs = self.data_augmentation(imgs)

        # standardize the image sizes.
        standard_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        imgs = tf.image.resize_images(imgs, standard_size)

        labels = input_queue[1]
        image_batch, label_batch = tf.train.shuffle_batch([imgs, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        # image_batch = shape(128,64,64,1)  label_batch = shape = (128,)
        return image_batch, label_batch

def build_graph(top_k):

    # (1-keep_prob) equals to dropout rate on fully-connected layers
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    # Set up places for data and label, so that we can feed data into network later
    images = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name="img_batch")
    labels = tf.placeholder(tf.int64, shape=[None], name="label_batch")
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')

    # Structure references to : http://yuhao.im/files/Zhang_CNNChar.pdf,
    # however I adjust a little bit due to limited computational resource.
    # Four convolutional layers with kernel size of [3,3], and ReLu as activation function
    conv1 = slim.conv2d(images, 64, [3, 3], 1, padding="SAME", scope="conv1")
    pool1 = slim.max_pool2d(conv1, [2, 2], [2, 2], padding="SAME")
    conv2 = slim.conv2d(pool1, 128, [3, 3], padding="SAME", scope="conv2")
    pool2 = slim.max_pool2d(conv2, [2, 2], [2, 2], padding="SAME")
    conv3 = slim.conv2d(pool2, 256, [3, 3], padding="SAME", scope="conv3")
    pool3 = slim.max_pool2d(conv3, [2, 2], [2, 2], padding="SAME")
    conv4 = slim.conv2d(pool3, 512, [3, 3], [2, 2], scope="conv4", padding="SAME")
    pool4 = slim.max_pool2d(conv4, [2, 2], [2, 2], padding="SAME")
    # Flat the feature map so that we can connect it to fully-connected layers
    flat = slim.flatten(pool4)
    # Two fully-connected layers with dropout rate as mentioned at the start
    # First layer used tanh() as activation function
    fcnet1 = slim.fully_connected(slim.dropout(flat, keep_prob=keep_prob), 1024, activation_fn=tf.nn.tanh,
                                  scope="fcnet1")
    fcnet2 = slim.fully_connected(slim.dropout(fcnet1, keep_prob=keep_prob), FLAGS.charset_size, activation_fn=None, scope="fcnet2")

    # loss function is defined as cross entropy on result of softmax function on last layer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fcnet2, labels=labels))

    # compare result to actual label to get accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fcnet2, 1), labels), tf.float32))

    step = tf.get_variable("step", shape=[], initializer=tf.constant_initializer(0), trainable=False)

    # learning rate with exponential decay
    lrate = tf.train.exponential_decay(2e-4, step, decay_rate=0.97, decay_steps=2000, staircase=True)

    # Adam optimizer to decrease loss value
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss, global_step=step)
    # added
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)

    prob_dist = tf.nn.softmax(fcnet2)
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(prob_dist, 3)
    # Write log into TensorBoard
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    summary = tf.summary.merge_all()
    return {"images": images,
            "labels": labels,
            'keep_prob': keep_prob,
            "global_step": step,
            "optimizer": optimizer,
            "loss": loss,
            "accuracy": accuracy,
            "predicted_val_top_k": predicted_val_top_k,
            "predicted_index_top_k": predicted_index_top_k,
            "merged_summary_op": summary,
            'is_training': is_training,
            'train_op': train_op
            }

def build_graph_orig(top_k):

    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.Session() as sess:
    #with tf.device('/gpu:0'):
    #with tf.device('/cpu:0'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
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
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None, scope='logits')

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
        probabilities = tf.nn.softmax(logits)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('Accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'global_step': global_step,
            "optimizer": optimizer,
            'loss': loss,
            'accuracy': accuracy,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k,
            'merged_summary_op': merged_summary_op,
            'is_training': is_training,

            'accuracy_top_k': accuracy_in_top_k,
            'train_op': train_op,
            'predicted_distribution': probabilities,
            'top_k': top_k
            }


def training():

    train_feeder = DataIterator(data_dir=DATA_TRAINING)
    test_feeder = DataIterator(data_dir=DATA_TEST)
    model_name = 'chinese-rec-model'
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
                                          log_device_placement=True)) as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph(top_k=1)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/training', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint %s" % ckpt)
                start_step += int(ckpt.split('-')[-1])

        logger.info('Start training')
        logger.info("Training data shape: %s" % train_images.get_shape())
        logger.info("Test data shape: %s" % test_images.get_shape())
        logger.info("Training data size: %d" % train_feeder.size)
        logger.info("Test data size: %d" % test_feeder.size)
        print("Getting training data...")
        try:
            i = 0
            while not coordinator.should_stop():
                i += 1
                start_time = datetime.now()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                print("Getting training data took %s " % utils.r(start_time))
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8,
                             graph['is_training']: True}
                #_, loss_val, train_summary, step = sess.run(
                #    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']], feed_dict=feed_dict)
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],  feed_dict=feed_dict)

                train_writer.add_summary(train_summary, step)
                logger.info("Step #%s took %s. Loss: %d \n" % (step, utils.r(start_time), loss_val))
                if step > FLAGS.max_steps:
                    break

                if (step % FLAGS.evaluation_step == 0) and (step >= FLAGS.evaluation_step):
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0,
                                 graph['is_training']: False}
                    accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']],
                                                           feed_dict=feed_dict)
                    #if step > 300:
                    test_writer.add_summary(test_summary, step)
                    logger.info('---------- Step #%d   Test accuracy: %.2f ' % (int(step), accuracy_test))
                if step % FLAGS.save_steps == 1:
                    logger.info('Saving checkpoint of step %s' % step)
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
        finally:
            logger.info('Training Completed in  %s ' % utils.r(start_time))
            coordinator.request_stop()
            train_writer.close()
            test_writer.close()
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
        coordinator.join(threads)


def validation():

    print('Begin validation')
    test_feeder = DataIterator(data_dir=DATA_TEST)
    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        i = 0
        acc_top_1 = 0.0
        acc_top_k = 0.0
        try:
            while not coordinator.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}

                batch_labels, probabilities, \
                indices, acc_1, acc_k = sess.run([graph['labels'],
                                                  graph['predicted_val_top_k'],
                                                  graph['predicted_index_top_k'],
                                                  graph['accuracy'],
                                                  graph['accuracy_top_k']],
                                                 feed_dict=feed_dict)
                final_predict_val += probabilities.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info("The batch {0} took {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))

        finally:
            coordinator.request_stop()
            coordinator.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def buildGraph(sess):
    # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    # Pass a shadow label 0. This label will not affect the computation graph.
    graph = build_graph(top_k=3)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    return graph, sess


def inference(image):
    print('inference')
    recognizer = get_predictor()
    return recognizer(image)


def get_predictor():

    sess = tf.Session()
    graph, sess = buildGraph(sess)

    def recognizer(input_image):
        resized_image = utils.resize_image(input_image, FLAGS.image_size, FLAGS.image_size)
        predicted_probabilities, predicted_indexes = sess.run(
            [graph['predicted_val_top_k'], graph['predicted_index_top_k']],
            feed_dict={graph['images']: resized_image, graph['keep_prob']: 1.0,
                       graph['is_training']: False})
        cols, rows = predicted_indexes.shape
        list_length = rows if (rows < 6) else 6
        predicted_indexes2 = predicted_indexes[0,:list_length]
        predicted_chars = [labelCharMap.get(str(index)) for index in predicted_indexes2]
        predicted_probabilities2 = ["%.1f" % (proba * 100) for proba in predicted_probabilities[0, :list_length]]
        return predicted_chars, predicted_indexes2, predicted_probabilities2

    return recognizer

def interactiveInference():
    recognizer = get_predictor()
    while (True):
        inputData = input(" Enter character image file name (/ to exit)")
        if inputData == "/":
            break
        firstIndex, predict_val, char = recognizer(inputData)
        logger.info('Predicted index: {0} \nPredicted val: {1} \nChar: {2}'
                    .format(firstIndex, predict_val * 100, char))

def main(_):
    print("Mode: %s " % FLAGS.mode)
    if FLAGS.mode == "training":
        training()

    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Write file ends')

    elif FLAGS.mode == 'inference':
        image_path = DATA_ROOT_DIR + '/test/00006/6729.png'
        final_predict_val, final_predict_index, char = inference(image_path)
        logger.info('Predicted index: {0} \nPredicted val: {1} \nChar: {2}'
                    .format(final_predict_index, final_predict_val * 100, char))

    elif FLAGS.mode == 'interactiveInference':
        interactiveInference()

    elif FLAGS.mode == 'testCrop':
        testCrop()


def testCrop():
    image_png = Image.open('webApp/onlineCharacter.png')
    cropped_image = utils.crop_image(image_png)
    cropped_image.save('cropped.png', 'PNG')
    return cropped_image


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
    labelCharMap = utils.loadLabelCharMap('labelCharMapFile.txt')
    tf.app.run()
else:  # webServer mode
    labelCharMap = utils.loadLabelCharMap('../labelCharMapFile.txt')
    tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoint/', 'the checkpoint dir')

