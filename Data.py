import random
import os
import tensorflow as tf
import utils
import codecs
from datetime import datetime

class Data:

    CHECKPOINT = 'C:\\Users\\I-Tang\\DATA\\DEV\\TENSORFLOW\\OnlineHanziRecognizer_tf\\new_checkpoint'
    DATA_ROOT_DIR = 'E:\CHINESE_CHARACTER_RECOGNIZER\CASIA\TEMP_GENERATED_DATASET'
    DATA_TRAINING = DATA_ROOT_DIR + '/training_light'
    DATA_TEST = DATA_ROOT_DIR + '/test_light'
    CHARSET_SIZE = 3755
    IMAGE_SIZE = 64

    def __init__(self, images_ph=None, labels_ph=None, batch_size_ph=1, random_flip_up_down=False, random_brightness=False, random_contrast=True):

        self.images_ph = images_ph
        self.labels_ph = labels_ph
        self.batch_size_ph = batch_size_ph
        self.random_flip_up_down = random_flip_up_down
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast


    @staticmethod
    def augmentation(self, images):

        if self.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if self.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        elif self.random_contrast:
            images = tf.image.random_contrast(images, 0.9, 1.1)  #  0.8 1.2
        return images

    @staticmethod
    def get_label_char_dico(file):

        path = os.getcwd() + os.sep + file
        char_label_dictionary = Data.load_char_label_dico(path)
        label_char_dico = Data.get_label_char_map(char_label_dictionary)
        return label_char_dico

    def get_label_char_map(character_label_dico):
        inverted_map = {v: k for k, v in character_label_dico.items()}
        return inverted_map

    @property
    def size(self):
        return len(self.labels)

    # returns a `tf.Operation` that can be run to initialize this iterator on the dataset
    def get_batch(self, aug=False):

        def _parse_function(filename):
            # convert to grey
            image = tf.read_file(filename)
            image_grey = tf.image.convert_image_dtype(tf.image.decode_png(image, channels=1), tf.float32)
            if aug:
                image_grey = self.augmentation(self, image_grey)

            # standardize the image size .
            standard_size = tf.constant([Data.IMAGE_SIZE, Data.IMAGE_SIZE], dtype=tf.int32)
            images = tf.image.resize_images(image_grey, standard_size)
            return images

        image_file_path_dataset = tf.data.Dataset.from_tensor_slices(self.images_ph)
        label_dataset = tf.data.Dataset.from_tensor_slices(self.labels_ph)
        image_file_path_dataset = image_file_path_dataset.map(_parse_function)

        # zip the x and y training data together and shuffle, batch etc.
        dataset = tf.data.Dataset.zip((image_file_path_dataset, label_dataset)).shuffle(500).repeat().batch(self.batch_size_ph)

        self.iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        init_iterator_operation = self.iterator.make_initializer(dataset) # returns a `tf.Operation` that can be run to initialize this iterator on the dataset
        return init_iterator_operation

    def get_next_element(self):
        next_element = self.iterator.get_next()
        return next_element

    def load_char_label_dico(filePath):

        print("Loading CharLabelDico ... ")
        start_time = datetime.now()
        charLabelMap = {}
        with codecs.open(filePath, 'r', 'gb2312') as f:
            for line in f:
                lineWithoutCR = line.split("\n")[0]
                splitted = lineWithoutCR.split(" ")
                char = splitted[0]
                label = int(splitted[1])
                charLabelMap[char] = label
        print("Execution time: %s s." % utils.r(start_time))
        return charLabelMap

    @staticmethod
    def prepare_data_for_training(data_dir, ):

        truncate_path = data_dir + os.sep + ('%05d' % Data.CHARSET_SIZE)  # display number with 5 leading 0
        image_file_paths = []
        for root, sub_folder, image_file_names in os.walk(data_dir):
            if root < truncate_path:
                image_file_paths += [os.path.join(root, image_file_name) for image_file_name in image_file_names]
        random.shuffle(image_file_paths)
        # the labels are the name of directories converted to int: {'00000', '00001', '00002', ...}
        labels = []
        for image_file_path in image_file_paths:
            # images_dir_name example : '00000', '00001', '00002'
            images_dir_name = image_file_path[len(data_dir) + 1:].split(os.sep)[0]
            img_dir_name = int(images_dir_name)
            labels.append(img_dir_name)
        print("self.labels size: %d" % len(labels))
        return image_file_paths, labels