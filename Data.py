import random
import os
import tensorflow as tf
import utils
import codecs
from datetime import datetime


class Data:

    CHARSET_SIZE = 3755
    IMAGE_SIZE = 64

    def __init__(self, data_dir, random_flip_up_down=False, random_brightness=False, random_contrast=False):

        self.random_flip_up_down = random_flip_up_down
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast

        truncate_path = data_dir + os.sep + ('%05d' % Data.CHARSET_SIZE)  # display number with 5 leading 0
        self.image_file_paths = []
        for root, sub_folder, image_file_names in os.walk(data_dir):
            if root < truncate_path:
                self.image_file_paths += [os.path.join(root, image_file_name) for image_file_name in image_file_names]
        random.shuffle(self.image_file_paths)
        # the labels are the name of directories converted to int
        # self.labels = {'00000', '00001', '00002', ...}
        self.labels = []
        for image_file_path in self.image_file_paths:
            # images_dir_name example : '00000', '00001', '00002'
            images_dir_name = image_file_path[len(data_dir) + 1:].split(os.sep)[0]
            self.labels.append(int(images_dir_name))
        print("self.labels size: %d" % len(self.labels))

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

    # converts image_file_paths and labels_tensor to tensors.
    # shuffle them
    def get_batch(self, batch_size, num_epochs=None, aug=False):
        # convert data into tensor
        images_tensor = tf.convert_to_tensor(self.image_file_paths, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)

        # Instead of loading the entire dataset in RAM, which is very RAM consuming,
        # slice the dataset in batch and read each batch one by one.
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        # Convert data to grey scale
        image = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(image, channels=1), tf.float32)
        if aug:
            images = self.augmentation(self, images)

        # standardize the image sizes.
        standard_size = tf.constant([Data.IMAGE_SIZE, Data.IMAGE_SIZE], dtype=tf.int32)
        images = tf.image.resize_images(images, standard_size)

        labels = input_queue[1]
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch

    def load_char_label_dico(filePath):

        print("Loading CharLabelMap ... ")
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