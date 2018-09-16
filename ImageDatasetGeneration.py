import os
import numpy as np
import cv2
import argparse
import random
import sys
import pickle
import struct
import codecs
import time;
import utils;

CHAR_LABEL_DICO_FILE_NAME = 'charIndexDicoFile.txt'
LABEL_CHAR_DICO_FILE_NAME ='labelCharMapFile.txt'

def read_gnt(file):

    while True:
        header = np.fromfile(file, dtype="uint8", count=10)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        image = np.fromfile(file, dtype='uint8', count=width * height).reshape((height, width))
        yield image, tag_code

def file_list(gnt_dir):
    return [os.path.join(gnt_dir, file_name) for file_name in os.listdir(gnt_dir)]


# returns a map . key = chinese character, value = index
#  {'一' :0, '丁' : 1, '七' : 2, ...}
def build_char_index_dictionary(dir):

    print("Building dictionary... ")
    start_time = time.time()
    dico_file = codecs.open(CHAR_LABEL_DICO_FILE_NAME, 'w', 'gb2312')
    files = file_list(dir)
    char_set = set()
    for file in files:
        f = open(file, 'r')
        for _, tag_code in read_gnt(f):
            uni = struct.pack('>H', tag_code).decode('gb2312')
            char_set.add(uni)

    #char_list = list(char_set)
    #cdict = dict(zip(sorted(char_list), range(len(char_list))))

    i = 0;
    character_index_dico = {}
    for char in char_set:
        dico_file.write(char + " " + str(i) + "\n")
        character_index_dico[char] = i
        i +=1
    dico_file.close()
    #execution_time = time.time() - start_time
    print("Total %s characters. Execution time: %d s." % (str(len(character_index_dico)), time.time() - start_time))
    return character_index_dico


def build_label_char_map_file(cdict):

    print("Building label_char_map ... ")
    start_time = time.time()
    file = codecs.open(LABEL_CHAR_DICO_FILE_NAME, 'w', 'gb2312')
    for char in cdict.keys():
        label = cdict[char]
        # print(label)
    file.write(str(label) + " " + char + "\n")
    file.close()
    print("Execution time: %d s." %  (time.time() - start_time))

#   all the character images contained in one gnt file and put each extracted
# image into its corresponding directory.
def extractor(in_dir, out_dir, char_dictionary):

    start_time = time.time()
    i = 0
    for file_name in file_list(in_dir):
        f = open(file_name, "r")
        for image, tag_code in read_gnt(f):
            i += 1
            tag_code_uni = struct.pack('>H', tag_code).decode('gb2312') # chinese character
            character_dir = out_dir + "/" + '%0.5d' % char_dictionary[tag_code_uni]
            # character_dir examples : '00000', '00001', '00002'...
            # character_dir is a dir that contains all the 240 images of a given character
            if os.path.isdir(character_dir):
                pass
            else:
                os.mkdir(character_dir)
            image_name =  str(i) + ".png"
            cv2.imwrite(character_dir + '/' + image_name, image)
        f.close()
    print("Execution time: %d s." %  time.time() - start_time)
    return i

def loadLabelCharMap():

    labelCharMap ={}
    with codecs.open('labelCharMapFile.txt', 'r', 'gb2312') as f:
        for line in f:
            lineWithoutCR = line.split("\n")[0]
            splitted = lineWithoutCR.split(" ")
            label = splitted[0]
            char =  splitted[1]
            labelCharMap[label]=char
    return labelCharMap

def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gnt-dir-training", type=str, help="Path to training gnt dataset",
                        default=os.path.abspath("C:\DATA\PROJECTS\CASIA\OFFLINE\HWDB1.1trn_gnt"))

    parser.add_argument("--gnt-dir-test", type=str, help="Path to test gnt dataset",
                        default=os.path.abspath("C:\DATA\PROJECTS\CASIA\OFFLINE\HWDB1.1tst_gnt"))
    parser.add_argument("--output-dir", type=str, help="Path to images dataset",
                        default=os.path.abspath("C:\TEMP_GENERATED_DATASET"))
    parser.add_argument("--dict", type=str, default="char_dict")
    return parser.parse_args()

def main(args):

    labelCharMap = loadLabelCharMap()
    s= u'角'
    hexadecimal = s.encode('unicode-escape')
    integ = ord(s)
    chargb2312Bytes = s.encode('gb2312')

    # trn_path = os.path.join(args.gnt_dir, "trn")
    training_path = args.gnt_dir_training
    # tst_path = os.path.join(args.gnt_dir, "tst")
    test_path = args.gnt_dir_test
    training_out_path = os.path.join(args.output_dir, "trn")
    test_out_path = os.path.join(args.output_dir, "tst")

    #char_dictionary = build_char_index_dictionary(training_path)
    char_dictionary = utils.loadCharLabelMap(CHAR_LABEL_DICO_FILE_NAME)

    build_label_char_map_file(char_dictionary)

    print("Extracting training images.. ")
    training_imgs = extractor(training_path, training_out_path, char_dictionary)
    print("Total " + str(training_imgs) + " images in training set.")

    print("Extracting test images.. ")
    #test_imgs = extractor(test_path, test_out_path, char_dictionary)
    #print("Total " + str(test_imgs) + " images in test set.")

if __name__ == '__main__':
    main(arg_parser())
