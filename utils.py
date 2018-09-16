from PIL import Image
import numpy as np
import codecs
import cv2
import time;

# image is a png image
# returns a png image
def crop_image(image, tolerance=255):
    data = np.array(image)
    gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    mask = gray_image < tolerance
    index = np.ix_(mask.any(1), mask.any(0))
    cropped_image_data = gray_image[index]
    return Image.fromarray(cropped_image_data)


def convert_png_to_jpg(image_file_name) :
    im = Image.open(image_file_name + "png")
    rgb_im = im.convert('RGB')
    rgb_im.save(image_file_name+'.jpg')


# image_file_name is a png image
def set_image_background_to_white(image):
    data = np.array(image)
    alpha1 = 0  # Original value
    r2, g2, b2, alpha2 = 255, 255, 255, 255  # Value that we want to replace it with
    red, green, blue, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    mask = (alpha == alpha1)
    data[:, :, :4][mask] = [r2, g2, b2, alpha2]
    img = Image.fromarray(data)
    return img


def resize_image(image, width, height):
    resized_image = Image.open(image).convert('L')
    resized_image = resized_image.resize((width, height), Image.ANTIALIAS)
    resized_image = np.asarray(resized_image) / 255.0
    resized_image = resized_image.reshape([-1, 64, 64, 1])
    return resized_image


def loadLabelCharMap(filePath):

    print("Loading LabelCharMap ... ")
    start_time = time.time()
    labelCharMap = {}
    with codecs.open(filePath, 'r', 'gb2312') as f:
        for line in f:
            lineWithoutCR = line.split("\n")[0]
            splitted = lineWithoutCR.split(" ")
            label = splitted[0]
            char = splitted[1]
            labelCharMap[label] = char
    print("Execution time: %s s." % str(time.time() - start_time))
    return labelCharMap

def loadCharLabelMap(filePath):

    print("Loading CharLabelMap ... ")
    start_time = time.time()
    charLabelMap = {}
    with codecs.open(filePath, 'r', 'gb2312') as f:
        for line in f:
            lineWithoutCR = line.split("\n")[0]
            splitted = lineWithoutCR.split(" ")
            char = splitted[0]
            label = int(splitted[1])
            charLabelMap[char] = label
    print("Execution time: %s s." % str(time.time() - start_time))
    return charLabelMap
