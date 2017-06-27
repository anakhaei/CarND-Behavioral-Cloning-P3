
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

IMAGE_W = 80
IMAGE_H = 80
STEERING_CORRECTION = 0.3


def open_image(img_path):
    """
    1- Read a image from a file 
    2- Convert it to RGB
    """
    img = cv2.imread(img_path)
    #print(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img)

def preprocess_image(img):
    """
    1- Crop image
    2- Resize the image
    """
    #img = img[50:140, 0:320]
    #img = cv2.resize(img, (IMAGE_W, IMAGE_H))
    return img


def augment_data(img, steering):
    """
    1- flip image
    2- negetive steering
    """
    return np.fliplr(img), -steering


def load_data(folder_path):
    lines = []
    with open(folder_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

        images = []
        measurements = []
        for line in lines:
            source_path_center = line[0]
            source_path_left = line[1]
            source_path_right = line[2]
            filename_center = source_path_center.split('/')[-1]
            filename_left = source_path_left.split('/')[-1]
            filename_right = source_path_right.split('/')[-1]
            image_center = open_image(folder_path + 'IMG/' + filename_center)
            image_left = open_image(folder_path + 'IMG/' + filename_left)
            image_right = open_image(folder_path + 'IMG/' + filename_right)
            measurement_c = float(line[3])
            measurement_l = measurement_c + STEERING_CORRECTION
            measurement_r = measurement_c - STEERING_CORRECTION
            images.append(image_center)
            measurements.append(measurement_c)
            images.append(image_left)
            measurements.append(measurement_l)
            images.append(image_right)
            measurements.append(measurement_r)
            images.append(np.fliplr(image_center))
            measurements.append(-measurement_c)
            images.append(np.fliplr(image_left))
            measurements.append(-measurement_l)
            images.append(np.fliplr(image_right))
            measurements.append(-measurement_r)

    return images, measurements


def create_nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model
