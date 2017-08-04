import csv
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", type=str, default='./data',
	help="(optional) folder where data images are located")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-i", "--offset", type=int, default=1,
	help="(optional) read images with offset, convenient for debugging")
ap.add_argument("-e", "--epochs", type=int, default=2,
	help="(optional) read images with offset, convenient for debugging")
args = vars(ap.parse_args())

DATA_FOLDER = args['data']
offset = args["offset"] # for DEBUG purposes

def load_logs():
    global DATA_FOLDER
    with open(DATA_FOLDER + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        return [line for line in reader]

def preprocess(image):
    return image

def load_image(source_path):
    image = cv2.imread(source_path)
    if image is None:
        raise Exception("Probably you've got broken dataset")
    return preprocess(image)

STEERING_CORRECTION = 0.2

"""Load images from DATA_FOLDER"""
def load_data(lines):
    images, measurements = [],[]
    correction = STEERING_CORRECTION
    for line in lines[::offset]:
        for i in range(3):
            image = load_image(line[i])
            images.append(image)
        measurement = float(line[3])
        measurements += [measurement,
                         measurement + STEERING_CORRECTION,
                         measurement - STEERING_CORRECTION];
    return images, measurements

"""Augment images: flip them horizontally"""
def run_augmentation(images, measurements):
    aug_images = []
    aug_measurements = []
    for image, measurement in zip(images, measurements):
        flipped_image = cv2.flip(image, 0)
        flipped_measurement = -measurement
        aug_images.append(flipped_image)
        aug_measurements.append(flipped_measurement)
    images += aug_images
    measurements += aug_measurements

print("Preparing data...")
lines = load_logs()
images,measurements = load_data(lines)
run_augmentation(images, measurements)


"""Setup training data"""
X_train = np.array(images)
y_train = np.array(measurements)
EPOCHS = args["epochs"]

print("Done. {} images loaded".format(X_train.shape[0]))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
"""Create our model"""
"""LeNet architecture"""
model = Sequential()
model.add(Lambda(lambda x: x/255. - .5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(24,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=.2,
          shuffle=True, nb_epoch=EPOCHS,
          callbacks=[
            # EarlyStopping(min_delta=0.0001),
            # TensorBoard(),
          ])


import os
import datetime

model_name = "model-{}.h5".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
model.save(model_name)
print('Model saved to ' + model_name)
symlink_name = 'model.h5'
try:
    os.remove(symlink_name)
except OSError:
    pass
os.symlink(model_name, symlink_name)
